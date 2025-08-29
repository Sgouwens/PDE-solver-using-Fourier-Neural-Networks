import torch
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

def central_diff_2d(f, h):
    dx, dy = h
    
    # Initialize derivative tensors with zeros, which handles boundaries
    # The central difference formula applies to interior points.
    u_x = torch.zeros_like(f)
    u_y = torch.zeros_like(f)

    # Compute derivative along x-axis (columns)
    # u_x[i, j] = (f[i, j+1] - f[i, j-1]) / (2 * dx)
    # The slicing [:, 1:-1] ensures we only compute for internal columns.
    u_x[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / (2 * dx)

    # Compute derivative along y-axis (rows)
    # u_y[i, j] = (f[i+1, j] - f[i-1, j]) / (2 * dy)
    # The slicing [1:-1, :] ensures we only compute for internal rows.
    u_y[1:-1, :] = (f[2:, :] - f[:-2, :]) / (2 * dy)

    return u_x, u_y

def source_term(X, Y, t):
    return 0 

def generate_pde_solutions(number_of_initial_states: int, number_of_timesteps: int = 1,
                           set_seed=None, testing=False):
    """
    Generates pairs of (input_state, output_state) for PDE training data.
    Each pair represents (u_t, u_{t+1}) for a given PDE simulation.

    Args:
        number_of_initial_states (int): The number of different initial
                                         conditions to sample and simulate.
        number_of_timesteps (int, optional): The number of timesteps to compute
                                             for each initial condition.
                                             Defaults to 1. If N, it generates
                                             N pairs (u_0, u_1), ..., (u_{N-1}, u_N)
                                             for each initial state.

    Returns:
        tuple: A tuple containing two torch.Tensor objects:
               - input_tensor: A tensor of shape (N_pairs, nx, ny)
                               containing the states at time t.
               - output_tensor: A tensor of shape (N_pairs, nx, ny)
                                containing the states at time t+dt.
               All tensors are moved to CPU before returning.
    """
    # Simulation parameters
    Lx, Ly = 2.0, 2.0   # Domain lengths
    nx, ny = 64, 64   # Grid resolution
    dt = 0.005  # Time step
    # nu = 0.02   # Diffusion coefficient ##########################################################

    g = None
    if testing:
        set_seed=12321
    if testing or set_seed:
        g = torch.Generator(device=DEVICE)
        g.manual_seed(set_seed)

    # nu = 0.01 + torch.rand(number_of_initial_states, generator=g, device=DEVICE) * 0.07
    nu = 0.04 + torch.randn(number_of_initial_states, generator=g, device=DEVICE)/1000

    cx, cy = 1.0, -1.0  # Advection speeds

    # Create grid coordinates (fixed for all simulations)
    X = torch.linspace(0, Lx, nx, device=DEVICE).repeat(ny, 1).T
    Y = torch.linspace(0, Ly, ny, device=DEVICE).repeat(nx, 1)
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)

    # Lists to store all input and output state pairs
    all_input_states = []
    all_output_states = []
    all_parameter_sets = []

    # Loop through the desired number of diverse initial states
    for i in range(number_of_initial_states):
        # Sum of random positive Gaussian bumps + a base constant.
        u = torch.full((nx, ny), 0.5, device=DEVICE)

        # Add a random number of Gaussian bumps
        num_bumps = torch.randint(2, 10, size=(1,), generator=g, device=DEVICE).item()
        for _ in range(num_bumps):
            amplitude = torch.empty(1, device=DEVICE).uniform_(0.5, 2.0, generator=g)
            center_x = torch.empty(1, device=DEVICE).uniform_(0.2 * Lx, 0.8 * Lx, generator=g)
            center_y = torch.empty(1, device=DEVICE).uniform_(0.2 * Ly, 0.8 * Ly, generator=g)
            sigma    = torch.empty(1, device=DEVICE).uniform_(0.05, 0.2, generator=g)

            u += amplitude * torch.exp(-((X - center_x)**2 + (Y - center_y)**2) / (2 * sigma**2))
        
        u = torch.relu(u) + 0.01

        # Initialize the current state and time for this specific simulation
        current_u = u.clone()
        current_t = torch.tensor(0.0, device=DEVICE)

        # Simulate the PDE
        for _ in range(number_of_timesteps):
            input_state = current_u.clone()

            # u_x = du/dx, u_y = du/dy
            u_x, u_y = central_diff_2d(current_u, [dx, dy])

            # u_xx = d/dx(du/dx), u_yy = d/dy(du/dy)
            u_xx, _ = central_diff_2d(u_x, [dx, dy])
            _, u_yy = central_diff_2d(u_y, [dx, dy])

            # Evolve one step in time using Euler's method
            next_u = current_u + dt * (
                - cx * u_x
                - cy * u_y
                + nu[i] * (u_xx + u_yy)
                + source_term(X, Y, current_t)
            )
            
            # Advance the simulation time for the next iteration
            current_t += dt
            output_state = next_u.clone()

            all_input_states.append(input_state)
            all_output_states.append(output_state)
            all_parameter_sets.append(nu[i])

            current_u = next_u.clone()

    # Stack all collected states into single tensors
    input_tensor = torch.stack(all_input_states)
    output_tensor = torch.stack(all_output_states)
    param_tensor = torch.stack(all_parameter_sets)

    return input_tensor.cpu(), output_tensor.cpu(), param_tensor.cpu()



# Old model generation without changing parameters.
def generate_pde_solutions_old(number_of_initial_states: int, number_of_timesteps: int = 1,
                               set_seed=None, testing=False, nu_par: float = 0.04):
    """
    Generates pairs of (input_state, output_state) for PDE training data.
    Each pair represents (u_t, u_{t+1}) for a given PDE simulation.
    """
    # Simulation parameters
    Lx, Ly = 2.0, 2.0   # Domain lengths
    nx, ny = 64, 64   # Grid resolution
    dt = 0.005  # Time step
    # nu = 0.02   # Diffusion coefficient ##########################################################

    g = None
    if testing:
        set_seed=12321
    if testing or set_seed:
        g = torch.Generator()
        g.manual_seed(set_seed)
        
    nu = nu_par * torch.randn(number_of_initial_states) ** 0

    cx, cy = 1.0, -1.0  # Advection speeds

    # Create grid coordinates (fixed for all simulations)
    X = torch.linspace(0, Lx, nx, device=DEVICE).repeat(ny, 1).T
    Y = torch.linspace(0, Ly, ny, device=DEVICE).repeat(nx, 1)
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)

    # Lists to store all input and output state pairs
    all_input_states = []
    all_output_states = []
    all_parameter_sets = []

    # Loop through the desired number of diverse initial states
    for i in range(number_of_initial_states):
        # Sum of random positive Gaussian bumps + a base constant.
        u = torch.full((nx, ny), 0.5, device=DEVICE)

        # Add a random number of Gaussian bumps
        num_bumps = torch.randint(2, 10, size=(1,), generator=g, device=DEVICE).item()
        for _ in range(num_bumps):
            
            # Add the Gaussian bumps to the field
            amplitude = torch.empty(1, device=DEVICE).uniform_(0.5, 2.0, generator=g)
            center_x = torch.empty(1, device=DEVICE).uniform_(0.2 * Lx, 0.8 * Lx, generator=g)
            center_y = torch.empty(1, device=DEVICE).uniform_(0.2 * Ly, 0.8 * Ly, generator=g)
            sigma    = torch.empty(1, device=DEVICE).uniform_(0.05, 0.2, generator=g)

            u += amplitude * torch.exp(-((X - center_x)**2 + (Y - center_y)**2) / (2 * sigma**2))
        
        # Ensuring strict positivity by applying ReLU and adding a small epsilon.
        u = torch.relu(u) + 0.01

        # Initialize the current state and time for this specific simulation
        current_u = u.clone()
        current_t = torch.tensor(0.0, device=DEVICE)

        # Simulate the PDE
        for _ in range(number_of_timesteps):
            # Store the current state as an input to the model
            input_state = current_u.clone()

            # u_x = du/dx, u_y = du/dy
            u_x, u_y = central_diff_2d(current_u, [dx, dy])

            # u_xx = d/dx(du/dx), u_yy = d/dy(du/dy)
            u_xx, _ = central_diff_2d(u_x, [dx, dy])
            _, u_yy = central_diff_2d(u_y, [dx, dy])

            # Evolve one step in time using Euler's method
            next_u = current_u + dt * (
                - cx * u_x
                - cy * u_y
                + nu[i] * (u_xx + u_yy)
                + source_term(X, Y, current_t)
            )
            
            # Advance the simulation time for the next iteration
            current_t += dt
            output_state = next_u.clone()

            all_input_states.append(input_state)
            all_output_states.append(output_state)
            all_parameter_sets.append(nu[i])

            current_u = next_u.clone()

        # Store a tensor containing the diffusion parameter used for this specific simulation
        # diffusion_parameter = torch.full(size=(all_input_states.shape[],1), fill_value=nu[i])

    # Stack all collected states into single tensors
    input_tensor = torch.stack(all_input_states)
    output_tensor = torch.stack(all_output_states)
    param_tensor = torch.stack(all_parameter_sets)

    return input_tensor.cpu(), output_tensor.cpu(), param_tensor.cpu()