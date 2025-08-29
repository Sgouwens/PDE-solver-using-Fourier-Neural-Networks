# Load required libraries and functions
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from neuralop.models import FNO, FNO_modified
from datetime import datetime
from tqdm import tqdm
import time
from utils.generate_data import generate_pde_solutions, generate_pde_solutions_old

from math import log
import argparse

import wandb
from dotenv import load_dotenv
load_dotenv()


wandb_api_key = os.getenv("WANDB_API_KEY")
if wandb_api_key:
    wandb.login(key=wandb_api_key)
    print("W&B login successful!")
else:
    print("WANDB_API_KEY not found in environment variables.")

class PDE_FNO_Solver:
    """
    A class to train a Fourier Neural Operator (FNO) model to solve PDEs. This class handles data generation, model initialization, training, and saving the trained model parameters.
    """
    def __init__(self, config):
        """
        Initializes the PDE_FNO_Solver with a given configuration.

        Args:
            config (dict): A dictionary containing all necessary parameters for the model
                           and training process.
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._initialize_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        self.criterion = nn.MSELoss()
        
        # Ensure save directory exists
        os.makedirs(self.config['save_dir'], exist_ok=True)
        
        # Load model weights if they exist
        self._load_model_weights()
        print(f"Number of parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")

    def _initialize_model(self):
        """Initializes and returns the FNO model based on the configuration."""
        if self.config.get('use_modified_fno', False):
            model = FNO_modified(
                in_channels=1,
                out_channels=1,
                input_param_dim=self.config.get('input_param_dim', 1),
                hidden_param_dim=self.config.get('hidden_param_dim', 16),
                n_modes=(self.config['modes_x'], self.config['modes_y']),
                hidden_channels=self.config['width'],
                num_layers=self.config['num_layers'],
                activation=self.config['activation']
            ).to(self.device)
        else:
            model = FNO(
                in_channels=1,
                out_channels=1,
                n_modes=(self.config['modes_x'], self.config['modes_y']),
                hidden_channels=self.config['width'],
                num_layers=self.config['num_layers'],
                activation=self.config['activation']
            ).to(self.device)
        return model

    def _load_model_weights(self):
        """Loads a pre-trained model from the specified path if it exists."""

        return None
        model_path = os.path.join(self.config['save_dir'], self.config['model_file_name'])
        if os.path.exists(model_path):
            print(f"Loading pre-trained model from {model_path}...")
            self.model.load_state_dict(torch.load(model_path, weights_only=False, map_location=self.device))
        else:
            print(f"No pre-trained model found at {model_path}. Training from scratch.")

    def load_data(self, nu_fixed=None):
        """Generates and loads training data."""
        print("Generating data...")
        
        if nu_fixed is not None:
            input_train, output_train, pars_train = generate_pde_solutions_old(
                self.config['num_simulations'], self.config['frames'], set_seed=1, nu_par=nu_fixed)
            
            input_test, output_test, pars_test = generate_pde_solutions_old(
                128, self.config['frames'], testing=True, nu_par=nu_fixed)
        else:
            input_train, output_train, pars_train = generate_pde_solutions(
                self.config['num_simulations'], self.config['frames'], set_seed=1)
            
            input_test, output_test, pars_test = generate_pde_solutions(
                128, self.config['frames'], testing=True)

        # Add channel dimension if necessary
        input_train = input_train.unsqueeze(1)
        output_train = output_train.unsqueeze(1)

        input_test = input_test.unsqueeze(1)
        output_test = output_test.unsqueeze(1)
        
        dataset_train = TensorDataset(input_train, output_train, pars_train)
        dataset_test = TensorDataset(input_test, output_test, pars_test)

        return (DataLoader(dataset_train, batch_size=self.config['batch_size'], shuffle=True),
                DataLoader(dataset_test,  batch_size=self.config['batch_size'], shuffle=False))
        
    def train_epoch(self, train_loader, test_loader, scheduler, epoch_num):
        
        self.model.train()
        running_training_loss = 0.0
        running_testing_loss = 0.0
        start_time = time.time()
        
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch_num+1}/{self.config['epochs']}", leave=True)

        for inputs, targets, param in train_loader_tqdm:
            inputs, targets, param = inputs.to(self.device), targets.to(self.device), param.to(self.device)
            
            if self.config.get('use_modified_fno', False):
                param = param.unsqueeze(1)

            self.optimizer.zero_grad()
            outputs = self.model(inputs, param) if self.config.get('use_modified_fno', False) else self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            scheduler.step()

            running_training_loss += loss.item()
            train_loader_tqdm.set_postfix(loss=f"{loss.item():.7f}")

        # Evaluate test error
        self.model.eval()
        for inputs, targets, param in test_loader:
            inputs, targets, param = inputs.to(self.device), targets.to(self.device), param.to(self.device)
            if self.config.get('use_modified_fno', False):
                param = param.unsqueeze(1)

            outputs = self.model(inputs, param) if self.config.get('use_modified_fno', False) else self.model(inputs)
            loss = self.criterion(outputs, targets)
            running_testing_loss += loss.item()

        avg_train_loss = running_training_loss / len(train_loader)
        avg_test_loss = running_testing_loss / len(test_loader)

        epoch_duration = time.time() - start_time

        return avg_train_loss, avg_test_loss, epoch_duration

    def train(self, train_loader, test_loader):
        """Runs the training loop for the specified number of epochs."""
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, 
            max_lr=self.config['learning_rate'],
            steps_per_epoch=len(train_loader), 
            epochs=self.config['epochs']
        )
        
        # Using wandb tracking
        run = wandb.init(
            project="Darcy Flow FNO",
            notes="NA",
            config=self.config
        )
    
        train_losses = []
        test_losses = []
        
        for epoch in range(self.config['epochs']):
            
            avg_train_loss, avg_test_loss, epoch_duration = self.train_epoch(train_loader, test_loader, scheduler, epoch_num=epoch)

            train_losses.append(avg_train_loss)
            test_losses.append(avg_test_loss)
            
            print(f"Epoch {epoch+1}/{self.config['epochs']}, "
                  f"Train Loss: {avg_train_loss:.7f}, "
                  f"Test Loss: {avg_test_loss:.7f} sec, "
                  f"Epoch Time: {epoch_duration:.2f} sec")
        
            wandb.log({
                "train_loss": min(avg_train_loss, 0.001),
                "test_loss": min(avg_test_loss, 0.001),
                "epoch_duration": epoch_duration,
                "train_logloss": log(avg_train_loss),
                "test_logloss": log(avg_test_loss),
            })

        self.save_model()

        wandb.run.summary["final_train_loss"] = train_losses[-1]
        wandb.run.summary["final_test_loss"] = test_losses[-1]
        wandb.finish()

        return train_losses
    

    def save_model(self):
        """Saves the trained model's state dictionary."""
        model_path = os.path.join(self.config['save_dir'], self.config['model_file_name'])
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

# Usage
def main():
    parser = argparse.ArgumentParser(description="Train an FNO model for PDE solving.")
    
    parser.add_argument('--modes-x', type=int, default=16, help='Number of Fourier modes in x.')
    parser.add_argument('--modes-y', type=int, default=16, help='Number of Fourier modes in y.')
    parser.add_argument('--width', type=int, default=64, help='Width of the FNO model.')
    parser.add_argument('--num-layers', type=int, default=4, help='Number of FNO layers.')
    parser.add_argument('--activation', type=str, default='gelu', help='Activation function.')
    parser.add_argument('--batch-size', type=int, default=128, help='Training batch size.')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--num-simulations', type=int, default=2000, help='Number of simulations to generate.')
    parser.add_argument('--frames', type=int, default=18, help='Number of frames per simulation.')
    parser.add_argument('--save-dir', type=str, default="saved_simulations", help='Directory to save the model.')
    
    parser.add_argument('--use-modified-fno', action='store_true', 
                        help='Use FNO_modified with parameter inputs (if present, sets to True).')    
    parser.add_argument('--input-param-dim', type=int, default=1, 
                        help='Input parameter dimension for FNO_modified.')
    parser.add_argument('--hidden-param-dim', type=int, default=16, 
                        help='Hidden parameter dimension for FNO_modified.')
    
    # Parse the arguments provided via the command line
    # Convert the parsed arguments (Namespace object) into a dictionary
    # This creates the 'config' dictionary dynamically based on command-line input.
    args = parser.parse_args()

    config = vars(args)
    
    # You might want to add model_file_name here based on parsed arguments
    # For example, to make it unique for each run:
    config['model_file_name'] = (
        f"FNO_model_modes{config['modes_x']}x{config['modes_y']}_"
        f"width{config['width']}_layers{config['num_layers']}"
        f"{'_modified' if config['use_modified_fno'] else ''}.pt"
    )

    # Initialize and train the solver with the dynamically created config
    solver = PDE_FNO_Solver(config)
    
    # Load data based on whether modified FNO is used (which implies varying nu)
    # If use_modified_fno is False, you might want a default nu_value for generate_pde_solutions_old
    nu_value = 0.02 # Example fixed nu, adjust as needed or make it an argparse argument too
    train_loader, test_loader = solver.load_data(nu_fixed=nu_value if not config['use_modified_fno'] else None)
    
    solver.train(train_loader, test_loader)

if __name__ == '__main__':
    main()