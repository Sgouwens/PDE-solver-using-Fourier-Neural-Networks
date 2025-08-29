# PDE-solver-using-Fourier-Neural-Networks


# Learning Parametric PDEs with Fourier Neural Operators

This repository contains work-in-progress on using Fourier Neural Operators (FNOs) to learn solutions to parametrized partial differential equations (PDEs), with a focus on the Darcy Flow problem.

Traditional numerical methods for solving PDEs are often computationally expensive and can require complex, problem-specific implementations. In contrast, neural operator methods such as the **Fourier Neural Operator (Li et al., 2021)** offer a way to learn mappings between function spaces, effectively modeling the solution operator of a PDE directly from data. Once trained, FNOs offer orders-of-magnitude faster inference compared to classical solvers, making them especially suitable for applications where repeated evaluations are required, such as uncertainty quantification, optimization, or real-time forecasting.

Before exploring FNOs, we tested multiple alternative methods:

* **Residual CNNs**: A convolutional architecture with residual connections was used to retain spatial dimensions. Residual layers align well with the form of many time-stepping schemes: $u_{t+1}(x,y) = u_t(x,y) + f(t, u_t(x,y)$. While this model was effective, it often learned a near-identity mapping — especially for small time steps where $u_{t+1}\approx u_t$., leading to low MSE even without meaningful learning.
** **Latent Dynamics via VAE + Flow Matching**: A variational autoencoder was used to embed the image time-series into a lower-dimensional latent space. A flow-matching model was then trained to learn the dynamics in this space. This setup showed promise but was limited by reconstruction errors and sensitivity to hyperparameters, leading to reduced accuracy and multiple potential failure points.

## Overview

We aim to learn the solution operator of a parametrized PDE — in this case, the Darcy Flow equation with a spatially-varying diffusion coefficient. A simulation of this PDE produces a spatiotemporal field, denoted by $u_t(x,y)$, where $t$ represents time and $(x,y)$ denotes spatial position. These simulations can be interpreted as **time-series of images**, each representing the state of the field at a given time. Because the PDE is parametrized (e.g., by a diffusion coefficient field), we include this additional information as input  to the neural network. A single input instance to the FNO consists of a tensor of shape: (channels, height, width), where channels can include: the current state $u_{t_0}(x,y)$ the diffusion coefficient field (or other PDE parameters), and optionally, learned embeddings or preprocessed features.

## Model and Implementation

We use the Fourier Neural Operator (FNO) architecture as described by Li et al. (2021). The key idea is to learn the PDE solution operator in Fourier space, enabling:

* Fast convolution in the spectral domain,

* Parameter sharing across spatial dimensions,

* Interpolation to arbitrary spatial resolutions due to the continuous nature of the operator.

We adapt the base FNO implementation to accommodate parametrized input by concatenating auxiliary channels representing the PDE parameters. This requires minor changes to the data pipeline and model architecture.

Most training is performed on a GPU. Multiple configurations (grid size, parameter ranges, etc.) are tested to evaluate the generalization ability of the model.

## Broader Context

This work is inspired by recent applications of FNOs to real-world systems. Notably, NASA's PritiWxC foundational model used satellite imagery time-series and FNO variants to build fast, accurate weather forecasting models.

## References

Li, Zongyi, et al. "Fourier Neural Operator for Parametric Partial Differential Equations." International Conference on Learning Representations (ICLR), 2021. arXiv:2010.08895
Schmude, Roy et al. "Prithvi WxC: Foundation Model for Weather and Climate", 2024. arXiv:2409.13598
