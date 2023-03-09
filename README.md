

# Diffusions.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://murrellb.github.io/Diffusions.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://murrellb.github.io/Diffusions.jl/dev/)
<!--- [![Build Status](https://github.com/murrellb/Diffusions.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/murrellb/Diffusions.jl/actions/workflows/CI.yml?query=branch%3Amain)--->
[![Coverage](https://codecov.io/gh/murrellb/Diffusions.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/murrellb/Diffusions.jl)

### Overview

This package aims to simplify the construction of diffusion denoising models by providing a standardized approach, based on endpoint-conditioned diffusion "bridge" sampling. We provide a number of processes to handle continuous Euclidean variables, discrete variables, angles, and rotations, and a simplified framework and interface to construct new diffusions over custom spaces.

The approach requires the user to build a model (typically a neural network) that takes as input a noised observation $x_t$, and predicts (in some sense - see below) the denoised observation $x_0$. The typical training setup involves a large set of samples from some target distribution, which are noised under the selected diffusion process, and the model is trained to minimize a loss, predicting the denoised observation.

### Spaces and loss functions

For the standard continuous Euclidean case, the least-squares loss allows a trained model to recover the expectation $E[x_0|x_t]$. With a perfect model, this would provide exact sampling of the target distribution when the diffusion process is run in reverse. For trickier manifolds, the exact target distribution is not exactly recovered by this approach, but can be approximated well-enough when a suitable loss function is chosen.

The discrete case, where an observation is a collection of discrete variables, is slightly different. Instead of providing a point estimate of some expectation, the cross-entropy loss must be used, and a sample from the resulting categorical distribution taken for each variable. Even though these samples are taken independently, the reverse diffusion can recover the joint distribution with a sufficiently capable model.

Importantly, multiple different kinds of diffusion can be combined, and jointly diffused and reconstructed together.

### Visualization

The inverse diffusion works by using the trained model to predict $\hat{x_0} = E[x_0|x_t]$, and then constructs an endpoint-conditioned sample along a stochastic bridge between $\hat{x_0}$ and $x_t$, at $x_{t'}$, from $P(x_{t'}|\hat{x_0},x_t,D)$, where $D$ is the diffusion process itself and $0 < t' < t$. This is visualized below, for a single variable under Ornstein–Uhlenbeck diffusion, showing multiple steps (with an exaggerated step size), clarifying how the randomness injected with each "bridge" influences the final sample:

<img src="https://user-images.githubusercontent.com/1152087/224138792-15dd802b-3e23-43bc-8e11-e332ce41d7a9.svg" width="650" background-color="white">

Even though each variable is diffused independently, for extremely strongly correlated variables the joint distribution can be recovered:

<img src="https://user-images.githubusercontent.com/1152087/224140026-870a42bb-94bf-48be-86ce-a639b44cd199.gif" width="450">

### Example

```julia
#Simplest example here - coming soon.
```

### Related work

To do.

### Manuscript

Coming soon.
