"""
elasticity_parameter_inference

Bayesian inference of material parameters (E, nu) in a 2D linear elasticity
model of a thin plate under tension.

This package provides:
- A forward elasticity solver mapping (E, nu) -> displacement field
- Observation operators extracting DIC-style displacement data
- Bayesian inference utilities (posterior grid + importance sampling)
"""
from forwardPDESolver import solve_elasticity

__all__ = [
    "solve_elasticity",
]

__version__ = "1.0.0"