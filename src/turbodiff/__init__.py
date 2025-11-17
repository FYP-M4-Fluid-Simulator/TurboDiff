"""
TurboDiff - Differentiable 2D Fluid Simulator

A fluid simulation library using JAX for
automatic differentiation and GPU acceleration.
"""

# Configure JAX
from . import _jax_config  # noqa: F401

from .core.fluid_grid import FluidGrid
from .core.fluid_cell import FluidCell
from .core.types import Array

__version__ = "0.1.0"

__all__ = [
    "FluidGrid",
    "FluidCell",
    "Array",
]
