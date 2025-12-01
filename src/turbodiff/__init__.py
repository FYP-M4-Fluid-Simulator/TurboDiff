"""
TurboDiff - Differentiable 2D Fluid Simulator

A fluid simulation library using JAX for
automatic differentiation and GPU acceleration.
"""

# Configure JAX
from . import _jax_config  # noqa: F401

from .core.fluid_grid_jax import FluidGrid, FluidState
from .core.grids import Grid, StaggeredGrid
from .core.types import Array
from .core import initial_conditions

__version__ = "0.1.0"

__all__ = [
    # JAX-based classes (recommended)
    "FluidGrid",
    "FluidState",
    "Grid",
    "StaggeredGrid",
    "Array",
    "initial_conditions",
    # Legacy classes (deprecated)
    "FluidGridLegacy",
    "FluidCell",
]
