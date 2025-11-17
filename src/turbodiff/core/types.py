"""
Type definitions for TurboDiff.

All operations use JAX arrays for automatic differentiation
and GPU acceleration.
"""

from typing import Tuple, Union
import jax.numpy as jnp

# JAX array type
Array = jnp.ndarray

# Common type aliases
Shape = Tuple[int, ...]
Scalar = Union[int, float]


__all__ = [
    "Array",
    "Shape",
    "Scalar",
]
