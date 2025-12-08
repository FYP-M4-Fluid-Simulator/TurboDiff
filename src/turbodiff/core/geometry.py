"""
Differentiable geometry primitives for shape optimization.

This module provides simple, JAX-compatible geometric shapes
represented as Signed Distance Functions (SDFs).
"""

import jax
import jax.numpy as jnp
from turbodiff.core.types import Array
from typing import Callable


@jax.jit
def sdf_circle(
    x: Array, y: Array, center_x: float, center_y: float, radius: float
) -> Array:
    """
    Signed distance function for a circle.

    Args:
        x, y: Grid coordinates
        center_x, center_y: Circle center
        radius: Circle radius

    Returns:
        SDF values: negative inside, positive outside, zero on boundary
    """
    dx = x - center_x
    dy = y - center_y
    return jnp.sqrt(dx**2 + dy**2) - radius


@jax.jit
def sdf_box(
    x: Array, y: Array, center_x: float, center_y: float, width: float, height: float
) -> Array:
    """
    Signed distance function for an axis-aligned rectangle.

    Uses implicit function formulation for gradient stability.
    Avoids division by parameters to prevent NaN gradients during optimization.

    Args:
        x, y: Grid coordinates
        center_x, center_y: Box center
        width, height: Box dimensions

    Returns:
        Implicit function values (negative inside, positive outside)
    """
    # Box local coordinates
    dx = jnp.abs(x - center_x)
    dy = jnp.abs(y - center_y)

    # Implicit formulation: box contains point if 2*|dx| < width AND 2*|dy| < height
    implicit_x = 2.0 * dx - width
    implicit_y = 2.0 * dy - height
    implicit_val = jnp.maximum(implicit_x, implicit_y)

    return implicit_val


@jax.jit
def sdf_ellipse(
    x: Array,
    y: Array,
    center_x: float,
    center_y: float,
    a: float,
    b: float,
    angle: float = 0.0,
) -> Array:
    """
    Signed distance function for an ellipse.

    Uses implicit function formulation for gradient stability.
    Avoids division by parameters to prevent NaN gradients during optimization.

    Args:
        x, y: Grid coordinates
        center_x, center_y: Ellipse center
        a, b: Semi-major and semi-minor axes
        angle: Rotation angle in radians

    Returns:
        Implicit function values (negative inside, positive outside)
    """
    # Translate and rotate to ellipse frame
    dx = x - center_x
    dy = y - center_y
    cos_a = jnp.cos(angle)
    sin_a = jnp.sin(angle)
    dx_rot = dx * cos_a + dy * sin_a
    dy_rot = -dx * sin_a + dy * cos_a

    # Implicit ellipse equation: b²x² + a²y² - a²b² = 0
    # Reformulated to avoid dividing by parameters a and b
    a_sq = a**2
    b_sq = b**2
    implicit_val = b_sq * dx_rot**2 + a_sq * dy_rot**2 - a_sq * b_sq

    # Scale for consistent magnitude across different ellipse sizes
    scale = (a + b) / 2.0
    return implicit_val / scale


# ============================================================================
# Simple Parametric Shape (for testing)
# ============================================================================


def sdf_parametric_blob(
    x: Array, y: Array, center_x: float, center_y: float, radius: float, params: Array
) -> Array:
    """
    Parametric blob shape with Fourier modulation.

    Creates a deformable shape by modulating a circle's radius with Fourier modes.
    Useful for testing shape optimization with multiple parameters.

    Args:
        x, y: Grid coordinates
        center_x, center_y: Blob center
        radius: Base radius
        params: Fourier coefficients, shape (n_modes,)

    Returns:
        SDF values (negative inside, positive outside)
    """
    # Convert to polar coordinates
    dx = x - center_x
    dy = y - center_y
    r = jnp.sqrt(dx**2 + dy**2)
    theta = jnp.arctan2(dy, dx)

    # Modulate radius with Fourier series
    r_modulated = radius
    for i, coef in enumerate(params):
        mode = i + 1  # Start from mode 1
        r_modulated = r_modulated + coef * jnp.cos(mode * theta) * radius

    return r - r_modulated


# ============================================================================
# Helper: Create SDF function for grid
# ============================================================================


def create_sdf_function(
    sdf_type: str = "circle", **kwargs
) -> Callable[[Array, Array], Array]:
    """
    Create an SDF function with fixed parameters.

    Args:
        sdf_type: Type of SDF ("circle", "box", "ellipse", "blob")
        **kwargs: Parameters for the SDF

    Returns:
        Function (x, y) -> sdf_values
    """
    if sdf_type == "circle":
        return lambda x, y: sdf_circle(x, y, **kwargs)
    elif sdf_type == "box":
        return lambda x, y: sdf_box(x, y, **kwargs)
    elif sdf_type == "ellipse":
        return lambda x, y: sdf_ellipse(x, y, **kwargs)
    elif sdf_type == "blob":
        return lambda x, y: sdf_parametric_blob(x, y, **kwargs)
    else:
        raise ValueError(f"Unknown SDF type: {sdf_type}")


__all__ = [
    "sdf_circle",
    "sdf_box",
    "sdf_ellipse",
    "sdf_parametric_blob",
    "create_sdf_function",
]
