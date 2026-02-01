"""
Differentiable soft masking for obstacle generation.

This module provides sigmoid-based soft masks that enable gradient flow
through obstacle boundaries for shape optimization.
"""

import jax
import jax.numpy as jnp
from jax import Array


def soft_sigmoid_mask(distance: Array, sharpness: float = 50.0) -> Array:
    """
    Create a soft mask using sigmoid function.

    Converts signed distance values to smooth [0, 1] mask.
    Positive distance -> 0 (outside), Negative distance -> 1 (inside).

    Args:
        distance: Signed distance field (negative inside, positive outside)
        sharpness: Controls transition sharpness. Higher = sharper boundary.
                   50.0 is sharp enough for physics, soft enough for gradients.

    Returns:
        Soft mask with values in [0, 1]
    """
    # sigmoid(-distance * sharpness) gives 1 inside, 0 outside
    return jax.nn.sigmoid(-distance * sharpness)


def soft_mask_from_bounds(
    values: Array,
    lower_bound: float,
    upper_bound: float,
    sharpness: float = 50.0,
) -> Array:
    """
    Create mask for values within bounds: lower_bound <= values <= upper_bound.

    Args:
        values: Input values to check
        lower_bound: Lower bound
        upper_bound: Upper bound
        sharpness: Transition sharpness

    Returns:
        Mask: ~1 where values are in bounds, ~0 outside
    """
    mask_above_lower = jax.nn.sigmoid((values - lower_bound) * sharpness)
    mask_below_upper = jax.nn.sigmoid((upper_bound - values) * sharpness)
    return mask_above_lower * mask_below_upper


def create_airfoil_mask(
    grid_x: Array,
    grid_y: Array,
    y_upper_at_grid: Array,
    y_lower_at_grid: Array,
    x_min: float = 0.0,
    x_max: float = 1.0,
    sharpness: float = 50.0,
) -> Array:
    """
    Create differentiable airfoil obstacle mask.

    A point is inside the airfoil if:
    - y_lower <= y <= y_upper (between surfaces)
    - x_min <= x <= x_max (within chord)

    Args:
        grid_x: X-coordinates of grid points, shape (H, W)
        grid_y: Y-coordinates of grid points, shape (H, W)
        y_upper_at_grid: Upper surface y-values interpolated to grid, shape (H, W)
        y_lower_at_grid: Lower surface y-values interpolated to grid, shape (H, W)
        x_min: Chord start (typically 0)
        x_max: Chord end (typically 1)
        sharpness: Sigmoid sharpness for all boundaries

    Returns:
        Soft obstacle mask, shape (H, W), ~1 inside airfoil, ~0 outside
    """
    # "How much is this point below the upper surface?"
    mask_below_upper = jax.nn.sigmoid((y_upper_at_grid - grid_y) * sharpness)

    # "How much is this point above the lower surface?"
    mask_above_lower = jax.nn.sigmoid((grid_y - y_lower_at_grid) * sharpness)

    # "How much is this point within the chord?"
    mask_after_start = jax.nn.sigmoid((grid_x - x_min) * sharpness)
    mask_before_end = jax.nn.sigmoid((x_max - grid_x) * sharpness)
    mask_in_chord = mask_after_start * mask_before_end

    # Combine: inside if all conditions are met
    return mask_below_upper * mask_above_lower * mask_in_chord


def interpolate_surface_to_grid(
    grid_x_normalized: Array,
    x_coords: Array,
    y_coords: Array,
) -> Array:
    """
    Interpolate 1D surface coordinates to 2D grid.

    Args:
        grid_x_normalized: Normalized x-positions on grid, shape (H, W)
        x_coords: Surface x-coordinates, shape (num_points,)
        y_coords: Surface y-coordinates, shape (num_points,)

    Returns:
        Interpolated y-values at grid positions, shape (H, W)
    """
    # Flatten grid, interpolate, reshape
    original_shape = grid_x_normalized.shape
    grid_x_flat = grid_x_normalized.flatten()

    # Use jnp.interp for 1D interpolation
    y_interp_flat = jnp.interp(grid_x_flat, x_coords, y_coords)

    return y_interp_flat.reshape(original_shape)


__all__ = [
    "soft_sigmoid_mask",
    "soft_mask_from_bounds",
    "create_airfoil_mask",
    "interpolate_surface_to_grid",
]
