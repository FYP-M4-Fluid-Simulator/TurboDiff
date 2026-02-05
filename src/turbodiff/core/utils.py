"""
Utility functions for fluid simulation operations.

This module provides JAX-compatible utility functions for:
- Bilinear interpolation
- Boundary conditions
- Grid operations

All functions are designed to be jit-compilable and differentiable.
"""

import jax
import jax.numpy as jnp
from turbodiff.core.types import Array


@jax.jit
def bilinear_interpolate(values: Array, x: Array, y: Array, nx: int, ny: int) -> Array:
    """
    Bilinear interpolation on a 2D grid.

    Fully differentiable bilinear interpolation for sampling values
    at arbitrary positions within the grid. JIT-compiled for performance.

    Args:
        values: 2D array of values, shape (ny, nx)
        x: x-coordinates to sample at (in grid coordinates)
        y: y-coordinates to sample at (in grid coordinates)
        nx: width of the grid
        ny: height of the grid

    Returns:
        Interpolated values at (x, y) positions

    Note:
        - Grid coordinates: (0, 0) at top-left, (nx-1, ny-1) at bottom-right
        - x, y can be scalars or arrays of the same shape
        - Fully vectorized, no Python loops
    """
    # Clamp coordinates to valid range
    x = jnp.clip(x, 0.5, nx - 1.5)
    y = jnp.clip(y, 0.5, ny - 1.5)

    # Get integer indices
    x0 = jnp.floor(x).astype(jnp.int32)
    y0 = jnp.floor(y).astype(jnp.int32)

    # Ensure indices are within bounds
    x0 = jnp.clip(x0, 0, nx - 2)
    y0 = jnp.clip(y0, 0, ny - 2)

    x1 = x0 + 1
    y1 = y0 + 1

    # Get interpolation weights
    wx = x - x0
    wy = y - y0

    # Bilinear interpolation (fully vectorized)
    v00 = values[y0, x0]
    v01 = values[y0, x1]
    v10 = values[y1, x0]
    v11 = values[y1, x1]

    v0 = v00 * (1 - wx) + v01 * wx
    v1 = v10 * (1 - wx) + v11 * wx

    return v0 * (1 - wy) + v1 * wy


@jax.jit
def sample_staggered_velocity_at(
    u: Array, v: Array, x: Array, y: Array, height: int, width: int
) -> tuple[Array, Array]:
    """
    Sample velocity components at arbitrary positions.

    Interpolates velocity from MAC grid at any position (x, y).
    Fully vectorized and JIT-compiled for performance.

    Args:
        u: x-velocity, shape (height, width+1)
        v: y-velocity, shape (height+1, width)
        x: x-coordinates in grid units
        y: y-coordinates in grid units
        height: grid height
        width: grid width

    Returns:
        (u_sampled, v_sampled): Velocity components at (x, y)
    """
    # Sample u at (x, y-0.5) since u is at vertical faces
    u_sampled = bilinear_interpolate(u, x, y - 0.5, width + 1, height)

    # Sample v at (x-0.5, y) since v is at horizontal faces
    v_sampled = bilinear_interpolate(v, x - 0.5, y, width, height + 1)

    return u_sampled, v_sampled


def create_solid_mask(
    resolution: tuple[int, int], boundary: int = 1, sdf_fn=None, smoothing=0.1
) -> Array:
    """
    Create solid cell mask for the grid.

    Args:
        resolution: (height, width) of the grid
        boundary: 0 -> No Boundary, 1 -> Complete Boundary, 2 -> No right boundary
        sdf_fn: Optional signed distance function sdf(i, j) < 0 means solid

    Returns:
        Boolean array, shape (height, width), True = solid
    """
    height, width = resolution

    # Initialize as all fluid
    solid_mask = jnp.zeros((height, width), dtype=bool)

    if boundary:
        # Mark boundary cells as solid
        solid_mask = solid_mask.at[0, :].set(True)
        solid_mask = solid_mask.at[-1, :].set(True)
        solid_mask = solid_mask.at[:, 0].set(True)
    if boundary == 1:
        solid_mask = solid_mask.at[:, -1].set(True)

    if sdf_fn is not None:
        # Create coordinate grids
        i_grid, j_grid = jnp.meshgrid(
            jnp.arange(height), jnp.arange(width), indexing="ij"
        )

        # Evaluate SDF and mark solid where SDF < 0
        sdf_values = sdf_fn(i_grid, j_grid)
        solid_mask = jnp.maximum(solid_mask, 1 - jax.nn.sigmoid(sdf_values / smoothing))

    return solid_mask

import jax.numpy as jnp

def create_solid_border(
    window_size: tuple[int, int],
    cell_size: float,
    sdf_fn=None,
    thickness=0.2
) -> list[tuple[int, int]]:
    """
    Create a list of solid border cell indices based on a signed distance function.

    Cells are marked as solid if the absolute value of the signed distance
    is smaller than the given thickness.

    Args:
        window_size: (height, width) of the grid
        cell_size: Physical size of a grid cell (used to scale coordinates)
        sdf_fn: Optional signed distance function sdf(i, j)
                Values near zero represent the boundary
        thickness: Rough measure of thickness of the solid border

    Returns:
        List of (i, j) index tuples representing solid border cells
    """
    if sdf_fn is None:
        return []

    # Create coordinate grid
    i_grid, j_grid = jnp.meshgrid(
        jnp.arange(window_size[0]),
        jnp.arange(window_size[1]),
        indexing="ij",
    )

    i_grid = i_grid / cell_size
    j_grid = j_grid / cell_size

    # Compute sdf at each pixel
    sdf_values = sdf_fn(i_grid, j_grid)

    # Identify points where sign changes (left to right)
    sdf_values_sign_left = jax.nn.sigmoid(sdf_values[:,:-1]) > 0.5 - thickness
    sdf_values_sign_right = jax.nn.sigmoid(sdf_values[:,1:]) > 0.5 + thickness
    border = jnp.logical_xor(sdf_values_sign_left, sdf_values_sign_right)
    
    # Identify border cells
    indices = jnp.argwhere(border)

    return [tuple(idx) for idx in indices.tolist()]




@jax.jit
def apply_zero_velocity_at_solids(
    u: Array, v: Array, solid_mask: Array
) -> tuple[Array, Array]:
    """
    Zero out velocities at solid boundaries.

    Vectorized implementation that sets velocities to zero at solid boundaries.
    Uses JAX's where() for functional updates.

    Args:
        u: x-velocity, shape (height, width+1)
        v: y-velocity, shape (height+1, width)
        solid_mask: Boolean mask, shape (height, width)

    Returns:
        (u_new, v_new): Velocities with zeros at solid boundaries
    """
    height, width = solid_mask.shape

    # Zero u-velocities where either adjacent cell is solid
    # u[i, j] is between cells [i, j-1] and [i, j]
    # For j=0, only check right cell; for j=width, only check left cell
    u_should_zero = jnp.zeros((height, width + 1), dtype=bool)

    # Middle u-faces: check both left and right cells
    u_should_zero = u_should_zero.at[:, 1:width].set(
        jnp.logical_or(solid_mask[:, :-1], solid_mask[:, 1:])
    )
    # Left boundary (j=0): check right cell
    u_should_zero = u_should_zero.at[:, 0].set(solid_mask[:, 0])
    # Right boundary (j=width): check left cell
    u_should_zero = u_should_zero.at[:, width].set(solid_mask[:, -1])

    u = jnp.where(u_should_zero, 0.0, u)

    # Zero v-velocities where either adjacent cell is solid
    # v[i, j] is between cells [i-1, j] and [i, j]
    v_should_zero = jnp.zeros((height + 1, width), dtype=bool)

    # Middle v-faces: check both top and bottom cells
    v_should_zero = v_should_zero.at[1:height, :].set(
        jnp.logical_or(solid_mask[:-1, :], solid_mask[1:, :])
    )
    # Top boundary (i=0): check bottom cell
    v_should_zero = v_should_zero.at[0, :].set(solid_mask[0, :])
    # Bottom boundary (i=height): check top cell
    v_should_zero = v_should_zero.at[height, :].set(solid_mask[-1, :])

    v = jnp.where(v_should_zero, 0.0, v)

    return u, v


__all__ = [
    "bilinear_interpolate",
    "sample_staggered_velocity_at",
    "create_solid_mask",
    "apply_zero_velocity_at_solids",
]
