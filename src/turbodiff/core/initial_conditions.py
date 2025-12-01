"""
Initial condition generators for fluid simulation.

Provides helper functions to create various initial velocity fields.
"""

import jax.numpy as jnp
from turbodiff.core.types import Array
from typing import Tuple


def create_zero_velocity(resolution: Tuple[int, int]) -> Tuple[Array, Array]:
    """
    Create zero velocity field.

    Args:
        resolution: (height, width) of the grid

    Returns:
        (u, v): Zero velocity components
    """
    height, width = resolution
    u = jnp.zeros((height, width + 1))
    v = jnp.zeros((height + 1, width))
    return u, v


def create_spiral_velocity(
    resolution: Tuple[int, int], strength: float = 2.0
) -> Tuple[Array, Array]:
    """
    Create circular/vortex velocity field.

    Args:
        resolution: (height, width) of the grid
        strength: Velocity magnitude scaling factor

    Returns:
        (u, v): Velocity components for spiral flow
    """
    height, width = resolution
    center_i = height / 2.0
    center_j = width / 2.0

    # u-velocities at vertical faces (i, j)
    i_u, j_u = jnp.meshgrid(jnp.arange(height), jnp.arange(width + 1), indexing="ij")

    di_u = i_u - center_i
    dj_u = j_u - center_j
    dist_u = jnp.maximum(0.001, jnp.sqrt(di_u**2 + dj_u**2))
    u = -di_u / dist_u * strength

    # v-velocities at horizontal faces (i, j)
    i_v, j_v = jnp.meshgrid(jnp.arange(height + 1), jnp.arange(width), indexing="ij")

    di_v = i_v - center_i
    dj_v = j_v - center_j
    dist_v = jnp.maximum(0.001, jnp.sqrt(di_v**2 + dj_v**2))
    v = dj_v / dist_v * strength

    return u, v


def create_wind_tunnel_velocity(
    resolution: Tuple[int, int], speed: float = 2.0
) -> Tuple[Array, Array]:
    """
    Create left-to-right wind tunnel velocity field.

    Args:
        resolution: (height, width) of the grid
        speed: Horizontal wind speed

    Returns:
        (u, v): Velocity components for wind tunnel
    """
    height, width = resolution

    u = jnp.zeros((height, width + 1))
    v = jnp.zeros((height + 1, width))

    # Set leftmost interior u-velocities to create inflow
    u = u.at[1:-1, 1].set(speed)

    return u, v


def create_random_velocity(
    resolution: Tuple[int, int], key, magnitude: float = 0.5
) -> Tuple[Array, Array]:
    """
    Create random velocity field.

    Args:
        resolution: (height, width) of the grid
        key: JAX random key
        magnitude: Maximum velocity magnitude

    Returns:
        (u, v): Random velocity components
    """
    import jax.random as random

    height, width = resolution

    key_u, key_v = random.split(key)

    u = random.uniform(
        key_u, shape=(height, width + 1), minval=-magnitude, maxval=magnitude
    )
    v = random.uniform(
        key_v, shape=(height + 1, width), minval=-magnitude, maxval=magnitude
    )

    return u, v


__all__ = [
    "create_zero_velocity",
    "create_spiral_velocity",
    "create_wind_tunnel_velocity",
    "create_random_velocity",
]
