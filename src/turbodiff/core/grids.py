"""
JAX-based grid structures for fluid simulation.

This module provides immutable, differentiable grid representations:
- Grid: For scalar fields (density, pressure)
- StaggeredGrid: For MAC grid velocity representation
"""

from dataclasses import dataclass
from typing import Tuple
import jax.numpy as jnp
from turbodiff.core.types import Array


@dataclass(frozen=True)
class Grid:
    """
    Immutable grid for scalar fields (density, pressure).

    Attributes:
        values: 2D array of scalar values, shape (height, width)
        resolution: (height, width) tuple
        cell_size: Physical size of each cell in meters
    """

    values: Array  # Shape: (height, width)
    resolution: Tuple[int, int]
    cell_size: float

    @property
    def height(self) -> int:
        """Grid height in cells."""
        return self.resolution[0]

    @property
    def width(self) -> int:
        """Grid width in cells."""
        return self.resolution[1]

    @property
    def shape(self) -> Tuple[int, int]:
        """Shape of the values array."""
        return self.values.shape

    def with_values(self, new_values: Array) -> "Grid":
        """Create new Grid with updated values (immutable update)."""
        return Grid(new_values, self.resolution, self.cell_size)

    @staticmethod
    def zeros(resolution: Tuple[int, int], cell_size: float) -> "Grid":
        """Create zero-initialized grid."""
        height, width = resolution
        values = jnp.zeros((height, width))
        return Grid(values, resolution, cell_size)

    @staticmethod
    def ones(resolution: Tuple[int, int], cell_size: float) -> "Grid":
        """Create ones-initialized grid."""
        height, width = resolution
        values = jnp.ones((height, width))
        return Grid(values, resolution, cell_size)

    @staticmethod
    def from_array(values: Array, cell_size: float) -> "Grid":
        """Create grid from existing array."""
        height, width = values.shape
        return Grid(values, (height, width), cell_size)


@dataclass(frozen=True)
class StaggeredGrid:
    """
    Immutable MAC (Marker-and-Cell) grid for velocity field.

    In MAC grid, velocity components are stored on cell faces:
    - u (x-velocity): stored at vertical faces (i, j+0.5)
    - v (y-velocity): stored at horizontal faces (i+0.5, j)

    This arrangement naturally satisfies incompressibility and
    provides better numerical stability.

    Attributes:
        u: x-velocity component, shape (height, width+1)
        v: y-velocity component, shape (height+1, width)
        resolution: (height, width) of pressure/density cells
        cell_size: Physical size of each cell in meters
    """

    u: Array  # Shape: (height, width+1) - horizontal velocities
    v: Array  # Shape: (height+1, width) - vertical velocities
    resolution: Tuple[int, int]
    cell_size: float

    @property
    def height(self) -> int:
        """Grid height in cells."""
        return self.resolution[0]

    @property
    def width(self) -> int:
        """Grid width in cells."""
        return self.resolution[1]

    def with_values(self, new_u: Array, new_v: Array) -> "StaggeredGrid":
        """Create new StaggeredGrid with updated velocities (immutable update)."""
        return StaggeredGrid(new_u, new_v, self.resolution, self.cell_size)

    def with_u(self, new_u: Array) -> "StaggeredGrid":
        """Create new StaggeredGrid with updated u velocity."""
        return StaggeredGrid(new_u, self.v, self.resolution, self.cell_size)

    def with_v(self, new_v: Array) -> "StaggeredGrid":
        """Create new StaggeredGrid with updated v velocity."""
        return StaggeredGrid(self.u, new_v, self.resolution, self.cell_size)

    @staticmethod
    def zeros(resolution: Tuple[int, int], cell_size: float) -> "StaggeredGrid":
        """Create zero-velocity field."""
        height, width = resolution
        u = jnp.zeros((height, width + 1))
        v = jnp.zeros((height + 1, width))
        return StaggeredGrid(u, v, resolution, cell_size)

    @staticmethod
    def from_function(
        fn, resolution: Tuple[int, int], cell_size: float
    ) -> "StaggeredGrid":
        """
        Create velocity field from function fn(x, y) -> (u, v).

        Args:
            fn: Function that takes (x, y) physical coordinates and returns (u, v)
            resolution: (height, width) of cells
            cell_size: Physical cell size

        Returns:
            StaggeredGrid with velocities sampled at face centers
        """
        height, width = resolution

        # u-velocities at vertical faces (i, j+0.5)
        i_u, j_u = jnp.meshgrid(
            jnp.arange(height) * cell_size,
            jnp.arange(width + 1) * cell_size,
            indexing="ij",
        )
        j_u += 0.5 * cell_size
        u_values, _ = fn(j_u, i_u)

        # v-velocities at horizontal faces (i+0.5, j)
        i_v, j_v = jnp.meshgrid(
            jnp.arange(height + 1) * cell_size,
            jnp.arange(width) * cell_size,
            indexing="ij",
        )
        i_v += 0.5 * cell_size
        _, v_values = fn(j_v, i_v)

        return StaggeredGrid(u_values, v_values, resolution, cell_size)


__all__ = [
    "Grid",
    "StaggeredGrid",
]
