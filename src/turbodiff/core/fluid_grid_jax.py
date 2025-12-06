"""
JAX-based fluid simulator with MAC grid.

This module provides a differentiable fluid simulator using:
- JAX arrays for all state (immutable, differentiable)
- MAC (Marker-and-Cell) staggered grid for velocity
- Semi-Lagrangian advection
- Pressure projection for incompressibility
"""

from dataclasses import dataclass
from typing import Callable, Optional
import math
import jax
import jax.numpy as jnp
import pygame
from jax import Array
from turbodiff.core.grids import Grid, StaggeredGrid
from turbodiff.core.utils import (
    create_solid_mask,
    apply_zero_velocity_at_solids,
    bilinear_interpolate,
)


@dataclass(frozen=True)
class FluidState:
    """
    Complete immutable state of the fluid simulation.

    Attributes:
        density: Scalar field for smoke/dye density, Grid
        velocity: Velocity field, StaggeredGrid
        pressure: Pressure field, Grid
        solid_mask: Boolean mask for solid cells, Array (height, width)
        sources: Array of source strengths, same shape as density
        time: Current simulation time
        step: Current step number
    """

    density: Grid
    velocity: StaggeredGrid
    pressure: Grid
    solid_mask: Array
    sources: Array
    time: float = 0.0
    step: int = 0


class FluidGrid:
    """
    JAX-based 2D fluid simulator with MAC grid.

    Implements Stable Fluids algorithm with:
    - Semi-Lagrangian advection
    - Diffusion (implicit, Gauss-Seidel)
    - Pressure projection (incompressibility)

    All operations are differentiable through JAX.

    Args:
        height: Grid height in cells
        width: Grid width in cells
        cell_size: Physical size of each cell in meters
        dt: Time step in seconds
        diffusion: Diffusion coefficient
        viscosity: Viscosity coefficient (not yet implemented)
        boundary_solid: If True, boundary cells are solid
        sdf: Optional signed distance function for obstacles
    """

    def __init__(
        self,
        height: int,
        width: int,
        cell_size: float,
        dt: float,
        diffusion: float = 0.001,
        viscosity: float = 0.0,
        boundary_solid: bool = True,
        sdf: Optional[Callable[[Array, Array], Array]] = None,
        visualise: bool = False,
        show_cell_property: str = "density",
        show_velocity: bool = False,
        show_cell_centered_velocity: bool = False,
    ):
        self.height = height
        self.width = width
        self.resolution = (height, width)
        self.cell_size = cell_size
        self.dt = dt
        self.diffusion = diffusion
        self.viscosity = viscosity

        # Visualization settings
        self.visualise = visualise
        self.show_cell_property = show_cell_property
        self.show_velocity = show_velocity
        self.show_cell_centered_velocity = show_cell_centered_velocity

        # Track wind tunnel mode
        self.is_wind_tunnel = False

        # Create solid mask
        self.solid_mask = create_solid_mask(
            self.resolution, boundary=boundary_solid, sdf_fn=sdf
        )

        # Initialize pygame if needed
        if self.visualise:
            pygame.init()
            pygame.display.set_caption("TurboDiff - JAX Fluid Simulation")
            self.display_size = 1000 // max(self.height, self.width)
            self.screen = pygame.display.set_mode(
                (self.width * self.display_size, self.height * self.display_size)
            )
            self.clock = pygame.time.Clock()

    def create_initial_state(
        self,
        density_init: Optional[Array] = None,
        velocity_u_init: Optional[Array] = None,
        velocity_v_init: Optional[Array] = None,
        sources: Optional[Array] = None,
    ) -> FluidState:
        """
        Create initial simulation state.

        Args:
            density_init: Initial density field, shape (height, width)
            velocity_u_init: Initial u-velocity, shape (height, width+1)
            velocity_v_init: Initial v-velocity, shape (height+1, width)
            sources: Source strengths, shape (height, width)

        Returns:
            Initial FluidState
        """
        # Initialize density
        if density_init is None:
            density_init = jnp.zeros((self.height, self.width))
        density = Grid.from_array(density_init, self.cell_size)

        # Initialize velocity
        if velocity_u_init is None:
            velocity_u_init = jnp.zeros((self.height, self.width + 1))
        if velocity_v_init is None:
            velocity_v_init = jnp.zeros((self.height + 1, self.width))

        # Apply solid boundary conditions
        velocity_u_init, velocity_v_init = apply_zero_velocity_at_solids(
            velocity_u_init, velocity_v_init, self.solid_mask
        )
        velocity = StaggeredGrid(
            velocity_u_init, velocity_v_init, self.resolution, self.cell_size
        )

        # Initialize pressure
        pressure = Grid.zeros(self.resolution, self.cell_size)

        # Initialize sources
        if sources is None:
            sources = jnp.zeros((self.height, self.width))

        return FluidState(
            density=density,
            velocity=velocity,
            pressure=pressure,
            solid_mask=self.solid_mask,
            sources=sources,
            time=0.0,
            step=0,
        )

    def add_sources_to_density(self, state: FluidState) -> FluidState:
        """
        Add density sources.

        Args:
            state: Current simulation state

        Returns:
            New state with updated density
        """
        new_density = state.density.values + self.dt * state.sources
        return state.__class__(
            density=state.density.with_values(new_density),
            velocity=state.velocity,
            pressure=state.pressure,
            solid_mask=state.solid_mask,
            sources=state.sources,
            time=state.time,
            step=state.step,
        )

    def diffuse_density(self, state: FluidState, num_iters: int = 20) -> FluidState:
        """
        Perform Gauss–Seidel diffusion on the density field in a JAX-friendly way.

        Args:
            state: FluidState

        Returns:
            Updated FluidState with diffused density.
        """

        dt = self.dt
        diffusion = self.diffusion
        cell_size = self.cell_size

        solid = state.solid_mask
        density = state.density.values
        a = dt * diffusion / (cell_size * cell_size)

        # Precompute a mask for fluid interior cells
        interior_mask = (~solid)[1:-1, 1:-1]  # True only where not solid

        def gs_iteration(d):
            """One Gauss–Seidel-like update using vectorized neighbor sampling."""

            # Shifted neighbor fields (up, down, left, right)
            up = d[:-2, 1:-1]
            down = d[2:, 1:-1]
            left = d[1:-1, :-2]
            right = d[1:-1, 2:]

            # Neighbor counts (skip solids)
            n_up = (~solid[:-2, 1:-1]).astype(jnp.float32)
            n_down = (~solid[2:, 1:-1]).astype(jnp.float32)
            n_left = (~solid[1:-1, :-2]).astype(jnp.float32)
            n_right = (~solid[1:-1, 2:]).astype(jnp.float32)

            neighbors = up * n_up + down * n_down + left * n_left + right * n_right

            neighbor_counts = n_up + n_down + n_left + n_right

            center = d[1:-1, 1:-1]

            # Gauss–Seidel update formula
            new_center = (center + a * neighbors) / (1.0 + a * neighbor_counts)

            # Do not modify solid cells
            new_center = jnp.where(interior_mask, new_center, center)

            # Write back into full grid
            return d.at[1:-1, 1:-1].set(new_center)

        # Run repeated iterations
        new_density = jax.lax.fori_loop(
            0, num_iters, lambda _, d: gs_iteration(d), density
        )

        # Produce new state
        return state.__class__(
            density=state.density.with_values(new_density),
            velocity=state.velocity,
            pressure=state.pressure,
            solid_mask=state.solid_mask,
            sources=state.sources,
            time=state.time,
            step=state.step,
        )

    def advect_density(self, state: FluidState) -> FluidState:
        """
        Advect density using semi-Lagrangian method.

        Traces particles backward through velocity field and
        interpolates density values.

        Args:
            state: Current simulation state

        Returns:
            New state with advected density
        """
        height, width = self.resolution
        dt0 = self.dt / self.cell_size

        # Create grid of cell centers
        i_grid, j_grid = jnp.meshgrid(
            jnp.arange(1, height - 1), jnp.arange(1, width - 1), indexing="ij"
        )

        # Cell center positions
        x = j_grid + 0.5
        y = i_grid + 0.5

        # Sample velocity at cell centers
        u_sampled = bilinear_interpolate(
            state.velocity.u, x, y - 0.5, width + 1, height
        )
        v_sampled = bilinear_interpolate(
            state.velocity.v, x - 0.5, y, width, height + 1
        )

        # Backward trace
        x_back = x - dt0 * u_sampled
        y_back = y - dt0 * v_sampled

        # Clamp to valid range
        x_back = jnp.clip(x_back, 0.5, width - 0.5)
        y_back = jnp.clip(y_back, 0.5, height - 0.5)

        # Sample density at traced-back positions (adjust for 0-indexed)
        x_back_adj = x_back - 0.5
        y_back_adj = y_back - 0.5

        # Interpolate density
        new_density_interior = bilinear_interpolate(
            state.density.values, x_back_adj, y_back_adj, width, height
        )

        # Update density field (keep boundaries unchanged)
        new_density = state.density.values.at[1:-1, 1:-1].set(new_density_interior)

        return state.__class__(
            density=state.density.with_values(new_density),
            velocity=state.velocity,
            pressure=state.pressure,
            solid_mask=state.solid_mask,
            sources=state.sources,
            time=state.time,
            step=state.step,
        )

    def advect_velocity(self, state: FluidState) -> FluidState:
        """
        Advect velocity field (self-advection).

        Args:
            state: Current simulation state

        Returns:
            New state with advected velocity
        """
        height, width = self.resolution
        dt0 = self.dt / self.cell_size

        # Advect u-velocities (at vertical faces)
        i_u, j_u = jnp.meshgrid(
            jnp.arange(height), jnp.arange(width + 1), indexing="ij"
        )
        x_u = j_u.astype(float)
        y_u = i_u.astype(float) + 0.5

        # Sample velocity at u positions
        u_at_u = bilinear_interpolate(
            state.velocity.u, x_u, y_u - 0.5, width + 1, height
        )
        v_at_u = bilinear_interpolate(
            state.velocity.v, x_u - 0.5, y_u, width, height + 1
        )

        # Backward trace for u (no clamping - let interpolation handle bounds)
        x_u_back = x_u - dt0 * u_at_u
        y_u_back = y_u - dt0 * v_at_u

        # Sample u at traced positions (bilinear_interpolate handles clamping)
        new_u = bilinear_interpolate(
            state.velocity.u, x_u_back, y_u_back - 0.5, width + 1, height
        )

        # Advect v-velocities (at horizontal faces)
        i_v, j_v = jnp.meshgrid(
            jnp.arange(height + 1), jnp.arange(width), indexing="ij"
        )
        x_v = j_v.astype(float) + 0.5
        y_v = i_v.astype(float)

        # Sample velocity at v positions
        u_at_v = bilinear_interpolate(
            state.velocity.u, x_v, y_v - 0.5, width + 1, height
        )
        v_at_v = bilinear_interpolate(
            state.velocity.v, x_v - 0.5, y_v, width, height + 1
        )

        # Backward trace for v (no clamping - let interpolation handle bounds)
        x_v_back = x_v - dt0 * u_at_v
        y_v_back = y_v - dt0 * v_at_v

        # Sample v at traced positions (bilinear_interpolate handles clamping)
        new_v = bilinear_interpolate(
            state.velocity.v, x_v_back - 0.5, y_v_back, width, height + 1
        )

        # Apply solid boundary conditions
        new_u, new_v = apply_zero_velocity_at_solids(new_u, new_v, state.solid_mask)

        return state.__class__(
            density=state.density,
            velocity=state.velocity.with_values(new_u, new_v),
            pressure=state.pressure,
            solid_mask=state.solid_mask,
            sources=state.sources,
            time=state.time,
            step=state.step,
        )

    def inject_wind_tunnel_velocity(self, state: FluidState) -> FluidState:
        """
        Continuously inject velocity throughout domain for wind tunnel mode.
        This maintains a constant rightward flow.

        Args:
            state: Current simulation state

        Returns:
            New state with wind tunnel velocity injected
        """
        if not self.is_wind_tunnel:
            return state

        # Maintain uniform rightward velocity throughout domain
        u = jnp.ones_like(state.velocity.u) * 2.0
        # Keep v as is (or set to zero for pure horizontal flow)
        v = state.velocity.v

        return state.__class__(
            density=state.density,
            velocity=state.velocity.with_values(u, v),
            pressure=state.pressure,
            solid_mask=state.solid_mask,
            sources=state.sources,
            time=state.time,
            step=state.step,
        )

    def step(self, state: FluidState) -> FluidState:
        """
        Advance simulation by one time step.

        TESTING: Only advection enabled (no diffusion, no projection)

        Performs:
        1. Add density sources
        2. Advect density
        3. Advect velocity
        4. Inject wind tunnel velocity (if active)

        Args:
            state: Current simulation state

        Returns:
            New state after one time step
        """
        # Density step (advection only)
        state = self.add_sources_to_density(state)
        state = self.diffuse_density(state)
        state = self.advect_density(state)

        # Velocity step (advection only)
        state = self.advect_velocity(state)

        # Inject wind tunnel velocity if in wind tunnel mode
        state = self.inject_wind_tunnel_velocity(state)

        # Update time and step
        new_state = state.__class__(
            density=state.density,
            velocity=state.velocity,
            pressure=state.pressure,
            solid_mask=state.solid_mask,
            sources=state.sources,
            time=state.time + self.dt,
            step=state.step + 1,
        )

        return new_state

    def set_sources(
        self, state: FluidState, source_positions: list[tuple[int, int, float]]
    ) -> FluidState:
        """
        Set density sources at specified positions.

        Args:
            state: Current simulation state
            source_positions: List of (i, j, strength) tuples

        Returns:
            New state with updated sources
        """
        sources = jnp.zeros((self.height, self.width))

        for i, j, strength in source_positions:
            if 0 <= i < self.height and 0 <= j < self.width:
                sources = sources.at[i, j].set(strength)

        return state.__class__(
            density=state.density,
            velocity=state.velocity,
            pressure=state.pressure,
            solid_mask=state.solid_mask,
            sources=sources,
            time=state.time,
            step=state.step,
        )

    def set_velocity_field(
        self, state: FluidState, field_type: str = "zero"
    ) -> FluidState:
        """
        Initialize velocity field with various patterns.

        Args:
            state: Current simulation state
            field_type: Type of velocity field - "zero", "random", "spiral", or "wind tunnel"

        Returns:
            New state with initialized velocity field
        """
        height, width = self.resolution

        if field_type == "zero":
            u = jnp.zeros((height, width + 1))
            v = jnp.zeros((height + 1, width))

        elif field_type == "random":
            key = jax.random.PRNGKey(0)
            key_u, key_v = jax.random.split(key)
            u = jax.random.uniform(key_u, (height, width + 1), minval=-0.5, maxval=0.5)
            v = jax.random.uniform(key_v, (height + 1, width), minval=-0.5, maxval=0.5)

        elif field_type == "spiral":
            # Create circular/vortex flow pattern
            center_i = height // 2
            center_j = width // 2

            # U-velocities (at vertical faces)
            i_u, j_u = jnp.meshgrid(
                jnp.arange(height), jnp.arange(width + 1), indexing="ij"
            )
            di_u = i_u - center_i
            dj_u = j_u - center_j
            dist_u = jnp.maximum(0.001, jnp.sqrt(di_u**2 + dj_u**2))
            u = -di_u / dist_u * 2.0

            # V-velocities (at horizontal faces)
            i_v, j_v = jnp.meshgrid(
                jnp.arange(height + 1), jnp.arange(width), indexing="ij"
            )
            di_v = i_v - center_i
            dj_v = j_v - center_j
            dist_v = jnp.maximum(0.1, jnp.sqrt(di_v**2 + dj_v**2))
            v = dj_v / dist_v * 2.0

        elif field_type == "wind tunnel":
            # Create uniform rightward flow throughout the domain
            u = jnp.ones((height, width + 1)) * 2.0
            v = jnp.zeros((height + 1, width))
            # Mark as wind tunnel mode
            self.is_wind_tunnel = True
            # Make right boundary non-solid
            self.solid_mask = self.solid_mask.at[1:-1, -1].set(False)

        else:
            raise ValueError(f"Unknown field_type: {field_type}")

        # Apply solid boundary conditions
        u, v = apply_zero_velocity_at_solids(u, v, self.solid_mask)

        return state.__class__(
            density=state.density,
            velocity=state.velocity.with_values(u, v),
            pressure=state.pressure,
            solid_mask=state.solid_mask,
            sources=state.sources,
            time=state.time,
            step=state.step,
        )

    def get_divergence(self, state: FluidState, i: int, j: int) -> float:
        """
        Compute divergence at cell (i, j).

        Args:
            state: Current simulation state
            i: Row index
            j: Column index

        Returns:
            Divergence value
        """
        import numpy as np

        u = np.asarray(state.velocity.u)
        v = np.asarray(state.velocity.v)

        div = 0.0
        div -= u[i, j]  # left edge - in
        div += u[i, j + 1]  # right edge - out
        div -= v[i, j]  # top edge - in
        div += v[i + 1, j]  # bottom edge - out

        return float(div / self.cell_size)

    def get_curl(self, state: FluidState, i: int, j: int) -> float:
        """
        Compute curl at cell (i, j).

        Args:
            state: Current simulation state
            i: Row index
            j: Column index

        Returns:
            Curl value
        """
        import numpy as np

        u = np.asarray(state.velocity.u)
        v = np.asarray(state.velocity.v)

        curl = 0.0
        if j > 0:
            curl += (v[i, j - 1] + v[i + 1, j - 1]) / 2  # down on left
        if j < self.width - 1:
            curl -= (v[i, j + 1] + v[i + 1, j + 1]) / 2  # down on right
        if i > 0:
            curl -= (u[i - 1, j] + u[i - 1, j + 1]) / 2  # right on up
        if i < self.height - 1:
            curl += (u[i + 1, j] + u[i + 1, j + 1]) / 2  # right on down

        return float(curl / self.cell_size)

    def add_velocity_from_mouse(
        self,
        state: FluidState,
        prev_pos: tuple[int, int],
        curr_pos: tuple[int, int],
        velocity_scale: float = 5.0,
    ) -> FluidState:
        """
        Add velocity to the grid based on mouse movement.

        Args:
            state: Current simulation state
            prev_pos: Previous mouse position (pixels)
            curr_pos: Current mouse position (pixels)
            velocity_scale: Scaling factor for velocity magnitude

        Returns:
            New state with added velocity
        """
        prev_x, prev_y = prev_pos
        curr_x, curr_y = curr_pos

        # Calculate mouse velocity in pixels
        dx_pixels = curr_x - prev_x
        dy_pixels = curr_y - prev_y

        # Convert to grid units
        dx_grid = dx_pixels / self.display_size
        dy_grid = dy_pixels / self.display_size

        # Scale velocity
        u_add = dx_grid * velocity_scale
        v_add = dy_grid * velocity_scale

        # Get current cell position
        j = curr_x // self.display_size
        i = curr_y // self.display_size

        # Check bounds
        if i < 1 or i >= self.height - 1 or j < 1 or j >= self.width - 1:
            return state

        # Apply velocity to brush area
        u = state.velocity.u
        v = state.velocity.v
        brush_radius = 1

        for di in range(-brush_radius, brush_radius + 1):
            for dj in range(-brush_radius, brush_radius + 1):
                ni = i + di
                nj = j + dj

                if ni < 1 or ni >= self.height - 1 or nj < 1 or nj >= self.width - 1:
                    continue

                if self.solid_mask[ni, nj]:
                    continue

                # Add to u-velocities (left and right edges)
                u = u.at[ni, nj].add(u_add)
                u = u.at[ni, nj + 1].add(u_add)

                # Add to v-velocities (top and bottom edges)
                v = v.at[ni, nj].add(v_add)
                v = v.at[ni + 1, nj].add(v_add)

        return state.__class__(
            density=state.density,
            velocity=state.velocity.with_values(u, v),
            pressure=state.pressure,
            solid_mask=state.solid_mask,
            sources=state.sources,
            time=state.time,
            step=state.step,
        )

    def draw_state(self, state: FluidState):
        """
        Draw current simulation state using pygame.

        Args:
            state: Current simulation state
        """
        if not self.visualise:
            return

        self.screen.fill((0, 0, 0))

        # Convert to numpy arrays once for fast access
        import numpy as np

        density_array = np.asarray(state.density.values)
        solid_mask_array = np.asarray(state.solid_mask)
        pressure_array = (
            np.asarray(state.pressure.values)
            if hasattr(state.pressure, "values")
            else None
        )

        # Draw cells with appropriate coloring
        for i in range(self.height):
            for j in range(self.width):
                if solid_mask_array[i, j]:
                    color = (0, 0, 0)
                else:
                    if self.show_cell_property == "density":
                        GRAY_VALUE = 30
                        val = max(
                            GRAY_VALUE,
                            GRAY_VALUE
                            + min(
                                255 - GRAY_VALUE,
                                int(density_array[i, j] * (255 - GRAY_VALUE)),
                            ),
                        )
                        color = (val, val, val)
                    elif self.show_cell_property == "divergence":
                        div = self.get_divergence(state, i, j)
                        color = (
                            max(0, min(255, int(div))),
                            0,
                            max(0, min(255, int(-div))),
                        )
                    elif self.show_cell_property == "pressure":
                        if pressure_array is not None:
                            pressure = pressure_array[i, j]
                            color = (
                                max(0, min(255, int(200 * pressure))),
                                0,
                                max(0, min(255, int(200 * -pressure))),
                            )
                        else:
                            GRAY_VALUE = 30
                            color = (GRAY_VALUE, GRAY_VALUE, GRAY_VALUE)
                    elif self.show_cell_property == "curl":
                        curl = self.get_curl(state, i, j)
                        color = (
                            max(0, min(255, int(curl))),
                            0,
                            max(0, min(255, int(-curl))),
                        )
                    elif self.show_cell_property == "advection":
                        # Show neutral background to highlight velocity arrows
                        GRAY_VALUE = 30
                        color = (GRAY_VALUE, GRAY_VALUE, GRAY_VALUE)
                    else:
                        GRAY_VALUE = 30
                        color = (GRAY_VALUE, GRAY_VALUE, GRAY_VALUE)

                rect = pygame.Rect(
                    j * self.display_size,
                    i * self.display_size,
                    self.display_size,
                    self.display_size,
                )
                pygame.draw.rect(self.screen, color, rect)

        # Draw velocity arrows if requested
        if self.show_velocity:
            import numpy as np

            u_array = np.asarray(state.velocity.u)
            v_array = np.asarray(state.velocity.v)

            if self.show_cell_centered_velocity:
                # Cell Centered Velocity
                # Vectorized velocity computation at cell centers
                u_at_centers = (u_array[:, :-1] + u_array[:, 1:]) / 2.0
                v_at_centers = (v_array[:-1, :] + v_array[1:, :]) / 2.0

                # Compute magnitudes
                mags = np.sqrt(u_at_centers**2 + v_at_centers**2)
                mags_clamped = np.minimum(mags, 1.0)

                # Normalized directions (avoid division by zero)
                mag_nonzero = mags > 1e-6
                # Use np.divide with where parameter to safely handle division
                mag_dir_x = np.divide(
                    u_at_centers,
                    mags,
                    out=np.zeros_like(u_at_centers),
                    where=mag_nonzero,
                )
                mag_dir_y = np.divide(
                    v_at_centers,
                    mags,
                    out=np.zeros_like(v_at_centers),
                    where=mag_nonzero,
                )

                for i in range(self.height):
                    for j in range(self.width):
                        mag = float(mags_clamped[i, j])
                        mag_dir_x_val = float(mag_dir_x[i, j])
                        mag_dir_y_val = float(mag_dir_y[i, j])

                        # Color based on magnitude
                        r = int(255 * mag)
                        g = 0
                        b = int(255 * (1 - mag))
                        color = (r, g, b)

                        # Draw arrow
                        scale = self.display_size * 0.4
                        start_x = (j + 0.5) * self.display_size
                        start_y = (i + 0.5) * self.display_size
                        end_x = start_x + scale * mag_dir_x_val
                        end_y = start_y + scale * mag_dir_y_val

                        pygame.draw.aaline(
                            self.screen, color, (start_x, start_y), (end_x, end_y), 2
                        )

                        # Draw arrowhead
                        angle = math.atan2(mag_dir_y_val, mag_dir_x_val)
                        tip_len = scale * mag / 2
                        spread = math.radians(25)

                        left_x = end_x - tip_len * math.cos(angle - spread)
                        left_y = end_y - tip_len * math.sin(angle - spread)
                        right_x = end_x - tip_len * math.cos(angle + spread)
                        right_y = end_y - tip_len * math.sin(angle + spread)

                        pygame.draw.aalines(
                            self.screen,
                            color,
                            True,
                            [(end_x, end_y), (left_x, left_y), (right_x, right_y)],
                        )

            else:
                # Face-centered velocity arrows
                # Horizontal Velocity (u-velocities at vertical faces)
                for i in range(self.height):
                    for j in range(self.width + 1):
                        mag_dir = float(u_array[i, j])
                        mag_dir = (
                            min(1.0, mag_dir) if mag_dir > 0 else max(-1.0, mag_dir)
                        )

                        # Color based on magnitude
                        r = int(255 * abs(mag_dir))
                        g = 0
                        b = int(255 * (1 - abs(mag_dir)))
                        color = (r, g, b)

                        scale = self.display_size * 0.4
                        start_x = j * self.display_size
                        end_x = start_x + scale * mag_dir
                        y = (i + 0.5) * self.display_size

                        pygame.draw.aaline(
                            self.screen, color, (start_x, y), (end_x, y), 2
                        )

                        # Draw arrowhead
                        angle = 0 if mag_dir > 0 else math.pi
                        tip_len = scale * abs(mag_dir) / 2
                        spread = math.radians(25)

                        left_x = end_x - tip_len * math.cos(angle - spread)
                        left_y = y - tip_len * math.sin(angle - spread)
                        right_x = end_x - tip_len * math.cos(angle + spread)
                        right_y = y - tip_len * math.sin(angle + spread)

                        pygame.draw.aalines(
                            self.screen,
                            color,
                            True,
                            [(end_x, y), (left_x, left_y), (right_x, right_y)],
                        )

                # Vertical Velocity (v-velocities at horizontal faces)
                for i in range(self.height + 1):
                    for j in range(self.width):
                        mag_dir = float(v_array[i, j])
                        mag_dir = (
                            min(1.0, mag_dir) if mag_dir > 0 else max(-1.0, mag_dir)
                        )

                        # Color based on magnitude
                        r = int(255 * abs(mag_dir))
                        g = 0
                        b = int(255 * (1 - abs(mag_dir)))
                        color = (r, g, b)

                        scale = self.display_size * 0.4
                        x = (j + 0.5) * self.display_size
                        start_y = i * self.display_size
                        end_y = start_y + scale * mag_dir

                        pygame.draw.aaline(
                            self.screen, color, (x, start_y), (x, end_y), 2
                        )

                        # Draw arrowhead
                        angle = math.pi / 2 if mag_dir > 0 else -math.pi / 2
                        tip_len = scale * abs(mag_dir) / 2
                        spread = math.radians(25)

                        left_x = x - tip_len * math.cos(angle - spread)
                        left_y = end_y - tip_len * math.sin(angle - spread)
                        right_x = x - tip_len * math.cos(angle + spread)
                        right_y = end_y - tip_len * math.sin(angle + spread)

                        pygame.draw.aalines(
                            self.screen,
                            color,
                            True,
                            [(x, end_y), (left_x, left_y), (right_x, right_y)],
                        )

        # Cursor indicator when in painting mode
        keys = pygame.key.get_pressed()
        if keys[pygame.K_a]:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            # Circle to show brush area
            brush_radius_pixels = self.display_size * 1.5
            pygame.draw.circle(
                self.screen,
                (0, 255, 0),
                (mouse_x, mouse_y),
                int(brush_radius_pixels),
                2,
            )
            # Crosshair
            pygame.draw.line(
                self.screen,
                (0, 255, 0),
                (mouse_x - 10, mouse_y),
                (mouse_x + 10, mouse_y),
                2,
            )
            pygame.draw.line(
                self.screen,
                (0, 255, 0),
                (mouse_x, mouse_y - 10),
                (mouse_x, mouse_y + 10),
                2,
            )

        pygame.display.flip()

    def simulate(self, state: FluidState, steps: int = -1) -> FluidState:
        """
        Run simulation loop with visualization.

        Args:
            state: Initial simulation state
            steps: Number of steps (-1 for infinite)

        Returns:
            Final simulation state
        """
        step = 0
        prev_mouse_pos = None

        while step != steps:
            if self.visualise:
                # Handle events
                keys = pygame.key.get_pressed()
                is_painting = keys[pygame.K_a]
                mouse_pos = pygame.mouse.get_pos()

                # Add velocity from mouse
                if is_painting and prev_mouse_pos is not None:
                    state = self.add_velocity_from_mouse(
                        state, prev_mouse_pos, mouse_pos
                    )

                prev_mouse_pos = mouse_pos if is_painting else None

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return state
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        mx, my = pygame.mouse.get_pos()
                        j = mx // self.display_size
                        i = my // self.display_size
                        print(f"Clicked cell: ({i}, {j})")

            # Simulation step
            state = self.step(state)

            if self.visualise:
                self.draw_state(state)
                self.clock.tick(30)

            step += 1

        if self.visualise:
            pygame.quit()

        return state


__all__ = [
    "FluidGrid",
    "FluidState",
]
