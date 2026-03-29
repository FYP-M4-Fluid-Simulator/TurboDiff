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
    create_solid_border,
    apply_ibm_continuous_forcing,
    bilinear_interpolate,
)

import time
from functools import wraps


def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} took {end - start:.6f} seconds")
        return result

    return wrapper


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
        nu_tilde: Modified eddy viscosity field for Spalart-Allmaras model (Grid),
                  or None when turbulence model is disabled.
        time: Current simulation time
        step: Current step number
    """

    density: Grid
    velocity: StaggeredGrid
    pressure: Grid
    solid_mask: Array
    sources: Array
    nu_tilde: Optional[Grid] = None
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
        viscosity: Kinematic viscosity coefficient (m²/s)
        rho: Fluid density (kg/m³), default 1.0
        boundary_type: 0 -> No Boundary, 1 -> Complete Boundary, 2 -> No right boundary
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
        rho: float = 1.0,
        boundary_type: int = 1,
        sdf: Optional[Callable[[Array, Array], Array]] = None,
        visualise: bool = False,
        show_cell_property: str = "density",
        show_velocity: bool = False,
        show_cell_centered_velocity: bool = False,
        use_sa_turbulence: bool = False,
        cv_rect: Optional[tuple[int, int, int, int]] = None,
    ):
        self.height = height
        self.width = width
        self.resolution = (height, width)
        self.cell_size = cell_size
        self.dt = dt
        self.diffusion = diffusion
        self.viscosity = viscosity
        self.rho = rho

        # ── Spalart-Allmaras turbulence model ─────────────────────────────────
        self.use_sa_turbulence = use_sa_turbulence
        # SA-1994 model constants
        self.sa_cb1 = 0.1355
        self.sa_cb2 = 0.622
        self.sa_sigma = 2.0 / 3.0
        self.sa_kappa = 0.41
        self.sa_cw2 = 0.3
        self.sa_cw3 = 2.0
        self.sa_cv1 = 7.1
        # Derived constant: cw1 = cb1/κ² + (1+cb2)/σ
        self.sa_cw1 = (
            self.sa_cb1 / (self.sa_kappa**2) + (1.0 + self.sa_cb2) / self.sa_sigma
        )
        # Wall-distance cache (set in _compute_wall_distance after solid_mask is known)
        self._wall_dist: Optional[Array] = None

        # Visualization settings
        self.visualise = visualise
        self.show_cell_property = show_cell_property
        self.show_velocity = show_velocity
        self.show_cell_centered_velocity = show_cell_centered_velocity

        # Track wind tunnel mode
        self.is_wind_tunnel = False
        self.inlet_velocity = 2.0
        self.inlet_angle_rad = 0.0
        self.cv_rect = cv_rect  # (i1, i2, j1, j2)

        # Create solid mask
        self.solid_mask = create_solid_mask(
            self.resolution, boundary=boundary_type, sdf_fn=sdf
        )

        # Pre-compute wall distance for SA model
        if self.use_sa_turbulence:
            self._wall_dist = self._compute_wall_distance(self.solid_mask)

        # Initialize pygame if needed
        if self.visualise:
            pygame.init()
            pygame.display.set_caption("TurboDiff - JAX Fluid Simulation")
            self.display_size = 1000 // max(self.height, self.width)
            self.screen = pygame.display.set_mode(
                (self.width * self.display_size, self.height * self.display_size)
            )
            self.solid_border = create_solid_border(
                pygame.display.get_window_size()[::-1], self.display_size, sdf
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

        # Apply continuous forcing solid boundary conditions
        velocity_u_init, velocity_v_init = apply_ibm_continuous_forcing(
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

        # Initialize SA modified eddy viscosity (ν̃) if turbulence model is active.
        # Seeded at 5ν in fluid cells, 0 in solids — a common SA initialisation.
        nu_tilde: Optional[Grid] = None
        if self.use_sa_turbulence:
            nu_init = jnp.where(
                self.solid_mask > 0.5,
                0.0,
                5.0 * self.viscosity,
            )
            nu_tilde = Grid.from_array(nu_init, self.cell_size)

        return FluidState(
            density=density,
            velocity=velocity,
            pressure=pressure,
            solid_mask=self.solid_mask,
            sources=sources,
            nu_tilde=nu_tilde,
            time=0.0,
            step=0,
        )

    # ============================================================================
    # Spalart-Allmaras (SA-1994) Turbulence Model
    # ============================================================================

    def _compute_wall_distance(self, solid_mask: Array) -> Array:
        """
        Compute the Euclidean distance from every cell centre to the nearest
        solid cell centre.  The result is a pure JAX array so that gradients
        can flow through it if needed.

        Strategy: iterate over *solid* cell indices (typically O(1 000) for an
        airfoil) and take the pixelwise minimum, giving O(N·S) cost instead of
        the O(N²) all-pairs approach.

        Args:
            solid_mask: float array (height, width), values in [0, 1]

        Returns:
            wall_dist: float array (height, width) — physical distance in metres
        """
        h, w = self.height, self.width
        cs = self.cell_size

        # Cell-centre coordinates (grid units)
        i_grid, j_grid = jnp.meshgrid(
            jnp.arange(h, dtype=jnp.float32),
            jnp.arange(w, dtype=jnp.float32),
            indexing="ij",
        )  # (h, w)

        # Solid cell centres — use the mask as a soft weight; treat cells with
        # solid_mask >= 0.5 as walls.  We collect ALL solid cells and use
        # vmap for a fully differentiable, JIT-friendly computation.
        solid_float = (solid_mask >= 0.5).astype(jnp.float32)

        # We vectorise over every solid cell by scanning row-major.
        # For small S this is fast; for large grids with many solid cells use
        # a sub-sampled boundary list instead.
        def body(carry, idx):
            dist_min = carry  # (h, w)
            si = idx // w
            sj = idx % w
            is_solid = solid_float[si, sj]
            d = jnp.sqrt((i_grid - si) ** 2 + (j_grid - sj) ** 2)
            # Only update where this cell is actually solid
            dist_min = jnp.where(is_solid > 0.5, jnp.minimum(dist_min, d), dist_min)
            return dist_min, None

        total_cells = h * w
        dist_init = jnp.full((h, w), fill_value=1e6, dtype=jnp.float32)
        wall_dist_grid, _ = jax.lax.scan(body, dist_init, jnp.arange(total_cells))
        # Convert from grid units to physical metres, clamp to small positive
        wall_dist_phys = jnp.maximum(wall_dist_grid * cs, 1e-10)
        return wall_dist_phys

    @jax.jit
    def compute_effective_viscosity(self, state: FluidState) -> Array:
        """
        Compute the effective (laminar + turbulent) kinematic viscosity field
        from the SA modified eddy viscosity ν̃.

        The SA f_v1 damping function reads:
            χ     = ν̃ / ν
            f_v1  = χ³ / (χ³ + cv1³)
            ν_t   = ν̃ · f_v1
            ν_eff = ν  + ν_t

        All operations are pure jnp — fully differentiable.

        Returns:
            nu_eff: Array (height, width) in m²/s
        """
        nu = self.viscosity
        cv1 = self.sa_cv1

        nu_tilde = state.nu_tilde.values  # (h, w)
        # Clamp ν̃ to non-negative to avoid spurious negatives
        nu_tilde = jnp.maximum(nu_tilde, 0.0)

        chi = nu_tilde / (nu + 1e-30)  # χ = ν̃/ν
        chi3 = chi**3
        fv1 = chi3 / (chi3 + cv1**3)  # damping function

        nu_t = nu_tilde * fv1  # turbulent eddy viscosity
        nu_eff = nu + nu_t  # effective viscosity
        return nu_eff

    @jax.jit(static_argnames=("num_diff_iters",))
    def step_sa_turbulence(
        self,
        state: FluidState,
        wall_dist: Array,
        num_diff_iters: int = 4,
    ) -> FluidState:
        """
        Advance the SA modified eddy viscosity field ν̃ by one time step.

        The SA transport equation (Spalart & Allmaras 1994):
            Dν̃/Dt = cb1*(1-ft2)*S̃*ν̃                       [production]
                   + (1/σ)*∇·((ν+ν̃)∇ν̃) + (cb2/σ)*|∇ν̃|²  [diffusion]
                   - (cw1*fw - cb1/κ²*ft2)*(ν̃/d)²          [destruction]

        Simplified SA (ft2=0, fw≈1 for moderate flows) implemented here:
            Dν̃/Dt ≈ cb1*Ω*ν̃ + (1/σ)*∇·((ν+ν̃)∇ν̃) - cw1*(ν̃/d)²

        All operations are pure jnp:
        - Semi-Lagrangian RK2 advection of ν̃
        - Explicit production/destruction source step
        - Gauss-Seidel diffusion with spatially varying ν+ν̃

        Gradients w.r.t. state.nu_tilde and state.velocity flow correctly
        through every jnp operation.

        Args:
            state:          Current FluidState (requires nu_tilde != None)
            wall_dist:      Pre-computed wall distance array (h, w)
            num_diff_iters: Gauss-Seidel diffusion iterations

        Returns:
            New FluidState with updated nu_tilde
        """
        height, width = self.resolution
        dt = self.dt
        h = self.cell_size
        nu = self.viscosity

        cb1 = self.sa_cb1
        cb2 = self.sa_cb2
        sigma = self.sa_sigma
        cw1 = self.sa_cw1

        nu_tilde = jnp.maximum(state.nu_tilde.values, 0.0)  # (h, w)
        fluid_mask = 1.0 - state.solid_mask  # 1 in fluid, 0 in solid

        # ── 1. RK2 semi-Lagrangian advection of ν̃ ─────────────────────────────
        dt0 = dt / h
        i_grid, j_grid = jnp.meshgrid(
            jnp.arange(1, height - 1, dtype=jnp.float32),
            jnp.arange(1, width - 1, dtype=jnp.float32),
            indexing="ij",
        )
        x = j_grid + 0.5
        y = i_grid + 0.5

        # Stage 1: velocity at current positions
        u1 = bilinear_interpolate(state.velocity.u, x, y - 0.5, width + 1, height)
        v1 = bilinear_interpolate(state.velocity.v, x - 0.5, y, width, height + 1)
        x_mid = x - 0.5 * dt0 * u1
        y_mid = y - 0.5 * dt0 * v1

        # Stage 2: velocity at midpoint
        u2 = bilinear_interpolate(
            state.velocity.u, x_mid, y_mid - 0.5, width + 1, height
        )
        v2 = bilinear_interpolate(
            state.velocity.v, x_mid - 0.5, y_mid, width, height + 1
        )
        x_back = x - dt0 * u2
        y_back = y - dt0 * v2

        # Clamp & sample
        x_back = jnp.clip(x_back, 0.5, width - 0.5)
        y_back = jnp.clip(y_back, 0.5, height - 0.5)
        nut_adv_interior = bilinear_interpolate(
            nu_tilde, x_back - 0.5, y_back - 0.5, width, height
        )
        nu_tilde = nu_tilde.at[1:-1, 1:-1].set(nut_adv_interior)

        # ── 2. Compute vorticity magnitude Ω = |∂v/∂x − ∂u/∂y| ───────────────
        # Use cell-centred finite differences on staggered fields.
        # v at cell centres (approximate via face average)
        v_cc = 0.5 * (state.velocity.v[:-1, :] + state.velocity.v[1:, :])  # (h, w)
        u_cc = 0.5 * (state.velocity.u[:, :-1] + state.velocity.u[:, 1:])  # (h, w)

        # Interior gradients only (avoid boundary artefacts)
        dvdx = jnp.zeros((height, width))
        dudy = jnp.zeros((height, width))
        dvdx = dvdx.at[1:-1, 1:-1].set((v_cc[1:-1, 2:] - v_cc[1:-1, :-2]) / (2.0 * h))
        dudy = dudy.at[1:-1, 1:-1].set((u_cc[2:, 1:-1] - u_cc[:-2, 1:-1]) / (2.0 * h))
        omega = jnp.abs(dvdx - dudy)  # vorticity magnitude Ω

        # ── 3. Production and destruction (explicit Euler on source) ────────────
        d = wall_dist  # (h, w)
        chi = nu_tilde / (nu + 1e-30)
        chi3 = chi**3
        fv1 = chi3 / (chi3 + self.sa_cv1**3)
        fv2 = 1.0 - chi / (1.0 + chi * fv1)

        # Modified strain rate: S̃ = Ω + ν̃/(κ²d²)*f_v2
        kappa2 = self.sa_kappa**2
        # SA-2000 / Edwards correction: S̃ must not fall below Clim·Ω.
        # Without this clip, S̃ can go to zero or negative near stagnation,
        # which zeroes production and causes SA to diverge.
        S_bar = nu_tilde / (kappa2 * d**2 + 1e-30) * fv2
        S_tilde = omega + S_bar
        clim = 0.3
        S_tilde = jnp.maximum(S_tilde, clim * omega)
        S_tilde = jnp.maximum(S_tilde, 1e-10)  # absolute safety floor

        production = cb1 * S_tilde * nu_tilde
        destruction = cw1 * (nu_tilde / (d + 1e-30)) ** 2

        # |∇ν̃|² cross-diffusion term  (cb2/σ)*|∇ν̃|²
        dntdx = jnp.zeros((height, width))
        dntdy = jnp.zeros((height, width))
        dntdx = dntdx.at[1:-1, 1:-1].set(
            (nu_tilde[1:-1, 2:] - nu_tilde[1:-1, :-2]) / (2.0 * h)
        )
        dntdy = dntdy.at[1:-1, 1:-1].set(
            (nu_tilde[2:, 1:-1] - nu_tilde[:-2, 1:-1]) / (2.0 * h)
        )
        cross_diff = (cb2 / sigma) * (dntdx**2 + dntdy**2)

        source = (production - destruction + cross_diff) * fluid_mask
        nu_tilde = nu_tilde + dt * source
        nu_tilde = jnp.maximum(nu_tilde, 0.0)  # ν̃ must stay non-negative

        # ── 4. Gauss-Seidel diffusion of ν̃ ────────────────────────────────────
        # Effective diffusivity: (ν + ν̃) / σ  — spatially varying.
        nu_tilde_0 = nu_tilde  # save for RHS

        def gs_nut(carry, _):
            nt, nt0 = carry
            # Spatially varying diffusion factor at each cell
            a_field = dt * (nu + nt) / (sigma * h * h)  # (h, w)
            a_c = a_field[1:-1, 1:-1]

            up = nt[:-2, 1:-1]
            down = nt[2:, 1:-1]
            left = nt[1:-1, :-2]
            right = nt[1:-1, 2:]

            # Implicit (Jacobi-style) update
            new_c = (nt0[1:-1, 1:-1] + a_c * (up + down + left + right)) / (
                1.0 + 4.0 * a_c
            )
            # Zero ν̃ in solid cells
            mask_c = fluid_mask[1:-1, 1:-1]
            new_c = mask_c * new_c + (1.0 - mask_c) * nt[1:-1, 1:-1]
            return (nt.at[1:-1, 1:-1].set(new_c), nt0), None

        (nu_tilde, _), _ = jax.lax.scan(
            gs_nut, (nu_tilde, nu_tilde_0), None, length=num_diff_iters
        )

        # Enforce zero ν̃ in solid cells
        nu_tilde = nu_tilde * fluid_mask

        return state.__class__(
            density=state.density,
            velocity=state.velocity,
            pressure=state.pressure,
            solid_mask=state.solid_mask,
            sources=state.sources,
            nu_tilde=state.nu_tilde.with_values(nu_tilde),
            time=state.time,
            step=state.step,
        )

    @jax.jit
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
            nu_tilde=state.nu_tilde,
            time=state.time,
            step=state.step,
        )

    @jax.jit(static_argnames=("num_iters",))
    def diffuse_density(self, state: FluidState, num_iters: int = 50) -> FluidState:
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
        density0 = state.density.values
        density = state.density.values
        a = dt * diffusion / (cell_size * cell_size)

        # Precompute a mask for fluid interior cells
        interior_mask = (1.0 - solid)[1:-1, 1:-1]  # True only where not solid

        @jax.jit
        def gs_iteration(d_tuple):
            """One Gauss–Seidel-like update using vectorized neighbor sampling."""

            d, d0 = d_tuple

            # Shifted neighbor fields (up, down, left, right)
            up = d[:-2, 1:-1]
            down = d[2:, 1:-1]
            left = d[1:-1, :-2]
            right = d[1:-1, 2:]

            # Neighbor counts (skip solids)
            n_up = (1.0 - solid[:-2, 1:-1]).astype(jnp.float32)
            n_down = (1.0 - solid[2:, 1:-1]).astype(jnp.float32)
            n_left = (1.0 - solid[1:-1, :-2]).astype(jnp.float32)
            n_right = (1.0 - solid[1:-1, 2:]).astype(jnp.float32)

            neighbors = up * n_up + down * n_down + left * n_left + right * n_right

            neighbor_counts = n_up + n_down + n_left + n_right

            center = density0[1:-1, 1:-1]

            # Gauss–Seidel update formula
            new_center = (center + a * neighbors) / (1.0 + a * neighbor_counts)

            # Do not modify solid cells
            new_center = interior_mask * new_center + (1 - interior_mask) * center

            # Write back into full grid
            return (d.at[1:-1, 1:-1].set(new_center), d0)

        # Run repeated iterations
        new_density, _ = jax.lax.fori_loop(
            0, num_iters, lambda _, d: gs_iteration(d), (density, density0)
        )

        # Produce new state
        return state.__class__(
            density=state.density.with_values(new_density),
            velocity=state.velocity,
            pressure=state.pressure,
            solid_mask=state.solid_mask,
            sources=state.sources,
            nu_tilde=state.nu_tilde,
            time=state.time,
            step=state.step,
        )

    @jax.jit
    def advect_density(self, state: FluidState) -> FluidState:
        """
        Advect density using semi-Lagrangian method with RK2 (midpoint) tracing.

        Uses a two-stage Runge-Kutta backward trace to halve the truncation
        error compared to 1st-order Euler, reducing numerical diffusion.

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

        # ── Stage 1: sample velocity at current positions ─────────────────────
        u1 = bilinear_interpolate(state.velocity.u, x, y - 0.5, width + 1, height)
        v1 = bilinear_interpolate(state.velocity.v, x - 0.5, y, width, height + 1)

        # Half-step backward to midpoint
        x_mid = x - 0.5 * dt0 * u1
        y_mid = y - 0.5 * dt0 * v1

        # ── Stage 2: sample velocity at midpoint ──────────────────────────────
        u2 = bilinear_interpolate(
            state.velocity.u, x_mid, y_mid - 0.5, width + 1, height
        )
        v2 = bilinear_interpolate(
            state.velocity.v, x_mid - 0.5, y_mid, width, height + 1
        )

        # Full-step backward using midpoint velocity
        x_back = x - dt0 * u2
        y_back = y - dt0 * v2

        # Clamp to valid range
        x_back = jnp.clip(x_back, 0.5, width - 0.5)
        y_back = jnp.clip(y_back, 0.5, height - 0.5)

        # Sample density at traced-back positions
        x_back_adj = x_back - 0.5
        y_back_adj = y_back - 0.5

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
            nu_tilde=state.nu_tilde,
            time=state.time,
            step=state.step,
        )

    @jax.jit
    def advect_velocity(self, state: FluidState) -> FluidState:
        """
        Advect velocity field (self-advection) using RK2 (midpoint) tracing.

        Uses a two-stage Runge-Kutta backward trace for both u and v components,
        halving the truncation error versus 1st-order Euler.

        Args:
            state: Current simulation state

        Returns:
            New state with advected velocity
        """
        height, width = self.resolution
        dt0 = self.dt / self.cell_size

        # ── Advect u-velocities (at vertical faces) ───────────────────────────
        i_u, j_u = jnp.meshgrid(
            jnp.arange(height), jnp.arange(width + 1), indexing="ij"
        )
        x_u = j_u.astype(float)
        y_u = i_u.astype(float) + 0.5

        # Stage 1: velocity at u-face positions
        u1_u = bilinear_interpolate(state.velocity.u, x_u, y_u - 0.5, width + 1, height)
        v1_u = bilinear_interpolate(state.velocity.v, x_u - 0.5, y_u, width, height + 1)

        # Half-step to midpoint
        x_u_mid = x_u - 0.5 * dt0 * u1_u
        y_u_mid = y_u - 0.5 * dt0 * v1_u

        # Stage 2: velocity at midpoint
        u2_u = bilinear_interpolate(
            state.velocity.u, x_u_mid, y_u_mid - 0.5, width + 1, height
        )
        v2_u = bilinear_interpolate(
            state.velocity.v, x_u_mid - 0.5, y_u_mid, width, height + 1
        )

        # Full-step backward using midpoint velocity
        x_u_back = x_u - dt0 * u2_u
        y_u_back = y_u - dt0 * v2_u

        new_u = bilinear_interpolate(
            state.velocity.u, x_u_back, y_u_back - 0.5, width + 1, height
        )

        # ── Advect v-velocities (at horizontal faces) ─────────────────────────
        i_v, j_v = jnp.meshgrid(
            jnp.arange(height + 1), jnp.arange(width), indexing="ij"
        )
        x_v = j_v.astype(float) + 0.5
        y_v = i_v.astype(float)

        # Stage 1: velocity at v-face positions
        u1_v = bilinear_interpolate(state.velocity.u, x_v, y_v - 0.5, width + 1, height)
        v1_v = bilinear_interpolate(state.velocity.v, x_v - 0.5, y_v, width, height + 1)

        # Half-step to midpoint
        x_v_mid = x_v - 0.5 * dt0 * u1_v
        y_v_mid = y_v - 0.5 * dt0 * v1_v

        # Stage 2: velocity at midpoint
        u2_v = bilinear_interpolate(
            state.velocity.u, x_v_mid, y_v_mid - 0.5, width + 1, height
        )
        v2_v = bilinear_interpolate(
            state.velocity.v, x_v_mid - 0.5, y_v_mid, width, height + 1
        )

        # Full-step backward using midpoint velocity
        x_v_back = x_v - dt0 * u2_v
        y_v_back = y_v - dt0 * v2_v

        new_v = bilinear_interpolate(
            state.velocity.v, x_v_back - 0.5, y_v_back, width, height + 1
        )

        return state.__class__(
            density=state.density,
            velocity=state.velocity.with_values(new_u, new_v),
            pressure=state.pressure,
            solid_mask=state.solid_mask,
            sources=state.sources,
            nu_tilde=state.nu_tilde,
            time=state.time,
            step=state.step,
        )

    @jax.jit(static_argnames=("num_iters",))
    def diffuse_velocity(
        self,
        state: FluidState,
        num_iters: int = 50,
        nu_eff_field: Optional[Array] = None,
    ) -> FluidState:
        """
        Perform Gauss–Seidel diffusion on the velocity field (implicit viscosity).
        Should be applied before advection/projection.

        Args:
            state:         Current simulation state
            num_iters:     Number of Gauss–Seidel iterations
            nu_eff_field:  Optional spatially-varying effective kinematic viscosity
                           (h, w) array in m²/s.  When provided (SA turbulence mode),
                           each cell uses its local value; otherwise the scalar
                           self.viscosity is used uniformly.
                           All paths are pure jnp — fully differentiable.

        Returns:
            New state with diffused velocity.
        """
        # Short-circuit if no viscosity at all
        if self.viscosity <= 0.0 and nu_eff_field is None:
            return state

        dt = self.dt
        nu = self.viscosity
        cs = self.cell_size
        solid = state.solid_mask
        height, width = self.resolution

        # Build spatially-varying diffusion factor a(i,j) = dt * nu_eff(i,j) / h²
        # If nu_eff_field is None, use scalar nu uniformly (backward compatible).
        if nu_eff_field is None:
            a_cc = jnp.full((height, width), dt * nu / (cs * cs))
        else:
            a_cc = dt * nu_eff_field / (cs * cs)  # (h, w) — differentiable

        u0 = state.velocity.u
        v0 = state.velocity.v

        # Face masks (zero out velocity adjacent to solid)
        u_interior_mask = jnp.ones_like(u0)
        solid_u_left = solid[:, :-1]
        solid_u_right = solid[:, 1:]
        u_interior_mask = u_interior_mask.at[:, 1:-1].set(
            (1.0 - jnp.logical_or(solid_u_left, solid_u_right)).astype(jnp.float32)
        )

        v_interior_mask = jnp.ones_like(v0)
        solid_v_top = solid[:-1, :]
        solid_v_bottom = solid[1:, :]
        v_interior_mask = v_interior_mask.at[1:-1, :].set(
            (1.0 - jnp.logical_or(solid_v_top, solid_v_bottom)).astype(jnp.float32)
        )

        # Diffusion factor interpolated to u / v faces (average of two cell centres)
        # u-face (i, j): between cell [i, j-1] and [i, j]  → average a_cc cols
        a_u = jnp.zeros_like(u0)
        a_u = a_u.at[:, 1:-1].set(0.5 * (a_cc[:, :-1] + a_cc[:, 1:]))  # interior faces
        a_u = a_u.at[:, 0].set(a_cc[:, 0])  # left boundary
        a_u = a_u.at[:, -1].set(a_cc[:, -1])  # right boundary

        # v-face (i, j): between cell [i-1, j] and [i, j]  → average a_cc rows
        a_v = jnp.zeros_like(v0)
        a_v = a_v.at[1:-1, :].set(0.5 * (a_cc[:-1, :] + a_cc[1:, :]))  # interior faces
        a_v = a_v.at[0, :].set(a_cc[0, :])  # top boundary
        a_v = a_v.at[-1, :].set(a_cc[-1, :])  # bottom boundary

        def gs_u(carry, _):
            u, u_init = carry
            a = a_u[1:-1, 1:-1]  # (h, w-1) interior

            up = u[:-2, 1:-1]
            down = u[2:, 1:-1]
            left = u[1:-1, :-2]
            right = u[1:-1, 2:]

            neighbors = up + down + left + right
            new_u_c = (u_init[1:-1, 1:-1] + a * neighbors) / (1.0 + 4.0 * a)

            mask = u_interior_mask[1:-1, 1:-1]
            new_u_c = mask * new_u_c + (1.0 - mask) * u[1:-1, 1:-1]
            return (u.at[1:-1, 1:-1].set(new_u_c), u_init), None

        def gs_v(carry, _):
            v, v_init = carry
            a = a_v[1:-1, 1:-1]  # (h-1, w) interior

            up = v[:-2, 1:-1]
            down = v[2:, 1:-1]
            left = v[1:-1, :-2]
            right = v[1:-1, 2:]

            neighbors = up + down + left + right
            new_v_c = (v_init[1:-1, 1:-1] + a * neighbors) / (1.0 + 4.0 * a)

            mask = v_interior_mask[1:-1, 1:-1]
            new_v_c = mask * new_v_c + (1.0 - mask) * v[1:-1, 1:-1]
            return (v.at[1:-1, 1:-1].set(new_v_c), v_init), None

        # Use jax.lax.scan (differentiable) instead of fori_loop
        (u_new, _), _ = jax.lax.scan(gs_u, (u0, u0), None, length=num_iters)
        (v_new, _), _ = jax.lax.scan(gs_v, (v0, v0), None, length=num_iters)

        # Enforce exact solid boundaries via IBM continuous forcing
        u_new, v_new = apply_ibm_continuous_forcing(u_new, v_new, state.solid_mask)

        return state.__class__(
            density=state.density,
            velocity=state.velocity.with_values(u_new, v_new),
            pressure=state.pressure,
            solid_mask=state.solid_mask,
            sources=state.sources,
            nu_tilde=state.nu_tilde,
            time=state.time,
            step=state.step,
        )

    @jax.jit
    def inject_wind_tunnel_velocity(self, state: FluidState) -> FluidState:
        """
        Inject velocity at the left inlet boundary for wind tunnel mode.
        This creates inflow conditions without forcing uniform flow everywhere.

        Args:
            state: Current simulation state

        Returns:
            New state with wind tunnel velocity injected at inlet
        """
        if not self.is_wind_tunnel:
            return state

        # Only set velocity at the LEFT BOUNDARY (inlet), not everywhere
        # This allows the flow to develop naturally and create pressure gradients
        u = state.velocity.u
        v = state.velocity.v

        # Set inlet velocity at left boundary (first few columns)
        inlet_velocity = self.inlet_velocity
        angle = self.inlet_angle_rad
        inlet_u = inlet_velocity * jnp.cos(angle)
        inlet_v = -inlet_velocity * jnp.sin(angle)
        # For staggered grid: u has shape (height, width+1)
        # Set u at the leftmost vertical faces (column 0 and 1)
        u = u.at[:, 0:2].set(inlet_u)

        # Set v at the leftmost horizontal faces (column 0)
        v = v.at[:, 0:1].set(inlet_v)

        # Keep v unchanged (horizontal flow, so v should remain small)

        return state.__class__(
            density=state.density,
            velocity=state.velocity.with_values(u, v),
            pressure=state.pressure,
            solid_mask=state.solid_mask,
            sources=state.sources,
            nu_tilde=state.nu_tilde,
            time=state.time,
            step=state.step,
        )

    @jax.jit
    def compute_divergence(self, state: FluidState) -> Array:
        """
        Compute divergence of volume flux for Immersed Boundary Method.

        Args:
            state: Current simulation state

        Returns:
            Divergence field, shape (height, width)
        """
        u = state.velocity.u
        v = state.velocity.v
        solid = state.solid_mask

        # Fractional fluid masks at faces for volume-penalised divergence
        fluid_u = jnp.ones((self.height, self.width + 1))
        fluid_u = fluid_u.at[:, 1:-1].set(1.0 - 0.5 * (solid[:, :-1] + solid[:, 1:]))
        fluid_u = fluid_u.at[:, 0].set(1.0 - solid[:, 0])
        fluid_u = fluid_u.at[:, -1].set(1.0 - solid[:, -1])

        fluid_v = jnp.ones((self.height + 1, self.width))
        fluid_v = fluid_v.at[1:-1, :].set(1.0 - 0.5 * (solid[:-1, :] + solid[1:, :]))
        fluid_v = fluid_v.at[0, :].set(1.0 - solid[0, :])
        fluid_v = fluid_v.at[-1, :].set(1.0 - solid[-1, :])

        div = (
            u[:, 1:] * fluid_u[:, 1:] - u[:, :-1] * fluid_u[:, :-1]
        ) / self.cell_size + (
            v[1:, :] * fluid_v[1:, :] - v[:-1, :] * fluid_v[:-1, :]
        ) / self.cell_size

        return div

    @jax.jit(static_argnames=("num_iters",))
    def solve_pressure(self, state: FluidState, num_iters: int = 80) -> FluidState:
        """
        Solve for pressure using Gauss-Seidel iterations.

        Handles partial masking where solid_mask values between 0 and 1
        represent fractional solid coverage for sub-grid resolution obstacles.

        Args:
            state: Current simulation state
            num_iters: Number of Gauss-Seidel iterations

        Returns:
            New state with updated pressure field
        """
        divergence = self.compute_divergence(state)
        solid = state.solid_mask

        # Fractional fluid masks at faces for volume-penalised pressure Poisson
        # These represent the exact area fraction of the face open to fluid
        fluid_u = jnp.ones((self.height, self.width + 1))
        fluid_u = fluid_u.at[:, 1:-1].set(1.0 - 0.5 * (solid[:, :-1] + solid[:, 1:]))
        fluid_u = fluid_u.at[:, 0].set(1.0 - solid[:, 0])
        fluid_u = fluid_u.at[:, -1].set(1.0 - solid[:, -1])

        fluid_v = jnp.ones((self.height + 1, self.width))
        fluid_v = fluid_v.at[1:-1, :].set(1.0 - 0.5 * (solid[:-1, :] + solid[1:, :]))
        fluid_v = fluid_v.at[0, :].set(1.0 - solid[0, :])
        fluid_v = fluid_v.at[-1, :].set(1.0 - solid[-1, :])

        # Warm-start from the previous pressure
        pressure = state.pressure.values
        scale = self.rho * self.cell_size * self.cell_size / self.dt

        @jax.jit
        def gs_iteration(p_div):
            p, div = p_div

            # Gather fluid-neighbour pressures via padding (zero outside domain → Neumann)
            left_p = jnp.pad(p[:, :-1], ((0, 0), (1, 0)))
            left_mask = fluid_u[:, :-1]

            right_p = jnp.pad(p[:, 1:], ((0, 0), (0, 1)))
            right_mask = fluid_u[:, 1:]

            up_p = jnp.pad(p[:-1, :], ((1, 0), (0, 0)))
            up_mask = fluid_v[:-1, :]

            down_p = jnp.pad(p[1:, :], ((0, 1), (0, 0)))
            down_mask = fluid_v[1:, :]

            neighbor_sum = (
                left_p * left_mask
                + right_p * right_mask
                + up_p * up_mask
                + down_p * down_mask
            )
            neighbor_count = left_mask + right_mask + up_mask + down_mask

            # GS update: p_c = (Σ fractional_neighbour_pressures − scale·div) / N_fractional
            new_p = jnp.where(
                neighbor_count > 1e-6,
                (neighbor_sum - div * scale) / neighbor_count,
                0.0,
            )
            # Pressure is zero inside fully solid cells
            new_p = jnp.where(solid >= 0.999, 0.0, new_p)

            return (new_p, div)

        pressure, _ = jax.lax.fori_loop(
            0, num_iters, lambda _, p: gs_iteration(p), (pressure, divergence)
        )

        return state.__class__(
            density=state.density,
            velocity=state.velocity,
            pressure=state.pressure.with_values(pressure),
            solid_mask=state.solid_mask,
            sources=state.sources,
            nu_tilde=state.nu_tilde,
            time=state.time,
            step=state.step,
        )

    @jax.jit
    def project_velocity(self, state: FluidState) -> FluidState:
        """
        Make velocity field divergence-free by subtracting pressure gradient.

        Handles partial masking by scaling pressure gradients by the average
        fluid fraction at face boundaries.

        Args:
            state: Current simulation state (with solved pressure)

        Returns:
            New state with projected velocity
        """
        pressure = state.pressure.values
        u = state.velocity.u
        v = state.velocity.v
        solid = state.solid_mask
        h = self.cell_size
        rho = self.rho
        dt = self.dt

        # Correct fractional-step projection:
        #   u -= (dt/ρ) · ∂p/∂x
        #   v -= (dt/ρ) · ∂p/∂y
        # No fluid-fraction weighting — that was halving the correction at every
        # wall-adjacent face.  No-penetration is enforced afterwards by
        # apply_ibm_continuous_forcing, which is the proper way to handle it.

        # Interior u-faces: u[i, j] is between cell [i, j-1] and [i, j]
        dp_dx = (pressure[:, 1:] - pressure[:, :-1]) / h  # shape (H, W-1)
        new_u = u.at[:, 1:-1].add(-dt / rho * dp_dx)

        # Interior v-faces: v[i, j] is between cell [i-1, j] and [i, j]
        # (y increases downward in grid indexing)
        dp_dy = (pressure[1:, :] - pressure[:-1, :]) / h  # shape (H-1, W)
        new_v = v.at[1:-1, :].add(-dt / rho * dp_dy)

        new_u, new_v = apply_ibm_continuous_forcing(new_u, new_v, solid)

        return state.__class__(
            density=state.density,
            velocity=state.velocity.with_values(new_u, new_v),
            pressure=state.pressure,
            solid_mask=state.solid_mask,
            sources=state.sources,
            nu_tilde=state.nu_tilde,
            time=state.time,
            step=state.step,
        )

    @timeit
    @jax.jit
    def step(self, state: FluidState) -> FluidState:
        """
        Advance simulation by one time step.

        Performs:
        1. Add density sources
        2. Diffuse density
        3. Advect density
        4. [SA] Advance SA turbulence transport (ν̃)
        5. [SA] Compute effective viscosity  ν_eff = ν + ν_t
        6. Diffuse velocity  (using ν_eff when SA is active)
        7. Advect velocity
        8. Inject wind tunnel velocity (if active)
        9. Solve pressure
        10. Project velocity (enforce incompressibility)

        Args:
            state: Current simulation state.
                   NOTE: state.solid_mask is used for differentiability.
                   For shape optimization, update state.solid_mask before
                   calling step() to enable gradient flow through shape params.

        Returns:
            New state after one time step
        """
        # Density step
        state = self.add_sources_to_density(state)
        state = self.diffuse_density(state)
        state = self.advect_density(state)

        # ── Spalart-Allmaras turbulence step ──────────────────────────────────
        # Fully differentiable: all SA ops are pure jnp.  The Python-level
        # `use_sa_turbulence` flag causes JAX to compile two specialised traces
        # (SA-on / SA-off) — no dynamic branching inside the traced graph.
        if self.use_sa_turbulence and state.nu_tilde is not None:
            state = self.step_sa_turbulence(state, self._wall_dist, num_diff_iters=4)
            nu_eff = self.compute_effective_viscosity(state)
        else:
            nu_eff = None

        # Velocity step — pass nu_eff so turbulent diffusion uses ν_eff spatially
        state = self.diffuse_velocity(state, num_iters=20, nu_eff_field=nu_eff)
        state = self.advect_velocity(state)

        # Inject wind tunnel velocity if in wind tunnel mode
        state = self.inject_wind_tunnel_velocity(state)

        # Pressure projection (enforce incompressibility)
        state = self.solve_pressure(state, num_iters=30)
        state = self.project_velocity(state)

        # Update time and step
        new_state = state.__class__(
            density=state.density,
            velocity=state.velocity,
            pressure=state.pressure,
            solid_mask=state.solid_mask,
            sources=state.sources,
            nu_tilde=state.nu_tilde,
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
            inlet_u = self.inlet_velocity * jnp.cos(self.inlet_angle_rad)
            inlet_v = -self.inlet_velocity * jnp.sin(self.inlet_angle_rad)
            u = jnp.ones((height, width + 1)) * inlet_u
            v = jnp.ones((height + 1, width)) * inlet_v
            # Mark as wind tunnel mode
            self.is_wind_tunnel = True
            # Make right boundary non-solid
            self.solid_mask = self.solid_mask.at[1:-1, -1].set(False)

        else:
            raise ValueError(f"Unknown field_type: {field_type}")

        # Apply solid boundary conditions
        u, v = apply_ibm_continuous_forcing(u, v, self.solid_mask)

        return state.__class__(
            density=state.density,
            velocity=state.velocity.with_values(u, v),
            pressure=state.pressure,
            solid_mask=state.solid_mask,
            sources=state.sources,
            nu_tilde=state.nu_tilde,
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
            nu_tilde=state.nu_tilde,
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
                    shade = solid_mask_array[i, j] * 255
                    color = (shade, shade, shade)
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

            # Dynamic color scaling based on mean velocity magnitude
            u_centers = (u_array[:, :-1] + u_array[:, 1:]) / 2.0
            v_centers = (v_array[:-1, :] + v_array[1:, :]) / 2.0
            mags_all = np.sqrt(u_centers**2 + v_centers**2)

            # Use mean velocity of non-solid cells for scaling if available
            fluid_mags = (
                mags_all[solid_mask_array == 0]
                if solid_mask_array is not None
                else mags_all
            )
            mean_vel = np.mean(fluid_mags) if fluid_mags.size > 0 else 0.0
            # Scale so that 2x mean velocity is red, mean is green, 0 is blue
            v_scale = 2.0 * mean_vel if mean_vel > 1e-4 else 1.0

            if self.show_cell_centered_velocity:
                # Cell Centered Velocity
                # Normalized directions (avoid division by zero)
                mag_nonzero = mags_all > 1e-6
                mag_dir_x = np.divide(
                    u_centers,
                    mags_all,
                    out=np.zeros_like(u_centers),
                    where=mag_nonzero,
                )
                mag_dir_y = np.divide(
                    v_centers,
                    mags_all,
                    out=np.zeros_like(v_centers),
                    where=mag_nonzero,
                )

                for i in range(self.height):
                    for j in range(self.width):
                        mag = float(mags_all[i, j])
                        mag_dir_x_val = float(mag_dir_x[i, j])
                        mag_dir_y_val = float(mag_dir_y[i, j])

                        # Color based on magnitude relative to mean
                        norm = min(1.0, mag / v_scale)
                        if norm < 0.5:
                            # Blue to Green
                            t = norm * 2.0
                            color = (0, int(255 * t), int(255 * (1 - t)))
                        else:
                            # Green to Red
                            t = (norm - 0.5) * 2.0
                            color = (int(255 * t), int(255 * (1 - t)), 0)

                        # Draw arrow (length is fixed scaling, arrowhead scales with norm)
                        scale = self.display_size * 0.4
                        start_x = (j + 0.5) * self.display_size
                        start_y = (i + 0.5) * self.display_size
                        end_x = start_x + scale * mag_dir_x_val
                        end_y = start_y + scale * mag_dir_y_val

                        pygame.draw.aaline(
                            self.screen, color, (start_x, start_y), (end_x, end_y), 2
                        )

                        # Draw arrowhead scaled by normalized magnitude
                        angle = math.atan2(mag_dir_y_val, mag_dir_x_val)
                        tip_len = scale * norm / 2
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
                        u_val = float(u_array[i, j])

                        # Color based on magnitude relative to mean
                        norm = min(1.0, abs(u_val) / v_scale)
                        if norm < 0.5:
                            t = norm * 2.0
                            color = (0, int(255 * t), int(255 * (1 - t)))
                        else:
                            t = (norm - 0.5) * 2.0
                            color = (int(255 * t), int(255 * (1 - t)), 0)

                        scale = self.display_size * 0.4
                        start_x = j * self.display_size
                        end_x = start_x + scale * (
                            u_val / v_scale if abs(u_val) > 1e-6 else 0
                        )
                        y = (i + 0.5) * self.display_size

                        pygame.draw.aaline(
                            self.screen, color, (start_x, y), (end_x, y), 2
                        )

                        # Draw arrowhead scaled by normalized magnitude
                        angle = 0 if u_val > 0 else math.pi
                        tip_len = scale * norm / 2
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
                        v_val = float(v_array[i, j])

                        # Color based on magnitude relative to mean
                        norm = min(1.0, abs(v_val) / v_scale)
                        if norm < 0.5:
                            t = norm * 2.0
                            color = (0, int(255 * t), int(255 * (1 - t)))
                        else:
                            t = (norm - 0.5) * 2.0
                            color = (int(255 * t), int(255 * (1 - t)), 0)

                        scale = self.display_size * 0.4
                        x = (j + 0.5) * self.display_size
                        start_y = i * self.display_size
                        end_y = start_y + scale * (
                            v_val / v_scale if abs(v_val) > 1e-6 else 0
                        )

                        pygame.draw.aaline(
                            self.screen, color, (x, start_y), (x, end_y), 2
                        )

                        # Draw arrowhead scaled by normalized magnitude
                        angle = math.pi / 2 if v_val > 0 else -math.pi / 2
                        tip_len = scale * norm / 2
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

        # Draw shape border
        for i, j in self.solid_border:
            x = j * 1
            y = i * 1
            pygame.draw.rect(
                self.screen,
                (255, 191, 0),
                pygame.Rect(x, y, 1, 1),
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
        # Draw control volume border if specified
        if self.cv_rect is not None:
            i1, i2, j1, j2 = self.cv_rect
            # j is horizontal (x), i is vertical (y)
            # rect = (x, y, width, height)
            rect = pygame.Rect(
                j1 * self.display_size,
                i1 * self.display_size,
                (j2 - j1) * self.display_size,
                (i2 - i1) * self.display_size,
            )
            # Draw with a distinct color (green) and thick border
            pygame.draw.rect(self.screen, (0, 255, 0), rect, 2)

        pygame.display.flip()

    def simulate(
        self,
        state: FluidState,
        steps: int = -1,
        custom_step_fn: Optional[
            Callable[["FluidGrid", FluidState], FluidState]
        ] = None,
        callback_fn: Optional[Callable[["FluidGrid", FluidState], None]] = None,
    ) -> FluidState:
        """
        Run simulation loop with visualization.

        Args:
            state: Initial simulation state
            steps: Number of steps (-1 for infinite)
            custom_step_fn: Optional custom function to advance simulation
            callback_fn: Optional function called after each step

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
            if custom_step_fn is not None:
                state = custom_step_fn(self, state)
            else:
                state = self.step(state)

            if callback_fn is not None:
                if callback_fn(self, state):
                    break

            if self.visualise:
                self.draw_state(state)
                self.clock.tick(30)

            step += 1

        if self.visualise:
            pygame.quit()

        return state


# ============================================================================
# JAX PyTree Registration
# ============================================================================
# Register FluidState as a JAX pytree to enable automatic differentiation
# through fluid simulation steps.


def _fluid_state_flatten(state):
    """Flatten FluidState for JAX transformation.

    nu_tilde is an Optional[Grid].  We include it as a leaf so that:
    1. Gradients flow through ν̃ (differentiable SA model).
    2. JAX compiles separate traces for SA-on (nu_tilde=Grid) and SA-off
       (nu_tilde=None), keeping the pytree structure consistent within each.
    """
    children = (
        state.density,
        state.velocity,
        state.pressure,
        state.solid_mask,
        state.sources,
        state.nu_tilde,  # Optional[Grid] — None or Grid
        state.time,
        state.step,
    )
    metadata = None
    return children, metadata


def _fluid_state_unflatten(metadata, children):
    """Reconstruct FluidState from flattened representation."""
    density, velocity, pressure, solid_mask, sources, nu_tilde, time, step = children
    return FluidState(
        density=density,
        velocity=velocity,
        pressure=pressure,
        solid_mask=solid_mask,
        sources=sources,
        nu_tilde=nu_tilde,
        time=time,
        step=step,
    )


jax.tree_util.register_pytree_node(
    FluidState, _fluid_state_flatten, _fluid_state_unflatten
)


def _fluid_grid_flatten(grid):
    """Flatten FluidGrid for JAX transformation.

    Treats the solid_mask as dynamic data that can be differentiated,
    while storing all other configuration as static metadata.
    """
    children = (grid.solid_mask,)
    metadata = {
        "height": grid.height,
        "width": grid.width,
        "cell_size": grid.cell_size,
        "dt": grid.dt,
        "diffusion": grid.diffusion,
        "viscosity": grid.viscosity,
        "visualise": grid.visualise,
        "show_cell_property": grid.show_cell_property,
        "show_velocity": grid.show_velocity,
        "show_cell_centered_velocity": grid.show_cell_centered_velocity,
        "is_wind_tunnel": grid.is_wind_tunnel,
        "cv_rect": grid.cv_rect,
    }
    return children, metadata


def _fluid_grid_unflatten(metadata, children):
    """Reconstruct FluidGrid from flattened representation."""
    (solid_mask,) = children

    # Create a new FluidGrid with the stored configuration
    # We need to create it without SDF since we already have the solid_mask
    grid = FluidGrid(
        height=metadata["height"],
        width=metadata["width"],
        cell_size=metadata["cell_size"],
        dt=metadata["dt"],
        diffusion=metadata["diffusion"],
        viscosity=metadata["viscosity"],
        boundary_type=0,  # Use no boundary since we have the mask
        sdf=None,
        visualise=metadata["visualise"],
        show_cell_property=metadata["show_cell_property"],
        show_velocity=metadata["show_velocity"],
        show_cell_centered_velocity=metadata["show_cell_centered_velocity"],
        cv_rect=metadata.get("cv_rect"),
    )

    # Replace the solid_mask with the one from children
    grid.solid_mask = solid_mask
    grid.is_wind_tunnel = metadata["is_wind_tunnel"]

    return grid


jax.tree_util.register_pytree_node(
    FluidGrid, _fluid_grid_flatten, _fluid_grid_unflatten
)


__all__ = [
    "FluidGrid",
    "FluidState",
]
