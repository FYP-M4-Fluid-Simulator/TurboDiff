"""
RANS Airfoil Shape Optimization — CST + Spalart-Allmaras
=========================================================
Combines gradient-based CST optimization with the full RANS simulation setup
to find an airfoil that maximises lift-to-drag ratio under turbulent flow.

The final optimized shape is exported as a Selig-format .dat file that can be
fed directly into xfoil_validation.py for independent XFoil verification.

Physics conventions
-------------------
  - Chord length  c = 1.0 m  (fixed)
  - Free-stream   U_inf = 1.0 m/s  (fixed)
  - Viscosity     ν = 1 / Re  (so Re = U_inf · c / ν)
  - AoA rotates the FREE-STREAM velocity vector:
        u_inlet = U_inf · cos(AoA)
        v_inlet = U_inf · sin(AoA)
  - All four domain boundaries carry the same free-stream BC (no hard walls).

Gradient strategy
-----------------
  Gradients are computed with jax.grad (exact, analytic) rather than finite
  differences.  This requires the full loss computation — mask building,
  RANS time integration, and force summation — to be a single pure-JAX
  expression that JAX's autodiff engine can trace end-to-end.

  The solid mask is built from the CST weights using jnp operations only,
  so gradients flow from the pressure/viscous forces back through the
  velocity field, through the IBM forcing, and into the mask geometry,
  and from there into the CST weights.

  Wall distance (needed for the SA turbulence model) is *not* differentiated:
  it is computed once from the thresholded mask via scipy EDT and treated as
  a fixed constant (jax.lax.stop_gradient).  This is a reasonable
  approximation because turbulent viscosity changes are slow compared to the
  pressure-force gradient signal.

Setup:
  - CST parametrization : N_ORDER = 5 (6 weights per surface)
  - Grid                : HEIGHT × WIDTH cells, cell_size = 0.005 m
  - RANS model          : Spalart-Allmaras one-equation (SA-1994)
  - Objective           : minimise drag (with lift & geometry constraints)

Output:
  - optimized_airfoil.dat   — Selig-format coordinates for XFoil
  - rans_opt_history.png    — loss / Cl / Cd versus iteration
  - rans_opt_shapes.png     — initial vs final airfoil overlaid

Run:
  cd /Users/musab/FYP/TurboDiff
  .venv/bin/python examples/optimize_airfoil_rans.py \\
      --re 2e6 --aoa 4

  # Re options : 5e5 | 2e6 | 6e6
  # AoA is any float (degrees).
"""

import argparse
import math
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.special import comb as sp_comb
import matplotlib

matplotlib.use("Agg")  # headless — no display needed
import matplotlib.pyplot as plt

from turbodiff.core.fluid_grid_jax import FluidGrid, FluidState
from turbodiff.core.utils import apply_ibm_continuous_forcing
from turbodiff.core.airfoil import (
    generate_cst_coords,
    thickness_at_x,
    max_thickness,
)
from turbodiff.core.loss_functions import (
    thickness_constraint_loss,
    crossover_validity_loss,
)

# ─────────────────────────────────────────────────────────────────────────────
# Persistent JIT-compiled infrastructure
# ─────────────────────────────────────────────────────────────────────────────
# Keyed by (u_inlet_rounded, v_inlet_rounded, nu_lam_rounded).
# Stores (FluidGrid, rans_step) so every call to evaluate_rans reuses the
# same compiled objects.
_GRID_CACHE: dict = {}


# ─────────────────────────────────────────────────────────────────────────────
# Hyper-parameters  (geometry / simulation budget only — physics set by CLI)
# ─────────────────────────────────────────────────────────────────────────────

# Grid / domain
HEIGHT = 128  # cells in y
WIDTH = 384  # cells in x
CELL_SIZE = 0.01  # m  → domain 0.32 m × 0.96 m

# RANS physics (fixed)
DT = 0.0002  # time-step (s) — CFL ≈ 0.08 at U=1, h=2.5 mm
RHO = 1.0  # density (kg/m³)
U_INF = 1.0  # free-stream speed (m/s)  – FIXED
CHORD = 1.0  # chord length (m)         – FIXED (ν = U_INF * CHORD / Re)

# Airfoil placement (leading-edge position inside the domain, in metres)
AIRFOIL_X0 = 0.4  # m  — x of leading edge
AIRFOIL_Y0 = HEIGHT * CELL_SIZE / 2.0  # m  — vertical midpoint

# Simulation budget per gradient evaluation
N_SIM_STEPS = 1000  # steps run per loss evaluation (force evaluated on final state)
N_WARMUP = 0  # unused (kept for variable definition)
N_AVG_STEPS = 1000  # unused (kept for variable definition)

# Mask sub-sampling (anti-aliasing)
N_SUB = 3  # sub-cell samples per direction

# Optimization
NUM_ITERATIONS = 10
LEARNING_RATE = 0.005
N_ORDER = 5  # CST polynomial order — 6 weights per surface

# Geometry constraints
MIN_THICKNESS_RATIO = 0.10  # t/c must exceed this somewhere
MAX_THICKNESS_RATIO = 0.25  # t/c must not exceed this anywhere

# Loss weights
W_DRAG = 1.0  # drag minimization
W_LIFT = 0.5  # soft lift-maximization bonus
W_GEO = 10.0  # geometric feasibility

# Output files  — will be written next to this script
OUTPUT_DAT = "optimized_airfoil.dat"
OUTPUT_HIST = "rans_opt_history.png"
OUTPUT_SHAPE = "rans_opt_shapes.png"


# ─────────────────────────────────────────────────────────────────────────────
# CLI argument parsing
# ─────────────────────────────────────────────────────────────────────────────

_VALID_RE = {
    "5e5": 5e5,
    "2e6": 2e6,
    "6e6": 6e6,
}


def parse_args():
    """Parse command-line arguments for Re and AoA."""
    parser = argparse.ArgumentParser(
        description="RANS CST Airfoil Optimisation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--re",
        choices=list(_VALID_RE.keys()),
        default="2e6",
        metavar="{5e5|2e6|6e6}",
        help=(
            "Reynolds number.  Chord and velocity are fixed at 1; "
            "kinematic viscosity is derived as ν = 1/Re."
        ),
    )
    parser.add_argument(
        "--aoa",
        type=float,
        default=0.0,
        metavar="DEG",
        help=(
            "Angle of attack in degrees.  Rotates the free-stream velocity "
            "vector; all domain boundaries carry the same free-stream BC."
        ),
    )
    parser.add_argument(
        "--visualise",
        default=False,
        action="store_true",
        help=(
            "Launch a Pygame window to visualise the airflow on the initial shape. "
            "Disables optimization."
        ),
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# JAX-native solid-mask builder
#
# This version uses only jnp operations so that jax.grad can differentiate
# through it.  The key difference vs the old numpy version:
#   - the per-subcell accumulation is a jax.lax.fori_loop
#   - "inside" is a smooth soft-mask via jnp.where instead of boolean comparison
#   - no Python conditionals — JAX traces the whole thing symbolically
#
# The gradient ∂mask/∂weights flows through  y_u_g = C_g * (B_g @ weights)
# and the comparison against cy_body (which is treated as constant w.r.t.
# weights because the grid coordinates do not depend on weights).
# ─────────────────────────────────────────────────────────────────────────────

# Pre-compute geometry constants that never change (depend only on grid params)
_JJ, _II = np.meshgrid(np.arange(WIDTH), np.arange(HEIGHT))  # (H, W), numpy
_XI_COORDS: list[tuple] = []  # (cx_body, cy_body, xi_grid, xi_c, C_g, B_g) per sub


def _precompute_subcell_geometry(aoa_deg: float) -> list[dict]:
    """
    Pre-compute all grid/geometry arrays (everything that doesn't depend on
    CST weights) for the given AoA.  Called once per (AoA) combination and
    cached.  Returns a list of dicts, one per sub-cell sample.
    """
    # Note: earlier we calculated cos_a/sin_a here, but we now keep the airfoil
    # un-rotated relative to the grid, and only rotate the free-stream velocity vector.
    n_order = N_ORDER
    i_vals = np.arange(n_order + 1)
    binom = np.array(
        [sp_comb(n_order, i, exact=True) for i in i_vals], dtype=np.float32
    )

    subcells = []
    for si in range(N_SUB):
        for sj in range(N_SUB):
            cx = (_JJ + (sj + 0.5) / N_SUB) * CELL_SIZE
            cy = (_II + (si + 0.5) / N_SUB) * CELL_SIZE
            cx_rel = cx - AIRFOIL_X0
            cy_rel = cy - AIRFOIL_Y0
            # Airfoil is aligned with grid; wind velocity is rotated instead.
            cx_body = cx_rel.astype(np.float32)
            cy_body = cy_rel.astype(np.float32)
            xi_grid = (cx_body / CHORD).astype(np.float32)
            xi_c = np.clip(xi_grid, 0.0, 1.0).astype(np.float32)
            C_g = (np.sqrt(np.maximum(xi_c, 0.0)) * (1.0 - xi_c)).astype(np.float32)
            xi_c_col = xi_c[..., None]  # (H, W, 1)
            B_g = (
                binom * xi_c_col**i_vals * (1.0 - xi_c_col) ** (n_order - i_vals)
            ).astype(
                np.float32
            )  # (H, W, N+1)
            subcells.append(
                dict(
                    cx_body=jnp.array(cx_body),
                    cy_body=jnp.array(cy_body),
                    xi_grid=jnp.array(xi_grid),
                    C_g=jnp.array(C_g),
                    B_g=jnp.array(B_g),
                )
            )
    return subcells


_SUBCELL_CACHE: dict = {}


def _get_subcell_geometry(aoa_deg: float) -> list[dict]:
    key = round(aoa_deg, 6)
    if key not in _SUBCELL_CACHE:
        _SUBCELL_CACHE[key] = _precompute_subcell_geometry(aoa_deg)
    return _SUBCELL_CACHE[key]


def build_solid_mask_jax(
    weights_upper: jnp.ndarray,  # (N+1,)  — must be JAX arrays for grad
    weights_lower: jnp.ndarray,  # (N+1,)
    subcells: list[dict],
) -> jnp.ndarray:
    """
    Differentiable solid-mask builder.

    All geometry arrays (cx_body, cy_body, C_g, B_g, xi_grid) are JAX
    constants pre-computed by _precompute_subcell_geometry.  Only the CST
    weight-to-surface computation is differentiated.

    Uses a *soft* inside indicator:
        inside = σ(sharpness * (y_upper_body - cy_body))
                * σ(sharpness * (cy_body - y_lower_body))
                * in_chord

    where σ is the sigmoid.  The sharpness = 500 makes this numerically
    identical to the hard step for cells well inside/outside the airfoil,
    but provides a smooth gradient at the surface (the IBM boundary).

    Returns:
        solid_mask : jnp float32 array (HEIGHT, WIDTH), values in [0, 1]
    """
    sharpness = 500.0
    scale = CHORD
    solid_accum = jnp.zeros((HEIGHT, WIDTH), dtype=jnp.float32)

    for sc in subcells:
        cy_body = sc["cy_body"]  # (H, W) constant
        C_g = sc["C_g"]  # (H, W) constant
        B_g = sc["B_g"]  # (H, W, N+1) constant
        xi_grid = sc["xi_grid"]  # (H, W) constant

        y_u = C_g * (B_g @ weights_upper)  # (H, W)  — differentiable
        y_l = C_g * (B_g @ weights_lower)  # (H, W)  — differentiable

        y_upper_body = y_u * scale  # physical surface y in body frame
        y_lower_body = y_l * scale

        # Soft "inside" indicator
        above_lower = jax.nn.sigmoid(sharpness * (cy_body - y_lower_body))
        below_upper = jax.nn.sigmoid(sharpness * (y_upper_body - cy_body))
        in_chord = jax.nn.sigmoid(sharpness * xi_grid) * jax.nn.sigmoid(
            sharpness * (1.0 - xi_grid)
        )

        inside = above_lower * below_upper * in_chord
        solid_accum = solid_accum + inside

    return solid_accum / (N_SUB * N_SUB)


# ─────────────────────────────────────────────────────────────────────────────
# Wall-distance helper (NOT differentiated — scipy EDT on thresholded mask)
# ─────────────────────────────────────────────────────────────────────────────


def compute_wall_dist_np(solid_mask_jax: jnp.ndarray) -> jnp.ndarray:
    """
    Compute wall distance from a JAX solid mask using scipy EDT.
    Returns a JAX array wrapped in stop_gradient so autodiff ignores it.
    """
    solid_np = np.array(solid_mask_jax)
    solid_np_open = solid_np.copy()
    solid_np_open[1:-1, -1] = 0.0  # open right edge for outflow
    wall_dist_np = distance_transform_edt(solid_np_open < 0.5) * CELL_SIZE
    wall_dist_np = np.maximum(wall_dist_np.astype(np.float32), 1e-10)
    # stop_gradient: wall_dist is a fixed input to the differentiable graph
    return jax.lax.stop_gradient(jnp.array(wall_dist_np))


# ─────────────────────────────────────────────────────────────────────────────
# Free-stream boundary condition helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_apply_freestream_bc(u_inlet: float, v_inlet: float):
    """
    Return a JIT-compiled function that imposes free-stream velocity on
    all four domain boundaries (inlet / top / bottom / outlet).
    """

    @jax.jit
    def apply_freestream_bc(state: FluidState) -> FluidState:
        u = state.velocity.u
        v = state.velocity.v

        # Left (inlet)
        u = u.at[:, :2].set(u_inlet)
        v = v.at[:, :2].set(v_inlet)

        # Right (outlet) — Neumann + free-stream floor
        u = u.at[:, -1].set(jnp.maximum(u[:, -2], u_inlet * 0.5))
        v = v.at[:, -1].set(v[:, -2])

        # Top — free-stream
        u = u.at[-1, :].set(u_inlet)
        v = v.at[-1, :].set(v_inlet)

        # Bottom — free-stream
        u = u.at[0, :].set(u_inlet)
        v = v.at[0, :].set(v_inlet)

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

    return apply_freestream_bc


# ─────────────────────────────────────────────────────────────────────────────
# RANS step factory
# ─────────────────────────────────────────────────────────────────────────────


def make_rans_step(grid: FluidGrid, u_inlet: float, v_inlet: float):
    """
    Build a JIT-compiled single RANS step (state, wall_dist) → state.

    wall_dist is an explicit argument (not closed over) so that JAX never
    needs to retrace when the wall geometry changes between optimization steps.
    """
    apply_freestream_bc = _make_apply_freestream_bc(u_inlet, v_inlet)

    @jax.jit
    def rans_step(state: FluidState, wall_dist: jnp.ndarray) -> FluidState:
        state = grid.step_sa_turbulence(state, wall_dist, num_diff_iters=4)
        nu_eff = grid.compute_effective_viscosity(state)
        state = grid.diffuse_velocity(state, num_iters=20, nu_eff_field=nu_eff)
        state = grid.advect_velocity(state)
        state = apply_freestream_bc(state)
        u, v = apply_ibm_continuous_forcing(
            state.velocity.u, state.velocity.v, state.solid_mask
        )
        state = state.__class__(
            density=state.density,
            velocity=state.velocity.with_values(u, v),
            pressure=state.pressure,
            solid_mask=state.solid_mask,
            sources=state.sources,
            nu_tilde=state.nu_tilde,
            time=state.time,
            step=state.step,
        )
        state = grid.solve_pressure(state, num_iters=60)
        state = grid.project_velocity(state)
        state = apply_freestream_bc(state)
        return state.__class__(
            density=state.density,
            velocity=state.velocity,
            pressure=state.pressure,
            solid_mask=state.solid_mask,
            sources=state.sources,
            nu_tilde=state.nu_tilde,
            time=state.time + grid.dt,
            step=state.step + 1,
        )

    return rans_step


# ─────────────────────────────────────────────────────────────────────────────
# Grid / step cache
# ─────────────────────────────────────────────────────────────────────────────


def _get_or_build_grid(
    u_inlet: float, v_inlet: float, nu_lam: float, visualise: bool = False
):
    """
    Return a persistent (FluidGrid, rans_step) pair, JIT-compiling on first
    call for each (u_inlet, v_inlet, nu_lam) combination.
    """
    key = (round(u_inlet, 8), round(v_inlet, 8), round(nu_lam, 12), visualise)
    if key not in _GRID_CACHE:
        print(
            "  [cache miss] Building FluidGrid and compiling rans_step ...",
            flush=True,
        )
        placeholder_mask = jnp.zeros((HEIGHT, WIDTH), dtype=jnp.float32)
        placeholder_wall_dist = jnp.ones((HEIGHT, WIDTH), dtype=jnp.float32) * (
            CELL_SIZE * 10
        )

        # Bounding box for visualization (control volume logic roughly covering airfoil)
        # Airfoil is spanning [AIRFOIL_X0, AIRFOIL_X0 + CHORD] and centered around AIRFOIL_Y0
        j1 = max(0, int((AIRFOIL_X0 - 0.05) / CELL_SIZE))
        j2 = min(WIDTH, int((AIRFOIL_X0 + CHORD + 0.05) / CELL_SIZE))
        i1 = max(0, int((AIRFOIL_Y0 - 0.15) / CELL_SIZE))
        i2 = min(HEIGHT, int((AIRFOIL_Y0 + 0.15) / CELL_SIZE))
        cv_rect = (i1, i2, j1, j2)

        grid = FluidGrid(
            height=HEIGHT,
            width=WIDTH,
            cell_size=CELL_SIZE,
            dt=DT,
            diffusion=0.0,
            viscosity=nu_lam,
            rho=RHO,
            boundary_type=0,
            use_sa_turbulence=False,  # avoid wall-dist scan in __init__
            visualise=visualise,
            show_velocity=True,
            show_cell_property="curl",
            show_cell_centered_velocity=True,
            cv_rect=cv_rect,
        )
        grid.solid_mask = placeholder_mask
        grid.use_sa_turbulence = True
        grid._wall_dist = placeholder_wall_dist

        rans_step = make_rans_step(grid, u_inlet, v_inlet)

        # Warm-up compile
        state0 = grid.create_initial_state()
        _ = rans_step(state0, placeholder_wall_dist)
        jax.block_until_ready(state0.velocity.u)
        print("  [cache] rans_step compiled and cached.", flush=True)

        _GRID_CACHE[key] = (grid, rans_step)

    return _GRID_CACHE[key]


# ─────────────────────────────────────────────────────────────────────────────
# Aerodynamic force integrator
# ─────────────────────────────────────────────────────────────────────────────


def compute_aerodynamic_forces(
    state: FluidState,
    solid_mask: jnp.ndarray,
    nu_eff: jnp.ndarray,
    rho: float,
    aoa_rad: float,
) -> tuple:
    """
    Compute pressure + viscous forces on the airfoil.
    Fully differentiable: operates only on jnp arrays.
    """
    u = state.velocity.u
    v = state.velocity.v
    p = state.pressure.values
    h = CELL_SIZE
    mu = rho * nu_eff

    fluid = 1.0 - solid_mask

    is_right_wall = fluid[:, :-1] * solid_mask[:, 1:]
    is_left_wall = fluid[:, 1:] * solid_mask[:, :-1]
    is_bottom_wall = fluid[:-1, :] * solid_mask[1:, :]
    is_top_wall = fluid[1:, :] * solid_mask[:-1, :]

    Fpx = jnp.sum(p[:, :-1] * is_right_wall) * h - jnp.sum(p[:, 1:] * is_left_wall) * h
    Fpy = jnp.sum(p[:-1, :] * is_bottom_wall) * h - jnp.sum(p[1:, :] * is_top_wall) * h

    u_cc = 0.5 * (u[:, :-1] + u[:, 1:])
    v_cc = 0.5 * (v[:-1, :] + v[1:, :])

    dudy = jnp.zeros_like(p).at[1:-1, :].set((u_cc[2:, :] - u_cc[:-2, :]) / (2.0 * h))
    dvdx = jnp.zeros_like(p).at[:, 1:-1].set((v_cc[:, 2:] - v_cc[:, :-2]) / (2.0 * h))
    dudx = jnp.zeros_like(p).at[:, 1:-1].set((u_cc[:, 2:] - u_cc[:, :-2]) / (2.0 * h))
    dvdy = jnp.zeros_like(p).at[1:-1, :].set((v_cc[2:, :] - v_cc[:-2, :]) / (2.0 * h))

    tau_xy = mu * (dudy + dvdx)
    tau_xx = 2.0 * mu * dudx
    tau_yy = 2.0 * mu * dvdy

    Fvx = (
        jnp.sum(tau_xx[:, :-1] * is_right_wall) * h
        - jnp.sum(tau_xx[:, 1:] * is_left_wall) * h
        + jnp.sum(tau_xy[:-1, :] * is_bottom_wall) * h
        - jnp.sum(tau_xy[1:, :] * is_top_wall) * h
    )
    Fvy = (
        jnp.sum(tau_xy[:, :-1] * is_right_wall) * h
        - jnp.sum(tau_xy[:, 1:] * is_left_wall) * h
        + jnp.sum(tau_yy[:-1, :] * is_bottom_wall) * h
        - jnp.sum(tau_yy[1:, :] * is_top_wall) * h
    )

    Fx = Fpx + Fvx
    Fy = Fpy + Fvy

    cos_a = jnp.cos(aoa_rad)
    sin_a = jnp.sin(aoa_rad)
    F_drag = Fx * cos_a + Fy * sin_a
    F_lift = -Fx * sin_a + Fy * cos_a

    return F_drag, F_lift


# ─────────────────────────────────────────────────────────────────────────────
# Differentiable RANS loss
# ─────────────────────────────────────────────────────────────────────────────


def make_loss_fn(
    grid: FluidGrid,
    rans_step,
    subcells: list[dict],
    wall_dist_jnp: jnp.ndarray,
    solid_mask_open_hard: jnp.ndarray,
    aoa_deg: float,
    nu_lam: float,
):
    """
    Build and return a pure-JAX loss function:

        loss_fn(params: jnp.ndarray) -> (scalar_loss, (Cl, Cd))

    Gradient strategy:
      - RANS time-stepping uses jax.lax.fori_loop, and we stop_gradient
        the resulting state.
      - We compute aerodynamic forces ONLY on the final steady-state
        flow field, using the differentiable soft_mask.
    """
    aoa_rad = jnp.float32(math.radians(aoa_deg))
    q_inf = 0.5 * RHO * U_INF**2
    ref_area = CHORD
    u_inlet = float(U_INF * math.cos(math.radians(aoa_deg)))
    v_inlet = float(U_INF * math.sin(math.radians(aoa_deg)))
    n_upper = N_ORDER + 1

    # Pre-fetch the grid's initial state template; solid_mask will be
    # overridden dynamically inside the loss.
    grid.solid_mask = solid_mask_open_hard  # set for create_initial_state

    def loss_fn(params: jnp.ndarray):
        weights_upper = params[:n_upper]
        weights_lower = params[n_upper:]

        # ── 1. Geometry: differentiable soft-mask ─────────────────────────────
        soft_mask = build_solid_mask_jax(weights_upper, weights_lower, subcells)

        # Open the right boundary for outflow (stop_gradient on this operation
        # since the slice modification doesn't carry gradient information we need)
        soft_mask_open = soft_mask.at[1:-1, -1].set(0.0)

        # ── 2. Geometric feasibility loss (fully JAX) ─────────────────────────
        x_, y_upper_, y_lower_ = generate_cst_coords(
            weights_upper, weights_lower, num_points=200
        )
        thickness = thickness_at_x(y_upper_, y_lower_)
        geo_loss = crossover_validity_loss(
            y_upper_, y_lower_
        ) + thickness_constraint_loss(
            thickness, MIN_THICKNESS_RATIO, MAX_THICKNESS_RATIO
        )

        # ── 3. Initial fluid state ─────────────────────────────────────────────
        # We use the hard (thresholded) mask for IBM forcing and nu_tilde
        # initialisation; gradients don't need to flow through these.
        hard_mask = jax.lax.stop_gradient(
            jnp.where(soft_mask_open >= 0.5, jnp.float32(1.0), jnp.float32(0.0))
        )

        u_init = jnp.ones((HEIGHT, WIDTH + 1), dtype=jnp.float32) * u_inlet
        v_init = jnp.ones((HEIGHT + 1, WIDTH), dtype=jnp.float32) * v_inlet
        u_init, v_init = apply_ibm_continuous_forcing(u_init, v_init, hard_mask)

        # Create a base state from grid (uses grid.solid_mask = hard mask)
        base_state = grid.create_initial_state()

        # Seed SA modified eddy viscosity: 0 inside solid, 5ν in fluid.
        nu_init = jnp.where(
            hard_mask > 0.5, jnp.float32(0.0), jnp.float32(5.0 * nu_lam)
        )
        nu_tilde_init = base_state.nu_tilde.with_values(nu_init)

        # Override with free-stream velocity and the nondifferentiable hard mask
        state = base_state.__class__(
            density=base_state.density,
            velocity=base_state.velocity.with_values(u_init, v_init),
            pressure=base_state.pressure,
            solid_mask=hard_mask,  # ← simulation uses hard mask
            sources=base_state.sources,
            nu_tilde=nu_tilde_init,
            time=jnp.float32(0.0),
            step=jnp.int32(0),
        )

        # ── 4. RANS time integration ──────────────────────────────────────────
        wd = wall_dist_jnp  # already stop_grad from compute_wall_dist_np

        def step_body(i, s):
            return rans_step(s, wd)

        final_state = jax.lax.fori_loop(0, N_SIM_STEPS, step_body, state)

        # Stop gradient on final state to prevent backpropagation through time
        final_state = jax.lax.stop_gradient(final_state)

        # ── 5. Aerodynamic loss on final state ────────────────────────────────
        nu_eff = grid.compute_effective_viscosity(final_state)
        nu_eff = jax.lax.stop_gradient(nu_eff)

        # Calculate forces using differentiable soft mask and constants state fields
        F_drag, F_lift = compute_aerodynamic_forces(
            final_state, soft_mask_open, nu_eff, RHO, aoa_rad
        )

        Cl = F_lift / (q_inf * ref_area)
        Cd = F_drag / (q_inf * ref_area)

        # ── 6. Composite loss ─────────────────────────────────────────────────
        # Objective: Maximize aerodynamic efficiency (Lift/Drag ratio).
        # To do this, we minimize the inverse L/D ratio: (Cd / Cl).
        #
        # Safety: We must clamp Cl in the denominator. If Cl is negative,
        # minimizing (Cd / Cl) would become more negative by INCREASING Cd!
        # Clamping prevents this "ratio trap" and division by zero.
        min_cl = jnp.float32(0.1)
        safe_cl = jnp.maximum(Cl, min_cl)

        # Base objective: Inverse L/D ratio
        inverse_ld = jnp.abs(Cd) / safe_cl

        # Strong penalty to push Cl into positive domain if it starts negative/low
        low_lift_penalty = jnp.maximum(min_cl - Cl, jnp.float32(0.0))

        # Weighting: Use W_DRAG for the ratio, and strongly weight the lift penalty
        # to ensure the optimizer escapes the clamped region quickly.
        aero_loss = W_DRAG * inverse_ld + W_LIFT * low_lift_penalty * 10.0

        total_loss = aero_loss + W_GEO * geo_loss

        return total_loss, (Cl, Cd)

    return loss_fn


# ─────────────────────────────────────────────────────────────────────────────
# .dat file writer — Selig format (compatible with xfoil_validation.py)
# ─────────────────────────────────────────────────────────────────────────────


def write_dat_file(
    weights_upper: np.ndarray,
    weights_lower: np.ndarray,
    filepath: str,
    label: str = "Optimized CST Airfoil",
    num_points: int = 200,
) -> None:
    wu = jnp.array(weights_upper)
    wl = jnp.array(weights_lower)

    x_norm, y_upper, y_lower = generate_cst_coords(wu, wl, num_points=num_points)
    x_np = np.array(x_norm)
    yu = np.array(y_upper)
    yl = np.array(y_lower)

    x_upper_out = x_np[::-1]
    y_upper_out = yu[::-1]
    x_lower_out = x_np
    y_lower_out = yl

    with open(filepath, "w") as f:
        f.write(f"{label}\n")
        for xi, yi in zip(x_upper_out, y_upper_out):
            f.write(f"  {xi:.6f}  {yi:.6f}\n")
        for xi, yi in zip(x_lower_out, y_lower_out):
            f.write(f"  {xi:.6f}  {yi:.6f}\n")

    print(f"  → Saved: {filepath}  ({2*num_points} coordinate pairs)")


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────


def plot_optimization_history(losses, cls, cds, filepath):
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)

    axes[0].plot(losses, "o-", color="#E74C3C", linewidth=2, markersize=5)
    axes[0].set_ylabel("Loss", fontsize=12)
    axes[0].set_title(
        "RANS Airfoil Optimization History", fontsize=14, fontweight="bold"
    )
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(cls, "s-", color="#2ECC71", linewidth=2, markersize=5, label="Cl")
    axes[1].set_ylabel("Lift coefficient  Cl", fontsize=12)
    axes[1].axhline(0.3, linestyle="--", color="#2ECC71", alpha=0.5, label="target Cl")
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(cds, "^-", color="#3498DB", linewidth=2, markersize=5)
    axes[2].set_ylabel("Drag coefficient  Cd\n", fontsize=12)
    axes[2].set_xlabel("Optimization iteration", fontsize=12)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    print(f"  → Saved: {filepath}")
    plt.close(fig)


def plot_shape_comparison(
    initial_upper,
    initial_lower,
    final_upper,
    final_lower,
    filepath,
):
    wu_i = jnp.array(initial_upper)
    wl_i = jnp.array(initial_lower)
    wu_f = jnp.array(final_upper)
    wl_f = jnp.array(final_lower)

    x_i, yu_i, yl_i = generate_cst_coords(wu_i, wl_i, num_points=300)
    x_f, yu_f, yl_f = generate_cst_coords(wu_f, wl_f, num_points=300)
    x_i, yu_i, yl_i = map(np.array, (x_i, yu_i, yl_i))
    x_f, yu_f, yl_f = map(np.array, (x_f, yu_f, yl_f))

    fig, ax = plt.subplots(figsize=(11, 5))

    ax.fill_between(x_i, yl_i, yu_i, alpha=0.25, color="#E74C3C", label="Initial")
    ax.plot(x_i, yu_i, "--", color="#E74C3C", linewidth=1.8)
    ax.plot(x_i, yl_i, "--", color="#E74C3C", linewidth=1.8)

    ax.fill_between(
        x_f, yl_f, yu_f, alpha=0.25, color="#2ECC71", label="Optimized (RANS)"
    )
    ax.plot(x_f, yu_f, "-", color="#2ECC71", linewidth=2.2)
    ax.plot(x_f, yl_f, "-", color="#2ECC71", linewidth=2.2)

    ax.set_xlabel("x/c", fontsize=13)
    ax.set_ylabel("y/c", fontsize=13)
    ax.set_title("Airfoil Shape Evolution — RANS-driven CST Optimization", fontsize=14)
    ax.legend(fontsize=12)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    print(f"  → Saved: {filepath}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main optimization loop
# ─────────────────────────────────────────────────────────────────────────────


def main():
    args = parse_args()

    # ── Physics: Re → viscosity ───────────────────────────────────────────────
    RE = _VALID_RE[args.re]
    NU_LAM = (U_INF * CHORD) / RE  # ν = 1/Re  (since U_inf = c = 1)
    AoA_DEG = args.aoa
    aoa_rad = math.radians(AoA_DEG)
    u_inlet = U_INF * math.cos(aoa_rad)
    v_inlet = U_INF * math.sin(aoa_rad)

    print()
    print("═" * 70)
    print("  RANS Airfoil Shape Optimization — CST + Spalart-Allmaras")
    print(
        f"  Grid: {HEIGHT}×{WIDTH}  cell={CELL_SIZE*1000:.1f} mm  "
        f"Re={RE:.2e}  ν={NU_LAM:.3e} m²/s  AoA={AoA_DEG:.1f}°"
    )
    print(
        f"  U_inf={U_INF} m/s  chord={CHORD} m  "
        f"(u_inlet={u_inlet:.4f}, v_inlet={v_inlet:.4f})"
    )
    print(
        f"  Iterations: {NUM_ITERATIONS}  |  CST order: {N_ORDER}  "
        f"({N_ORDER+1} weights/surface)"
    )
    print("  Gradient: jax.grad (analytic)  |  Boundaries: free-stream")
    print("═" * 70)

    # ── Initial CST weights (NACA 0012 profile) ───────────────────────────────
    n_weights = N_ORDER + 1
    initial_upper = jnp.array([0.1728, 0.1650, 0.1400, 0.1550, 0.1300, 0.1150])
    initial_lower = -initial_upper  # perfect symmetry

    assert (
        len(initial_upper) == n_weights and len(initial_lower) == n_weights
    ), "Weight arrays must have length N_ORDER + 1"

    params = jnp.concatenate([initial_upper, initial_lower])
    n_upper = n_weights

    # ── Pre-compute subcell geometry (depends only on AoA, grid constants) ────
    print("\n  Pre-computing subcell geometry ...")
    subcells = _get_subcell_geometry(AoA_DEG)

    # ── Build initial hard mask (for wall-dist + IBM; not differentiated) ─────
    print("  Building initial solid mask ...")
    init_soft_mask = build_solid_mask_jax(initial_upper, initial_lower, subcells)
    hard_mask_open = jnp.array(
        np.where(np.array(init_soft_mask) >= 0.5, 1.0, 0.0), dtype=jnp.float32
    )
    hard_mask_open = hard_mask_open.at[1:-1, -1].set(0.0)

    print("  Computing wall distances ...")
    wall_dist_jnp = compute_wall_dist_np(init_soft_mask)

    # ── Retrieve / build persistent grid + compiled step ─────────────────────
    grid, rans_step = _get_or_build_grid(
        u_inlet, v_inlet, NU_LAM, visualise=args.visualise
    )
    grid.solid_mask = hard_mask_open  # seed for create_initial_state

    # ── Build differentiable loss function ────────────────────────────────────
    print("  Building differentiable loss function (this triggers JIT) ...")
    loss_fn = make_loss_fn(
        grid, rans_step, subcells, wall_dist_jnp, hard_mask_open, AoA_DEG, NU_LAM
    )

    # value_and_grad returns (loss, aux), grad simultaneously
    # has_aux=True because loss_fn returns (scalar, (Cl, Cd))
    val_and_grad = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))

    # ── Adam optimizer state ─────────────────────────────────────────────────
    m = jnp.zeros_like(params)
    v_m = jnp.zeros_like(params)
    beta1, beta2, eps_adam = 0.9, 0.999, 1e-8

    loss_history, cl_history, cd_history = [], [], []

    # ── Evaluate initial design (also triggers compilation of val_and_grad) ───
    print("\n  Evaluating initial design (first call compiles the graph) ...")
    t0 = time.time()
    (loss0, (Cl0, Cd0)), grad0 = val_and_grad(params)
    jax.block_until_ready(grad0)
    compile_time = time.time() - t0
    print(
        f"  Initial → Loss={float(loss0):.5f}  Cl={float(Cl0):.4f}  "
        f"Cd={float(Cd0):.5f}  (compile+eval: {compile_time:.1f}s)"
    )

    if args.visualise:
        print("\n  Visualisation mode enabled. Starting simulation loop.")
        print("  Close the window or press ESC to exit.")
        import pygame

        clock = pygame.time.Clock()
        running = True

        # Build initial state directly
        base_state = grid.create_initial_state()
        nu_init = jnp.where(hard_mask_open > 0.5, 0.0, 5.0 * NU_LAM)
        nu_tilde_init = base_state.nu_tilde.with_values(nu_init)

        u_init = jnp.ones((HEIGHT, WIDTH + 1), dtype=jnp.float32) * u_inlet
        v_init = jnp.ones((HEIGHT + 1, WIDTH), dtype=jnp.float32) * v_inlet
        u_init, v_init = apply_ibm_continuous_forcing(u_init, v_init, hard_mask_open)

        state = base_state.__class__(
            density=base_state.density,
            velocity=base_state.velocity.with_values(u_init, v_init),
            pressure=base_state.pressure,
            solid_mask=hard_mask_open,
            sources=base_state.sources,
            nu_tilde=nu_tilde_init,
            time=jnp.float32(0.0),
            step=jnp.int32(0),
        )

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False

            state = rans_step(state, wall_dist_jnp)
            grid.draw_state(state)
            clock.tick(120)

        pygame.quit()
        return

    loss_history.append(float(loss0))
    cl_history.append(float(Cl0))
    cd_history.append(float(Cd0))

    header = (
        f"  {'Iter':>5}  {'Loss':>10}  {'Cl':>9}  {'Cd':>9}  "
        f"{'Cl/Cd':>9}  {'|grad|':>9}  {'Time(s)':>8}"
    )
    print(f"\n  Starting optimization ...\n{header}")
    print("  " + "-" * 73)

    t_start = time.time()

    for it in range(1, NUM_ITERATIONS + 1):
        t_iter = time.time()

        # ── Analytic gradient via jax.grad ────────────────────────────────────
        (loss_val, (Cl, Cd)), grad = val_and_grad(params)
        jax.block_until_ready(grad)

        loss_history.append(float(loss_val))
        cl_history.append(float(Cl))
        cd_history.append(float(Cd))

        # ── Adam update ───────────────────────────────────────────────────────
        m = beta1 * m + (1.0 - beta1) * grad
        v_m = beta2 * v_m + (1.0 - beta2) * grad**2
        m_hat = m / (1.0 - beta1**it)
        v_hat = v_m / (1.0 - beta2**it)
        params = params - LEARNING_RATE * m_hat / (jnp.sqrt(v_hat) + eps_adam)

        cl_cd_ratio = float(Cl) / float(Cd) if abs(float(Cd)) > 1e-12 else 0.0
        grad_norm = float(jnp.linalg.norm(grad))
        elapsed = time.time() - t_iter

        print(
            f"  {it:>5}  {float(loss_val):>10.5f}  {float(Cl):>9.4f}  {float(Cd):>9.5f}"
            f"  {cl_cd_ratio:>9.3f}  {grad_norm:>9.4f}  {elapsed:>8.1f}s"
        )

    total_time = time.time() - t_start
    print("  " + "-" * 73)
    print(f"\n  Total wall-clock time: {total_time/60:.1f} min")

    final_upper = np.array(params[:n_upper])
    final_lower = np.array(params[n_upper:])

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n  {'Metric':<20} {'Initial':>10}  {'Final':>10}")
    print(f"  {'─'*42}")
    print(f"  {'Loss':<20} {loss_history[0]:>10.5f}  {loss_history[-1]:>10.5f}")
    print(f"  {'Cl':<20} {cl_history[0]:>10.4f}  {cl_history[-1]:>10.4f}")
    print(f"  {'Cd':<20} {cd_history[0]:>10.5f}  {cd_history[-1]:>10.5f}")
    cl_cd_0 = cl_history[0] / cd_history[0] if abs(cd_history[0]) > 1e-12 else 0
    cl_cd_f = cl_history[-1] / cd_history[-1] if abs(cd_history[-1]) > 1e-12 else 0
    print(f"  {'Cl/Cd':<20} {cl_cd_0:>10.3f}  {cl_cd_f:>10.3f}")

    _, yu_i, yl_i = generate_cst_coords(initial_upper, initial_lower)
    _, yu_f, yl_f = generate_cst_coords(params[:n_upper], params[n_upper:])
    max_t_0 = float(max_thickness(yu_i, yl_i))
    max_t_f = float(max_thickness(yu_f, yl_f))
    print(f"  {'Max Thickness':<20} {max_t_0:>10.4f}  {max_t_f:>10.4f}")
    drag_red = (
        (1.0 - loss_history[-1] / loss_history[0]) * 100 if loss_history[0] != 0 else 0
    )
    print(f"\n  Loss reduction : {drag_red:.1f}%")

    # ── Export .dat file ──────────────────────────────────────────────────────
    print("\n  Writing XFoil-compatible .dat file ...")
    write_dat_file(
        final_upper,
        final_lower,
        filepath=OUTPUT_DAT,
        label=(
            f"RANS-Optimized CST Airfoil  "
            f"AoA={AoA_DEG:.1f}deg  Re={RE:.2e}  "
            f"Cl={cl_history[-1]:.4f}  Cd={cd_history[-1]:.5f}"
        ),
    )
    print("\n  To validate with XFoil, edit xfoil_validation.py:")
    print(f'    input_file  = "{OUTPUT_DAT}"')
    print('    output_polar = "optimized_airfoil_polar.txt"')

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\n  Generating plots ...")
    plot_optimization_history(loss_history, cl_history, cd_history, OUTPUT_HIST)
    plot_shape_comparison(
        np.array(initial_upper),
        np.array(initial_lower),
        final_upper,
        final_lower,
        OUTPUT_SHAPE,
    )

    print("\n  Done.")
    print("═" * 70)


if __name__ == "__main__":
    main()
