"""
RANS Airfoil Shape Optimization — CST + Spalart-Allmaras
=========================================================
Combines gradient-based CST optimization (optimize_airfoil.py) with the
full RANS simulation setup (naca0012_rans_benchmark.py) to find an airfoil
that maximises lift-to-drag ratio under turbulent flow conditions.

The final optimized shape is exported as a Selig-format .dat file that can be
fed directly into xfoil_validation.py for independent XFoil verification.

Setup:
  - CST parametrization : N_ORDER = 5 (6 weights per surface)
  - Grid                : HEIGHT × WIDTH cells, cell_size = 0.005 m
  - RANS model          : Spalart-Allmaras one-equation (SA-1994)
  - Re                  : 2 × 10⁶  (U∞=1 m/s, ν=5×10⁻⁷ m²/s)
  - Angle of Attack     : configurable via AoA_DEG
  - Objective           : minimise drag (with lift & geometry constraints)

Output:
  - optimized_airfoil.dat   — Selig-format coordinates for XFoil
  - rans_opt_history.png    — loss / Cl / Cd versus iteration
  - rans_opt_shapes.png     — initial vs final airfoil overlaid

Run:
  cd /Users/musab/FYP/TurboDiff
  .venv/bin/python examples/optimize_airfoil_rans.py

Validate with XFoil (edit xfoil_validation.py to point at the output):
  input_file = "optimized_airfoil.dat"
"""

import math
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.ndimage import distance_transform_edt
import matplotlib

matplotlib.use("Agg")  # headless — no display needed
import matplotlib.pyplot as plt

from turbodiff.core.fluid_grid_jax import FluidGrid, FluidState
from turbodiff.core.utils import apply_ibm_continuous_forcing
from turbodiff.core.airfoil import (
    generate_cst_coords,
    thickness_at_x,
)
from turbodiff.core.loss_functions import (
    thickness_constraint_loss,
    crossover_validity_loss,
)

# ─────────────────────────────────────────────────────────────────────────────
# Persistent JIT-compiled infrastructure
# ─────────────────────────────────────────────────────────────────────────────
# Keyed by (u_inlet_rounded, v_inlet_rounded).  Stores (FluidGrid, rans_step)
# so that every call to evaluate_rans reuses the same compiled objects.
_GRID_CACHE: dict = {}


# ─────────────────────────────────────────────────────────────────────────────
# Hyper-parameters
# ─────────────────────────────────────────────────────────────────────────────

# Grid / domain
HEIGHT = 64  # cells in y  (increase for finer resolution)
WIDTH = 192  # cells in x
CELL_SIZE = 0.005  # m  → domain 0.32 m × 0.96 m

# RANS physics
DT = 0.0005  # time-step (s) — CFL ≈ 0.1 at U=1, h=5 mm
NU_LAM = 5e-7  # laminar kinematic viscosity → Re = U∞/ν = 2×10⁶
RHO = 1.0  # density (kg/m³)
U_INF = 1.0  # free-stream speed (m/s)

# Airfoil placement (leading-edge position inside the domain, in metres)
CHORD = 0.3  # m  — scaled so it fits the smaller grid
AIRFOIL_X0 = 0.2  # m  — x of leading edge
AIRFOIL_Y0 = HEIGHT * CELL_SIZE / 2.0  # m  — vertical midpoint

# Angle of attack for optimization
AoA_DEG = 4.0  # degrees

# Simulation budget per gradient evaluation
# Each FD iteration runs 2×12 = 24 evaluate_rans calls.
# Total RANS steps per Adam step = 24 × N_SIM_STEPS.
# At 64×192 with 100 steps + 60 pressure iters ≈ 1–2s/eval → ~30–50s/Adam step.
N_SIM_STEPS = 100  # steps run per loss evaluation (increase for accuracy)
N_WARMUP = 20  # steps excluded from force averaging (transient flush)
N_AVG_STEPS = 10  # number of steps over which forces are time-averaged

# Optimization
NUM_ITERATIONS = 10
LEARNING_RATE = 0.005
N_ORDER = 5  # CST polynomial order — 6 weights per surface

# Geometry constraints
MIN_THICKNESS_RATIO = 0.06  # t/c must exceed this somewhere
MIN_THICKNESS_X = 0.25  # at this chord location

# Loss weights
W_DRAG = 1.0  # drag minimization
W_LIFT = 0.5  # soft lift-maximization bonus (penalty for low lift)
W_GEO = 10.0  # geometric feasibility

# Output files  — will be written next to this script
OUTPUT_DAT = "optimized_airfoil.dat"
OUTPUT_HIST = "rans_opt_history.png"
OUTPUT_SHAPE = "rans_opt_shapes.png"


# ─────────────────────────────────────────────────────────────────────────────
# Solid-mask builder (analytically exact, super-sampled, from RANS benchmark)
# ─────────────────────────────────────────────────────────────────────────────


def build_solid_mask_cst(
    weights_upper: np.ndarray,
    weights_lower: np.ndarray,
    height: int,
    width: int,
    cell_size: float,
    chord: float,
    x0: float,
    y0: float,
    aoa_deg: float,
    n_sub: int = 3,
) -> np.ndarray:
    """
    Build a super-sampled floating-point solid mask for a CST airfoil
    at the given angle of attack.

    Args:
        weights_upper / weights_lower : CST Bernstein coefficients (numpy arrays)
        height, width  : grid dimensions
        cell_size      : metres per cell
        chord          : chord length (m)
        x0, y0         : leading-edge position in the domain (m)
        aoa_deg        : angle of attack (degrees)
        n_sub          : sub-sampling factor for anti-aliasing

    Returns:
        solid_mask : JAX float32 array (height, width), values in [0, 1]
    """
    aoa_rad = math.radians(aoa_deg)
    cos_a, sin_a = math.cos(aoa_rad), math.sin(aoa_rad)

    # Shape functions (exact mathematical Bernstein coefficients)
    n_order = len(weights_upper) - 1
    i_vals = np.arange(n_order + 1)
    from scipy.special import comb as sp_comb

    binom = np.array([sp_comb(n_order, i, exact=True) for i in i_vals])

    # Cell-centre grid indices
    jj, ii = np.meshgrid(np.arange(width), np.arange(height))  # (H, W)

    solid_accum = np.zeros((height, width), dtype=np.float32)

    for si in range(n_sub):
        for sj in range(n_sub):
            cx = (jj + (sj + 0.5) / n_sub) * cell_size  # physical x
            cy = (ii + (si + 0.5) / n_sub) * cell_size  # physical y

            # Translate and rotate into body frame
            cx_rel = cx - x0
            cy_rel = cy - y0
            cx_body = cx_rel * cos_a - cy_rel * sin_a  # along chord
            cy_body = cx_rel * sin_a + cy_rel * cos_a  # perpendicular

            xi_grid = cx_body / chord  # normalised chord pos

            xi_c = np.clip(xi_grid, 0.0, 1.0)
            C_g = np.sqrt(np.maximum(xi_c, 0.0)) * (1.0 - xi_c)

            xi_c_col = xi_c[..., None]  # (H, W, 1)
            B_g = (
                binom * xi_c_col**i_vals * (1.0 - xi_c_col) ** (n_order - i_vals)
            )  # (H, W, N+1)

            y_u_g = C_g * (B_g @ weights_upper)  # (H, W)
            y_l_g = C_g * (B_g @ weights_lower)  # (H, W)

            # A point is inside if chord position in [0,1] and |y_body| ≤ surface
            inside = (
                (xi_grid >= 0.0)
                & (xi_grid <= 1.0)
                & (cy_body <= y_u_g * chord)
                & (cy_body >= y_l_g * chord)
            )
            solid_accum += inside.astype(np.float32)

    return jnp.array(solid_accum / (n_sub * n_sub))


# ─────────────────────────────────────────────────────────────────────────────
# Boundary condition helpers (taken verbatim from RANS benchmark)
# ─────────────────────────────────────────────────────────────────────────────


@jax.jit
def inject_uniform_inlet(
    state: FluidState, u_inlet: float, v_inlet: float
) -> FluidState:
    u = state.velocity.u.at[:, 0:2].set(u_inlet)
    v = state.velocity.v.at[:, 0:2].set(v_inlet)
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
def apply_slip_top_bottom_bc(state: FluidState) -> FluidState:
    u = state.velocity.u
    v = state.velocity.v
    v = v.at[0, :].set(0.0)
    v = v.at[-1, :].set(0.0)
    u = u.at[0, :].set(u[1, :])
    u = u.at[-1, :].set(u[-2, :])
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
def apply_outflow_bc(state: FluidState) -> FluidState:
    u = state.velocity.u.at[:, -1].set(state.velocity.u[:, -2])
    v = state.velocity.v.at[:, -1].set(state.velocity.v[:, -2])
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


# ─────────────────────────────────────────────────────────────────────────────
# RANS step factory
# ─────────────────────────────────────────────────────────────────────────────


def make_rans_step(grid: FluidGrid, u_inlet: float, v_inlet: float):
    """
    Build a JIT-compiled RANS step function.

    wall_dist is an EXPLICIT argument (not closed over from grid._wall_dist).
    This is critical: closed-over JAX arrays are embedded as XLA constants, so
    any change to grid._wall_dist would require a full retrace.  By making it an
    argument, JAX sees the same abstract shape every call and reuses the
    compiled executable regardless of the concrete wall_dist values.
    """

    @jax.jit
    def rans_step(state: FluidState, wall_dist: jnp.ndarray) -> FluidState:
        state = grid.step_sa_turbulence(state, wall_dist, num_diff_iters=4)
        nu_eff = grid.compute_effective_viscosity(state)
        state = grid.diffuse_velocity(state, num_iters=20, nu_eff_field=nu_eff)
        state = grid.advect_velocity(state)
        state = inject_uniform_inlet(state, u_inlet, v_inlet)
        state = apply_slip_top_bottom_bc(state)
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
        state = inject_uniform_inlet(state, u_inlet, v_inlet)
        state = apply_slip_top_bottom_bc(state)
        state = apply_outflow_bc(state)
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
# Force integration (from RANS benchmark — pressure + viscous)
# ─────────────────────────────────────────────────────────────────────────────


@jax.jit
def compute_aerodynamic_forces(
    state: FluidState,
    solid_mask: jnp.ndarray,
    cell_size: float,
    nu_eff: jnp.ndarray,
    rho: float,
    aoa_rad: float,
) -> tuple:
    u = state.velocity.u
    v = state.velocity.v
    p = state.pressure.values
    h = cell_size
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
# RANS evaluator: run simulation, return (C_L, C_D) for given CST weights
#
#  The FluidGrid and rans_step are created ONCE per AoA (cached in _GRID_CACHE)
#  and reused across every call.  Only the solid_mask and wall_dist change
#  between calls; both are passed as JAX array ARGUMENTS (not captured in
#  closures / baked as XLA constants), so JAX sees the same abstract shapes
#  and hits the compiled XLA cache every time.
# ─────────────────────────────────────────────────────────────────────────────


def _get_or_build_grid(u_inlet: float, v_inlet: float):
    """
    Return a persistent (FluidGrid, rans_step) pair for the given inlet
    conditions, building and JIT-compiling them on first call.

    Keyed on rounded (u_inlet, v_inlet) so a fixed AoA never rebuilds.
    """
    key = (round(u_inlet, 8), round(v_inlet, 8))
    if key not in _GRID_CACHE:
        print(
            "  [cache miss] Building persistent FluidGrid and compiling rans_step ...",
            flush=True,
        )
        # Placeholder solid mask — will be overwritten before every simulation.
        # We need it to have the right shape for create_initial_state.
        placeholder_mask = jnp.zeros((HEIGHT, WIDTH), dtype=jnp.float32)
        placeholder_wall_dist = jnp.ones((HEIGHT, WIDTH), dtype=jnp.float32) * (
            CELL_SIZE * 10
        )

        grid = FluidGrid(
            height=HEIGHT,
            width=WIDTH,
            cell_size=CELL_SIZE,
            dt=DT,
            diffusion=0.0,
            viscosity=NU_LAM,
            rho=RHO,
            boundary_type=0,
            use_sa_turbulence=False,  # set manually so __init__ skips wall-dist scan
            visualise=False,
        )
        grid.solid_mask = placeholder_mask
        grid.use_sa_turbulence = True
        grid._wall_dist = placeholder_wall_dist

        rans_step = make_rans_step(grid, u_inlet, v_inlet)

        # Warm-up compile: run one step with placeholder state so the XLA
        # executable is ready before the optimization loop starts.
        state0 = grid.create_initial_state()
        _ = rans_step(state0, placeholder_wall_dist)
        jax.block_until_ready(state0.velocity.u)
        print("  [cache] rans_step compiled and cached.", flush=True)

        _GRID_CACHE[key] = (grid, rans_step)

    return _GRID_CACHE[key]


def evaluate_rans(
    weights_upper_np: np.ndarray,
    weights_lower_np: np.ndarray,
    aoa_deg: float,
    verbose: bool = False,
) -> tuple:
    """
    Run a RANS simulation for the CST airfoil defined by weights at aoa_deg.

    Reuses the persistent FluidGrid / compiled rans_step — no recompilation.

    Returns:
        (C_L, C_D)  — dimensionless lift and drag coefficients (Python floats)
    """
    aoa_rad = math.radians(aoa_deg)
    u_inlet = U_INF * math.cos(aoa_rad)
    v_inlet = U_INF * math.sin(aoa_rad)
    q_inf = 0.5 * RHO * U_INF**2
    ref_area = CHORD

    # ── Geometry: solid mask + wall distances ─────────────────────────────────
    solid_mask_jax = build_solid_mask_cst(
        weights_upper_np,
        weights_lower_np,
        HEIGHT,
        WIDTH,
        CELL_SIZE,
        CHORD,
        AIRFOIL_X0,
        AIRFOIL_Y0,
        aoa_deg,
    )  # returns jnp array from build_solid_mask_cst

    solid_np = np.array(solid_mask_jax)
    solid_np_open = solid_np.copy()
    solid_np_open[1:-1, -1] = 0.0  # open right boundary for outflow

    wall_dist_np = distance_transform_edt(solid_np_open < 0.5) * CELL_SIZE
    wall_dist_np = np.maximum(wall_dist_np.astype(np.float32), 1e-10)
    wall_dist_jnp = jnp.array(wall_dist_np)
    solid_mask_open = jnp.array(solid_np_open, dtype=jnp.float32)

    # ── Retrieve persistent compiled infrastructure ───────────────────────────
    grid, rans_step = _get_or_build_grid(u_inlet, v_inlet)

    # Update the grid's solid_mask so create_initial_state uses the right mask
    # (nu_tilde is seeded to 0 inside solid cells via state.solid_mask).
    # The RANS step itself receives solid_mask through state, not from grid,
    # so changing grid.solid_mask is sufficient for correct initialization.
    grid.solid_mask = solid_mask_open

    # ── Initial state ─────────────────────────────────────────────────────────
    state = grid.create_initial_state()

    u_init = jnp.ones((HEIGHT, WIDTH + 1)) * u_inlet
    v_init = jnp.ones((HEIGHT + 1, WIDTH)) * v_inlet
    u_init, v_init = apply_ibm_continuous_forcing(u_init, v_init, solid_mask_open)
    state = state.__class__(
        density=state.density,
        velocity=state.velocity.with_values(u_init, v_init),
        pressure=state.pressure,
        solid_mask=solid_mask_open,  # use the opened mask in state
        sources=state.sources,
        nu_tilde=state.nu_tilde,
        time=0.0,
        step=0,
    )

    # ── Simulation loop (wall_dist passed as explicit arg — no retrace) ───────
    Cl_sum, Cd_sum, n_avg = 0.0, 0.0, 0

    for step in range(1, N_SIM_STEPS + 1):
        state = rans_step(state, wall_dist_jnp)

        # Time-average the last N_AVG_STEPS steps
        if step > (N_SIM_STEPS - N_AVG_STEPS):
            nu_eff = grid.compute_effective_viscosity(state)
            F_drag, F_lift = compute_aerodynamic_forces(
                state, solid_mask_open, CELL_SIZE, nu_eff, RHO, aoa_rad
            )
            Cl_sum += float(F_lift) / (q_inf * ref_area)
            Cd_sum += float(F_drag) / (q_inf * ref_area)
            n_avg += 1

    Cl = Cl_sum / max(n_avg, 1)
    Cd = Cd_sum / max(n_avg, 1)

    return Cl, Cd


# ─────────────────────────────────────────────────────────────────────────────
# Composite scalar loss (used to compute gradients via finite differences)
#
#  We use finite-difference (FD) gradients rather than jax.grad because the
#  full RANS loop involves Python-level control flow and numpy grid construction
#  that are not JAX-differentiable end-to-end.  This is honest — it avoids
#  silently wrong gradients from half-traced loops.
#
#  If you later move mask construction inside a fully JIT-traced function, you
#  can switch back to jax.grad for free.
# ─────────────────────────────────────────────────────────────────────────────


def compute_loss(
    weights_upper_np: np.ndarray,
    weights_lower_np: np.ndarray,
    aoa_deg: float,
) -> tuple:
    """
    Evaluate the scalar optimization loss and auxiliary aerodynamic information.

    Loss = W_DRAG * |Cd|  +  W_LIFT * max(0, target_Cl - Cl)  +  W_GEO * geo_penalty

    Returns:
        (loss, Cl, Cd)
    """
    wu = jnp.array(weights_upper_np)
    wl = jnp.array(weights_lower_np)

    # Geometric feasibility term (differentiable via JAX)
    x_, y_upper_, y_lower_ = generate_cst_coords(wu, wl, num_points=200)
    thickness = thickness_at_x(y_upper_, y_lower_)
    geo_loss = crossover_validity_loss(y_upper_, y_lower_) + thickness_constraint_loss(
        thickness, MIN_THICKNESS_RATIO, MIN_THICKNESS_X
    )

    # Aerodynamic evaluation
    Cl, Cd = evaluate_rans(weights_upper_np, weights_lower_np, aoa_deg)

    # Soft lift target: want Cl > ~0.3 at AoA=4°
    target_Cl = 0.3
    lift_penalty = max(0.0, target_Cl - Cl)

    loss = W_DRAG * abs(Cd) + W_LIFT * lift_penalty + W_GEO * float(geo_loss)

    return loss, Cl, Cd


def finite_difference_gradient(
    weights: np.ndarray,
    n_upper: int,
    aoa_deg: float,
    eps: float = 1e-3,
) -> np.ndarray:
    """
    Central-difference gradient of the RANS loss w.r.t. CST weight vector.

    NOTE: This runs 2 × len(weights) RANS simulations per call.
          Set eps large enough (≥1e-3) to avoid numerical noise.
    """
    grad = np.zeros_like(weights)
    for k in range(len(weights)):
        wp = weights.copy()
        wp[k] += eps
        wm = weights.copy()
        wm[k] -= eps
        lp, _, _ = compute_loss(wp[:n_upper], wp[n_upper:], aoa_deg)
        lm, _, _ = compute_loss(wm[:n_upper], wm[n_upper:], aoa_deg)
        grad[k] = (lp - lm) / (2.0 * eps)
    return grad


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
    """
    Write Selig-format .dat file of the airfoil coordinates.

    The file layout expected by xfoil_validation.py / XFoil:

        <label>
        x1  y1      ← upper surface, TE → LE
        x2  y2
        ...
        xN  yN      ← LE
        xN+1 yN+1   ← lower surface, LE → TE
        ...

    Args:
        weights_upper / weights_lower : CST weights (numpy arrays)
        filepath  : output .dat path
        label     : airfoil name printed on line 1
        num_points: number of points per surface
    """
    wu = jnp.array(weights_upper)
    wl = jnp.array(weights_lower)

    x_norm, y_upper, y_lower = generate_cst_coords(wu, wl, num_points=num_points)
    x_np = np.array(x_norm)
    yu = np.array(y_upper)
    yl = np.array(y_lower)

    # Selig: upper surface TE→LE, then lower surface LE→TE
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
    axes[2].set_ylabel("Drag coefficient  Cd", fontsize=12)
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
    print()
    print("═" * 70)
    print("  RANS Airfoil Shape Optimization — CST + Spalart-Allmaras")
    print(
        f"  Grid: {HEIGHT}×{WIDTH}  cell={CELL_SIZE*1000:.1f} mm  "
        f"Re={U_INF/NU_LAM:.1e}  AoA={AoA_DEG}°"
    )
    print(
        f"  Iterations: {NUM_ITERATIONS}  |  CST order: {N_ORDER}  "
        f"({N_ORDER+1} weights/surface)"
    )
    print("═" * 70)

    # ── Initial CST weights (NACA 0012 profile) ──────────────────
    n_weights = N_ORDER + 1
    initial_upper = np.array([0.1728, 0.1650, 0.1400, 0.1550, 0.1300, 0.1150])
    initial_lower = -initial_upper  # Ensures perfect symmetry

    assert (
        len(initial_upper) == n_weights and len(initial_lower) == n_weights
    ), "Weight arrays must have length N_ORDER + 1"

    params = np.concatenate([initial_upper, initial_lower])  # flat numpy vector
    n_upper = n_weights

    # ── Adam optimizer state (pure numpy for simplicity outside JAX) ─────────
    m = np.zeros_like(params)  # first moment
    v_m = np.zeros_like(params)  # second moment
    beta1, beta2, eps_adam = 0.9, 0.999, 1e-8

    loss_history, cl_history, cd_history = [], [], []

    print("\n  Evaluating initial design ...")
    loss0, Cl0, Cd0 = compute_loss(params[:n_upper], params[n_upper:], AoA_DEG)
    print(f"  Initial → Loss={loss0:.5f}  Cl={Cl0:.4f}  Cd={Cd0:.5f}")

    loss_history.append(loss0)
    cl_history.append(Cl0)
    cd_history.append(Cd0)

    print(
        f"\n  Starting optimization ...\n  {'Iter':>5}  {'Loss':>10}  {'Cl':>9}  {'Cd':>9}  {'Cl/Cd':>9}  {'Time(s)':>8}"
    )
    print("  " + "-" * 65)

    t_start = time.time()

    for it in range(1, NUM_ITERATIONS + 1):
        t_iter = time.time()

        # Finite-difference gradient
        grad = finite_difference_gradient(params, n_upper, AoA_DEG)

        # Adam update
        m = beta1 * m + (1.0 - beta1) * grad
        v_m = beta2 * v_m + (1.0 - beta2) * grad**2
        m_hat = m / (1.0 - beta1**it)
        v_hat = v_m / (1.0 - beta2**it)
        params = params - LEARNING_RATE * m_hat / (np.sqrt(v_hat) + eps_adam)

        # Evaluate updated design
        loss_val, Cl, Cd = compute_loss(params[:n_upper], params[n_upper:], AoA_DEG)
        loss_history.append(loss_val)
        cl_history.append(Cl)
        cd_history.append(Cd)

        cl_cd_ratio = Cl / Cd if abs(Cd) > 1e-12 else 0.0
        elapsed = time.time() - t_iter

        print(
            f"  {it:>5}  {loss_val:>10.5f}  {Cl:>9.4f}  {Cd:>9.5f}"
            f"  {cl_cd_ratio:>9.3f}  {elapsed:>8.1f}s"
        )

    total_time = time.time() - t_start
    print("  " + "-" * 65)
    print(f"\n  Total wall-clock time: {total_time/60:.1f} min")

    final_upper = params[:n_upper]
    final_lower = params[n_upper:]

    # Summary
    print(f"\n  {'Metric':<20} {'Initial':>10}  {'Final':>10}")
    print(f"  {'─'*42}")
    print(f"  {'Loss':<20} {loss_history[0]:>10.5f}  {loss_history[-1]:>10.5f}")
    print(f"  {'Cl':<20} {cl_history[0]:>10.4f}  {cl_history[-1]:>10.4f}")
    print(f"  {'Cd':<20} {cd_history[0]:>10.5f}  {cd_history[-1]:>10.5f}")
    cl_cd_0 = cl_history[0] / cd_history[0] if abs(cd_history[0]) > 1e-12 else 0
    cl_cd_f = cl_history[-1] / cd_history[-1] if abs(cd_history[-1]) > 1e-12 else 0
    print(f"  {'Cl/Cd':<20} {cl_cd_0:>10.3f}  {cl_cd_f:>10.3f}")
    drag_red = (
        (1.0 - loss_history[-1] / loss_history[0]) * 100 if loss_history[0] != 0 else 0
    )
    print(f"\n  Loss reduction : {drag_red:.1f}%")

    # ── Export .dat file ─────────────────────────────────────────────────────
    print("\n  Writing XFoil-compatible .dat file ...")
    write_dat_file(
        final_upper,
        final_lower,
        filepath=OUTPUT_DAT,
        label=(
            f"RANS-Optimized CST Airfoil  "
            f"AoA={AoA_DEG}deg  Re={U_INF/NU_LAM:.0e}  "
            f"Cl={cl_history[-1]:.4f}  Cd={cd_history[-1]:.5f}"
        ),
    )
    print("\n  To validate with XFoil, edit xfoil_validation.py:")
    print(f'    input_file  = "{OUTPUT_DAT}"')
    print('    output_polar = "optimized_airfoil_polar.txt"')

    # ── Plots ────────────────────────────────────────────────────────────────
    print("\n  Generating plots ...")
    plot_optimization_history(loss_history, cl_history, cd_history, OUTPUT_HIST)
    plot_shape_comparison(
        initial_upper,
        initial_lower,
        final_upper,
        final_lower,
        OUTPUT_SHAPE,
    )

    print("\n  Done.")
    print("═" * 70)


if __name__ == "__main__":
    main()
