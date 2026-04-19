"""
NACA 0012 RANS Benchmark — Spalart-Allmaras turbulence model
==============================================================
Validates the SA-1994 one-equation turbulence model implemented in
FluidGrid against published NACA 0012 aerodynamic coefficients.

Setup:
  - Grid  : 200 × 600 cells, cell_size = 0.005 m
  - Domain: 1.0 m × 3.0 m  (height × width)
  - Chord : c = 0.6 m,  airfoil centred at (x=0.9 m, y=0.5 m)
  - Re    : 2 × 10⁶  (U∞=1 m/s, ν=5×10⁻⁷ m²/s)
  - AoA   : 0° and 4°

Reference (NASA TMR, fully turbulent SA, Re=6M, chord-based):
  AoA=0°: Cl≈0.000, Cd≈0.0054
  AoA=4°: Cl≈0.439, Cd≈0.0059

Run:
  cd /Users/musab/FYP/TurboDiff
  .venv/bin/python examples/naca0012_rans_benchmark.py
"""

import time
import math
import jax
import jax.numpy as jnp
import numpy as np

from turbodiff.core.fluid_grid_jax import FluidGrid, FluidState
from turbodiff.core.utils import apply_ibm_continuous_forcing

# ─────────────────────────────────────────────────────────────────────────────
# Grid / simulation parameters
# ─────────────────────────────────────────────────────────────────────────────
HEIGHT = 200  # cells in y
WIDTH = 600  # cells in x
CELL_SIZE = 0.005  # m   → domain 1.0 m × 3.0 m
DT = 0.0005  # s   (CFL ~ 0.1 at U=1 m/s, h=0.005 m)
NU_LAM = 5e-7  # kinematic viscosity  →  Re = 1/5e-7 = 2e6
RHO = 1.0  # kg/m³
U_INF = 1.0  # m/s  free-stream speed

CHORD = 0.6  # m
AIRFOIL_X0 = 0.9  # m  — x-position of leading edge (in domain)
AIRFOIL_Y0 = 0.5  # m  — y-position of mid-chord

ANGLES_OF_ATTACK = [0.0, 4.0]  # degrees

N_WARMUP = 200  # steps before recording
N_STEPS = 2000  # total simulation steps per AoA
REPORT_EVERY = 100  # print Cl/Cd every N steps


# ─────────────────────────────────────────────────────────────────────────────
# NACA 0012 geometry helpers
# ─────────────────────────────────────────────────────────────────────────────


def naca0012_thickness(x_norm: float) -> float:
    """Half-thickness at normalised chord position x ∈ [0,1]."""
    t = 0.12
    c0, c1, c2, c3, c4 = 0.2969, -0.1260, -0.3516, 0.2843, -0.1015
    return (t / 0.2) * (
        c0 * math.sqrt(x_norm)
        + c1 * x_norm
        + c2 * x_norm**2
        + c3 * x_norm**3
        + c4 * x_norm**4
    )


def build_solid_mask_naca(
    height: int,
    width: int,
    cell_size: float,
    chord: float,
    x0: float,
    y0: float,
    aoa_deg: float,
) -> jnp.ndarray:
    """
    Build a floating-point solid mask for NACA 0012 at given AoA.

    The mask is 1.0 inside the airfoil, 0.0 outside.
    Uses an analytical point-in-polygon test against the upper/lower
    surface profiles discretised at N_PANELS chord-stations.

    Args:
        height, width : grid dimensions (cells)
        cell_size     : m/cell
        chord         : chord length (m)
        x0, y0        : leading-edge position (m)
        aoa_deg       : angle of attack (degrees)

    Returns:
        solid_mask: jnp float32 array (height, width)
    """
    aoa_rad = math.radians(aoa_deg)
    cos_a, sin_a = math.cos(aoa_rad), math.sin(aoa_rad)

    N_PANELS = 500
    xs_norm = np.linspace(0.0, 1.0, N_PANELS)

    # Upper and lower surfaces in the body frame (chord-aligned, leading edge at origin)
    upper = np.array([(x, naca0012_thickness(x)) for x in xs_norm])  # (N, 2)
    lower = np.array([(x, -naca0012_thickness(x)) for x in xs_norm])

    # Rotate by -AoA and scale by chord, then translate to (x0, y0) in domain
    def body_to_domain(pts_norm):
        # pts_norm: (N, 2) in chord units
        pts = pts_norm * chord  # physical body frame
        # rotate by -aoa around leading edge
        xr = pts[:, 0] * cos_a + pts[:, 1] * sin_a
        yr = -pts[:, 0] * sin_a + pts[:, 1] * cos_a
        # translate: leading edge at (x0, y0)
        xd = xr + x0
        yd = yr + y0
        return xd, yd

    xu, yu = body_to_domain(upper)
    xl, yl = body_to_domain(lower)

    # Cell-centre grid in physical coordinates
    j_idx = np.arange(width)
    i_idx = np.arange(height)
    jj, ii = np.meshgrid(j_idx, i_idx)  # (H, W)

    # Super-sampling for anti-aliased (continuous) solid mask
    N_SUB = 4
    solid_accum = np.zeros((height, width), dtype=np.float32)

    for sub_i in range(N_SUB):
        for sub_j in range(N_SUB):
            # Sub-grid cell centres
            cx = (jj + (sub_j + 0.5) / N_SUB) * cell_size  # physical x
            cy = (ii + (sub_i + 0.5) / N_SUB) * cell_size  # physical y

            # Transform cell to body frame
            cx_rel = cx - x0
            cy_rel = cy - y0

            # Rotate INTO body frame (inverse: +aoa)
            cx_body = cx_rel * cos_a - cy_rel * sin_a  # along chord
            cy_body = cx_rel * sin_a + cy_rel * cos_a  # perpendicular

            # Normalised chord position
            xi = cx_body / chord

            # Compute half-thickness at each xi
            xi_c = np.clip(xi, 0.0, 1.0)
            t0 = 0.12 / 0.2
            c0, c1, c2, c3, c4 = 0.2969, -0.1260, -0.3516, 0.2843, -0.1015
            half_t = (
                chord
                * t0
                * (
                    c0 * np.sqrt(np.maximum(xi_c, 0.0))
                    + c1 * xi_c
                    + c2 * xi_c**2
                    + c3 * xi_c**3
                    + c4 * xi_c**4
                )
            )

            # Inside if: 0 ≤ xi ≤ 1  AND  |cy_body| ≤ half_t
            inside = (xi >= 0.0) & (xi <= 1.0) & (np.abs(cy_body) <= half_t)
            solid_accum += inside.astype(np.float32)

    # Average sub-samples for fractional solid coverage
    solid = solid_accum / (N_SUB * N_SUB)

    # Top/bottom boundaries use free-slip (symmetry) BCs enforced in the RANS
    # step, NOT no-slip solid walls. Treating them as solid creates spurious
    # channel blockage (tunnel interference) that inflates Cd.
    return jnp.array(solid)


# ─────────────────────────────────────────────────────────────────────────────
# Inlet injection
# ─────────────────────────────────────────────────────────────────────────────


@jax.jit
def inject_uniform_inlet(
    state: FluidState,
    u_inlet: float,
    v_inlet: float,
) -> FluidState:
    """Set uniform inlet velocity at left boundary (first 2 face columns)."""
    u = state.velocity.u
    v = state.velocity.v
    u = u.at[:, 0:2].set(u_inlet)
    v = v.at[:, 0:2].set(v_inlet)  # was 0:1 — now consistent with u columns
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
    """
    Enforce free-slip (symmetry) BCs at the top and bottom domain boundaries.

    - Normal velocity v = 0 at the top (face row 0) and bottom (face row H).
    - Tangential velocity u satisfies du/dy = 0: copy from the first interior row.

    This replaces the previous solid-wall treatment which caused tunnel
    interference (blockage) and inflated Cd.
    """
    u = state.velocity.u
    v = state.velocity.v

    # No normal flow through horizontal boundaries
    v = v.at[0, :].set(0.0)  # top v-face
    v = v.at[-1, :].set(0.0)  # bottom v-face

    # Zero tangential gradient: copy from first/last interior u-row
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
    """
    Enforce a Neumann (zero-gradient) outflow BC at the right domain boundary.

    Copies velocity from the second-to-last column into the last column so
    that vortices and wakes can exit cleanly without reflecting pressure waves.
    """
    u = state.velocity.u
    v = state.velocity.v

    # du/dx = dv/dx = 0 at the right boundary
    u = u.at[:, -1].set(u[:, -2])
    v = v.at[:, -1].set(v[:, -2])

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
# One RANS step (velocity-only, no density transport)
# ─────────────────────────────────────────────────────────────────────────────


def make_rans_step(grid: FluidGrid, u_inlet: float, v_inlet: float):
    """Return a JIT-compiled RANS step function."""

    @jax.jit
    def rans_step(state: FluidState) -> FluidState:
        # SA turbulence transport
        state = grid.step_sa_turbulence(state, grid._wall_dist, num_diff_iters=4)
        nu_eff = grid.compute_effective_viscosity(state)

        # Velocity diffusion with turbulent viscosity
        state = grid.diffuse_velocity(state, num_iters=20, nu_eff_field=nu_eff)

        # Advect velocity
        state = grid.advect_velocity(state)

        # Re-inject inlet BC
        state = inject_uniform_inlet(state, u_inlet, v_inlet)

        # Free-slip on top/bottom walls (replaces old solid-wall treatment)
        state = apply_slip_top_bottom_bc(state)

        # Enforce solid BCs (airfoil no-penetration) using IBM
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

        # Pressure projection — 300 GS iters (up from 60) for adequate convergence
        # on the 200×600 grid.  Warm-start from previous pressure helps a lot.
        state = grid.solve_pressure(state, num_iters=300)
        state = grid.project_velocity(state)

        # Re-inject inlet and BCs (projection step can corrupt boundaries)
        state = inject_uniform_inlet(state, u_inlet, v_inlet)
        state = apply_slip_top_bottom_bc(state)

        # Neumann outflow at right boundary — lets wake/vortices exit cleanly
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
# Force integration
# ─────────────────────────────────────────────────────────────────────────────


@jax.jit
def compute_aerodynamic_forces(
    state: FluidState,
    solid_mask: jnp.ndarray,
    cell_size: float,
    nu_eff: jnp.ndarray,
    rho: float,
    aoa_rad: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Integrate pressure + viscous stresses on a control surface surrounding
    the airfoil and project onto lift/drag axes.

    Uses a cell-face approach: for every interior fluid cell adjacent to a
    solid cell, accumulate the pressure force and viscous traction.

    Returns:
        (F_drag, F_lift)  — forces in wind axes (N/m span)
    """
    u = state.velocity.u
    v = state.velocity.v
    p = state.pressure.values
    h = cell_size
    mu = rho * nu_eff  # dynamic viscosity field (h, w)

    height, width = solid_mask.shape
    fluid = 1.0 - solid_mask  # 1 in fluid, 0 in solid

    # --- Pressure force via surface integration ---
    # Right face of cell (i, j): outward normal = +x
    # Contribution to Fx if cell is fluid and right neighbour is solid:
    is_right_wall = fluid[:, :-1] * solid_mask[:, 1:]  # (H, W-1)
    is_left_wall = fluid[:, 1:] * solid_mask[:, :-1]  # (H, W-1)
    is_bottom_wall = fluid[:-1, :] * solid_mask[1:, :]  # (H-1, W)
    is_top_wall = fluid[1:, :] * solid_mask[:-1, :]  # (H-1, W)

    # Pressure forces (F = p * n * dA, dA = h * 1 span)
    Fpx = (
        jnp.sum(p[:, :-1] * is_right_wall) * h  # +x on right wall
        - jnp.sum(p[:, 1:] * is_left_wall) * h  # -x on left wall
    )
    Fpy = (
        jnp.sum(p[:-1, :] * is_bottom_wall) * h  # +y on bottom wall
        - jnp.sum(p[1:, :] * is_top_wall) * h  # -y on top wall
    )

    # --- Viscous force (shear stress at walls) ---
    # u-gradient at right solid wall: du/dy at (i, j+1/2)
    mu_c = mu  # (H, W)

    # du/dy at right faces (approximate): use velocity difference across face
    u_cc = 0.5 * (u[:, :-1] + u[:, 1:])  # (H, W) cell-centre u
    v_cc = 0.5 * (v[:-1, :] + v[1:, :])  # (H, W) cell-centre v

    dudy_int = jnp.zeros_like(p)
    dvdx_int = jnp.zeros_like(p)
    dudx_int = jnp.zeros_like(p)
    dvdy_int = jnp.zeros_like(p)

    dudy_int = dudy_int.at[1:-1, :].set((u_cc[2:, :] - u_cc[:-2, :]) / (2.0 * h))
    dvdx_int = dvdx_int.at[:, 1:-1].set((v_cc[:, 2:] - v_cc[:, :-2]) / (2.0 * h))
    dudx_int = dudx_int.at[:, 1:-1].set((u_cc[:, 2:] - u_cc[:, :-2]) / (2.0 * h))
    dvdy_int = dvdy_int.at[1:-1, :].set((v_cc[2:, :] - v_cc[:-2, :]) / (2.0 * h))

    # tau_xy = mu*(du/dy + dv/dx),  tau_xx = 2*mu*du/dx
    tau_xy = mu_c * (dudy_int + dvdx_int)
    tau_xx = 2.0 * mu_c * dudx_int
    tau_yy = 2.0 * mu_c * dvdy_int

    # Viscous forces on solid walls
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

    # Total force in x and y (domain axes)
    Fx = Fpx + Fvx
    Fy = Fpy + Fvy

    # Rotate from domain axes to wind axes (drag = along free-stream)
    cos_a = jnp.cos(aoa_rad)
    sin_a = jnp.sin(aoa_rad)
    F_drag = Fx * cos_a + Fy * sin_a
    F_lift = -Fx * sin_a + Fy * cos_a

    return F_drag, F_lift


# ─────────────────────────────────────────────────────────────────────────────
# Main benchmark loop
# ─────────────────────────────────────────────────────────────────────────────


def run_aoa(aoa_deg: float):
    aoa_rad = math.radians(aoa_deg)
    u_inlet = U_INF * math.cos(aoa_rad)
    v_inlet = U_INF * math.sin(aoa_rad)

    print(f"\n{'='*65}")
    print(f"  NACA 0012, Re={U_INF/NU_LAM:.1e}, AoA={aoa_deg:.1f}°")
    print(f"{'='*65}")

    # Build solid mask (includes wall boundaries)
    print("  Building airfoil geometry and solid mask...")
    solid_mask = build_solid_mask_naca(
        HEIGHT, WIDTH, CELL_SIZE, CHORD, AIRFOIL_X0, AIRFOIL_Y0, aoa_deg
    )

    # Construct grid with SA model enabled
    grid = FluidGrid(
        height=HEIGHT,
        width=WIDTH,
        cell_size=CELL_SIZE,
        dt=DT,
        diffusion=0.0,
        viscosity=NU_LAM,
        rho=RHO,
        boundary_type=0,  # no automatic boundary — we set it manually
        use_sa_turbulence=True,
        visualise=True,
    )
    # Assign the pre-computed solid mask (with airfoil + walls)
    grid.solid_mask = solid_mask.astype(float)

    # Recompute wall distance for the actual airfoil mask
    print("  Computing wall distances (this may take ~30 s on first run)...")
    t0 = time.time()
    grid._wall_dist = grid._compute_wall_distance(grid.solid_mask)
    print(f"  Wall distance computed in {time.time()-t0:.1f} s")

    # Initial state
    state = grid.create_initial_state()

    # Initialise uniform free-stream velocity
    u_init = jnp.ones((HEIGHT, WIDTH + 1)) * u_inlet
    v_init = jnp.ones((HEIGHT + 1, WIDTH)) * v_inlet
    u_init, v_init = apply_ibm_continuous_forcing(u_init, v_init, solid_mask)
    state = state.__class__(
        density=state.density,
        velocity=state.velocity.with_values(u_init, v_init),
        pressure=state.pressure,
        solid_mask=state.solid_mask,
        sources=state.sources,
        nu_tilde=state.nu_tilde,
        time=0.0,
        step=0,
    )

    # Right-boundary outflow: open the right column
    grid.solid_mask = grid.solid_mask.at[1:-1, -1].set(0.0)
    grid._wall_dist = grid._compute_wall_distance(grid.solid_mask)

    # Build RANS step function (JIT-compiled)
    rans_step = make_rans_step(grid, u_inlet, v_inlet)

    # Dynamic pressure and reference area
    q_inf = 0.5 * RHO * U_INF**2
    ref_area = CHORD  # per unit span

    # ── Warm up JIT ─────────────────────────────────────────────────────────
    print("  Warming up JIT (first step)...")
    t0 = time.time()
    state = rans_step(state)
    jax.block_until_ready(state.velocity.u)
    print(f"  JIT compilation took {time.time()-t0:.1f} s")

    # ── Simulation loop ──────────────────────────────────────────────────────
    print(f"\n  {'Step':>6}  {'Cl':>9}  {'Cd':>9}  {'max|div|':>10}  {'min ν̃':>10}")
    Cl_prev = 0.0
    converged = False

    wall_t0 = time.time()
    for step in range(1, N_STEPS + 1):
        state = rans_step(state)

        if step % REPORT_EVERY == 0:
            nu_eff = grid.compute_effective_viscosity(state)
            F_drag, F_lift = compute_aerodynamic_forces(
                state, solid_mask, CELL_SIZE, nu_eff, RHO, aoa_rad
            )
            Cl = float(F_lift) / (q_inf * ref_area)
            Cd = float(F_drag) / (q_inf * ref_area)

            # Divergence check
            div = grid.compute_divergence(state)
            max_div = float(jnp.max(jnp.abs(div)))

            # SA field health
            min_nt = float(jnp.min(state.nu_tilde.values))

            elapsed = time.time() - wall_t0
            print(
                f"  {step:>6}  {Cl:>9.5f}  {Cd:>9.6f}  {max_div:>10.2e}  {min_nt:>10.2e}"
                f"  ({elapsed:.1f}s)"
            )

            if step > N_WARMUP and abs(Cl - Cl_prev) < 5e-4 and Cd > 0:
                print("  ✓ Steady state reached")
                converged = True
                break
            Cl_prev = Cl

    # Final forces
    nu_eff = grid.compute_effective_viscosity(state)
    F_drag, F_lift = compute_aerodynamic_forces(
        state, solid_mask, CELL_SIZE, nu_eff, RHO, aoa_rad
    )
    Cl = float(F_lift) / (q_inf * ref_area)
    Cd = float(F_drag) / (q_inf * ref_area)

    return Cl, Cd, converged


def main():
    print("\n" + "═" * 65)
    print("  NACA 0012 RANS Benchmark — Spalart-Allmaras turbulence model")
    print("  TurboDiff (JAX)   Re = 2×10⁶   fully turbulent")
    print("═" * 65)

    # Reference values — NASA TMR NACA 0012, SA-1994, Re=6M (nearest public data)
    # Slightly different Re so discrepancy is expected; just checks sign/magnitude.
    # For Re=2M XFOIL (turbulent) gives rough Cl≈0.000/0.44, Cd≈0.006/0.007
    references = {
        0.0: (0.000, 0.0054),
        4.0: (0.439, 0.0059),
    }

    results = {}
    for aoa in ANGLES_OF_ATTACK:
        Cl, Cd, conv = run_aoa(aoa)
        results[aoa] = (Cl, Cd, conv)

    # ── Summary table ─────────────────────────────────────────────────────
    print("\n" + "═" * 65)
    print(
        f"  {'AoA':>5}  {'Cl (TD)':>9}  {'Cd (TD)':>9}  "
        f"{'Cl (ref)':>9}  {'Cd (ref)':>9}  {'Conv?':>6}"
    )
    print("─" * 65)
    for aoa in ANGLES_OF_ATTACK:
        Cl, Cd, conv = results[aoa]
        Cl_r, Cd_r = references[aoa]
        print(
            f"  {aoa:>5.1f}°  {Cl:>9.4f}  {Cd:>9.5f}  "
            f"{Cl_r:>9.4f}  {Cd_r:>9.5f}  {'YES' if conv else 'NO':>6}"
        )
    print("═" * 65)
    print(
        "  Note: TurboDiff uses a coarse 200×600 grid (h=5mm) — expect\n"
        "  O(10-20%) discrepancy vs reference at this resolution.\n"
        "  Increase HEIGHT/WIDTH for finer results."
    )


if __name__ == "__main__":
    main()
