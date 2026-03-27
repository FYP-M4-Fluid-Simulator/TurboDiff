"""
Lid-Driven Cavity Benchmark (Re = 100)
=======================================
Validates the solver against the well-known Ghia et al. (1982) reference data
for incompressible flow in a square lid-driven cavity.

Setup:
    - Square cavity: 1.0 m × 1.0 m
    - All four walls: no-slip (solid)
    - Top lid: u = U_LID = 1.0 m/s  (moving wall, Dirichlet BC)
    - Re = U_LID * L / nu = 1.0 * 1.0 / 0.01 = 100

Expected results (64×64 interior grid):
    - RMS error vs Ghia ≤ ~5%   (u along vertical centreline)
    - RMS error vs Ghia ≤ ~7%   (v along horizontal centreline)
    - Coarser grids will show 8–12% error
    - Finer grids (128×128) reduce to < 2%

Reference:
    Ghia, U., Ghia, K.N., Shin, C.T. (1982).
    "High-Re solutions for incompressible flow using the Navier-Stokes equations
     and a multigrid method."
    Journal of Computational Physics, 48(3), 387–411.
"""

import jax
import time
import numpy as np

from turbodiff.core.fluid_grid_jax import FluidGrid, FluidState
from turbodiff.core.utils import apply_zero_velocity_at_solids

# ── Parameters ───────────────────────────────────────────────────────────────
L = 1.0  # cavity side length (m)
U_LID = 1.0  # lid velocity (m/s)
nu = 0.01  # kinematic viscosity → Re = 100
rho = 1.0  # density (kg/m³)

N = 64  # interior cells per side (grid resolution)
cell_size = L / N  # ≈ 0.015625 m
height = N + 2  # +2 for top/bottom wall cells = 66
width = N + 2  # +2 for left/right wall cells = 66
dt = 0.005  # s  (CFL = U*dt/dx ≈ 0.32  ✓)

Re = U_LID * L / nu
print(f"Re = {Re:.0f}  |  Grid: {N}×{N} interior  |  cell_size = {cell_size:.5f} m")


# ── Ghia et al. (1982) reference data, Re = 100 ──────────────────────────────
# u-velocity along vertical centreline (x = 0.5)
# y is normalised (0 = bottom wall, 1 = top lid)
GHIA_U_Y = np.array(
    [
        1.0000,
        0.9766,
        0.9688,
        0.9609,
        0.9531,
        0.8516,
        0.7344,
        0.6172,
        0.5000,
        0.4531,
        0.2813,
        0.1719,
        0.1016,
        0.0703,
        0.0625,
        0.0547,
        0.0000,
    ]
)
GHIA_U = np.array(
    [
        1.00000,
        0.84123,
        0.78871,
        0.73722,
        0.68717,
        0.23151,
        0.00332,
        -0.13641,
        -0.20581,
        -0.20581,
        -0.15662,
        -0.10150,
        -0.06434,
        -0.04775,
        -0.04192,
        -0.03717,
        0.00000,
    ]
)

# v-velocity along horizontal centreline (y = 0.5)
# x is normalised (0 = left wall, 1 = right wall)
GHIA_V_X = np.array(
    [
        1.0000,
        0.9688,
        0.9531,
        0.9453,
        0.9063,
        0.8594,
        0.8047,
        0.5000,
        0.2344,
        0.2266,
        0.1563,
        0.0938,
        0.0781,
        0.0703,
        0.0625,
        0.0000,
    ]
)
GHIA_V = np.array(
    [
        0.00000,
        -0.05906,
        -0.07391,
        -0.08864,
        -0.09795,
        -0.10139,
        -0.10386,
        0.05765,
        0.27485,
        0.28003,
        0.30271,
        0.30239,
        0.29093,
        0.27468,
        0.24499,
        0.00000,
    ]
)


# ── Step function ─────────────────────────────────────────────────────────────
@jax.jit
def step_cavity(grid: FluidGrid, state: FluidState) -> FluidState:
    """One time step for the cavity.

    Lid BC is applied as a Dirichlet override: after enforcing no-slip at all
    solid walls, we set u[row=0, :] = U_LID (the top wall u-faces).
    This must be done both before and after pressure projection.
    """
    # Viscous diffusion
    state = grid.diffuse_velocity(state, num_iters=50)

    # Advection
    state = grid.advect_velocity(state)

    # Enforce no-slip at all walls, then override top lid
    u, v = apply_zero_velocity_at_solids(
        state.velocity.u, state.velocity.v, state.solid_mask
    )
    u = u.at[0, :].set(U_LID)  # moving lid (top wall horizontal face)

    state = state.__class__(
        density=state.density,
        velocity=state.velocity.with_values(u, v),
        pressure=state.pressure,
        solid_mask=state.solid_mask,
        sources=state.sources,
        time=state.time,
        step=state.step,
    )

    # Pressure solve + projection
    state = grid.solve_pressure(state, num_iters=100)
    state = grid.project_velocity(state)

    # Re-apply BCs after projection
    u, v = apply_zero_velocity_at_solids(
        state.velocity.u, state.velocity.v, state.solid_mask
    )
    u = u.at[0, :].set(U_LID)

    return state.__class__(
        density=state.density,
        velocity=state.velocity.with_values(u, v),
        pressure=state.pressure,
        solid_mask=state.solid_mask,
        sources=state.sources,
        time=state.time + grid.dt,
        step=state.step + 1,
    )


def extract_centerlines(state: FluidState):
    """
    Returns:
        y_norm   : y/L for vertical centreline (Ghia convention: 0=bottom, 1=top)
        u_cl     : u-velocity at x = L/2 (cell-centred)
        x_norm   : x/L for horizontal centreline
        v_cl     : v-velocity at y = L/2 (cell-centred)
    """
    u_field = np.array(state.velocity.u)
    v_field = np.array(state.velocity.v)

    # ── u along vertical centreline at j = width//2 ──────────────────────────
    j_mid = width // 2  # centre column index (interior)
    # Cell-centred u = average of left and right u-faces
    u_cc = 0.5 * (u_field[:, j_mid] + u_field[:, j_mid + 1])
    # Skip wall cells (rows 0 and height-1)
    u_cl = u_cc[1:-1]  # length N (interior rows)

    # Physical y of interior cell centres; flip so 0=bottom, 1=top
    i_arr = np.arange(N)
    # row 1 is the topmost interior row (just below the lid), row N is bottom
    # In code: row index 1 → highest y, row index N → lowest y
    y_phys = (N - i_arr - 0.5) * cell_size  # high y first
    y_norm = y_phys / L  # normalised 0→1 (low→high)
    # Flip to match Ghia order (low y first)
    y_norm = y_norm[::-1]
    u_cl = u_cl[::-1]

    # ── v along horizontal centreline at i = height//2 ───────────────────────
    i_mid = height // 2  # centre row index
    # Cell-centred v = average of top and bottom v-faces
    v_cc = 0.5 * (v_field[i_mid, :] + v_field[i_mid + 1, :])
    # Skip wall cells (columns 0 and width-1)
    v_cl = v_cc[1:-1]  # length N

    j_arr = np.arange(N)
    x_phys = (j_arr + 0.5) * cell_size
    x_norm = x_phys / L

    return y_norm, u_cl, x_norm, v_cl


def compute_rms_error(sim_y, sim_u, ref_y, ref_u):
    """Interpolate simulation onto reference y-points and compute RMS error."""
    u_interp = np.interp(ref_y, sim_y, sim_u)
    rms = np.sqrt(np.mean((u_interp - ref_u) ** 2))
    u_range = np.max(ref_u) - np.min(ref_u)
    return rms / (u_range + 1e-12) * 100  # relative %


def main():
    print("=" * 60)
    print("Lid-Driven Cavity Benchmark  (Re = 100)")
    print("=" * 60)
    print(f"Grid: {N}×{N} interior  |  cell_size = {cell_size:.5f} m")
    print(f"dt = {dt} s  |  nu = {nu} m²/s  |  Re = {Re:.0f}")
    print()

    # ── Build grid (all walls solid via boundary_type=1) ───────────────────
    sim = FluidGrid(
        height=height,
        width=width,
        cell_size=cell_size,
        dt=dt,
        diffusion=0.0,
        viscosity=nu,
        boundary_type=1,  # complete boundary — all four walls solid
        visualise=False,
    )

    state = sim.create_initial_state()

    # ── JIT warm-up ──────────────────────────────────────────────────────────
    t0 = time.time()
    state = step_cavity(sim, state)
    print(f"JIT compilation: {time.time() - t0:.2f} s\n")

    # ── Simulation loop ───────────────────────────────────────────────────────
    print(
        f"{'Step':>6} | {'u RMS err (%)':>14} | {'v RMS err (%)':>14} | {'delta_u_cl':>10}"
    )
    print("-" * 60)

    max_steps = 20000
    check_every = 1000
    u_cl_prev = None

    for step in range(1, max_steps + 1):
        state = step_cavity(sim, state)

        if step % check_every == 0:
            y_norm, u_cl, x_norm, v_cl = extract_centerlines(state)

            rms_u = compute_rms_error(y_norm, u_cl, GHIA_U_Y, GHIA_U)
            rms_v = compute_rms_error(x_norm, v_cl, GHIA_V_X, GHIA_V)

            u_mid = float(u_cl[N // 2])
            delta = abs(u_mid - u_cl_prev) if u_cl_prev is not None else float("inf")
            print(f"{step:6d} | {rms_u:14.4f} | {rms_v:14.4f} | {delta:10.6f}")

            if u_cl_prev is not None and delta < 1e-5:
                print("\n✓ Converged to steady state.")
                break
            u_cl_prev = u_mid

    # ── Final report ──────────────────────────────────────────────────────────
    y_norm, u_cl, x_norm, v_cl = extract_centerlines(state)
    rms_u = compute_rms_error(y_norm, u_cl, GHIA_U_Y, GHIA_U)
    rms_v = compute_rms_error(x_norm, v_cl, GHIA_V_X, GHIA_V)

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"u-centreline RMS error vs Ghia: {rms_u:.2f}%")
    print(f"v-centreline RMS error vs Ghia: {rms_v:.2f}%")
    print()
    print(f"Expected (N={N}×{N}):  u-RMS ≤ 5%,  v-RMS ≤ 7%")
    status_u = "✓ PASS" if rms_u <= 5.0 else "✗ FAIL"
    status_v = "✓ PASS" if rms_v <= 7.0 else "✗ FAIL"
    print(f"u-centreline: {status_u}   v-centreline: {status_v}")

    # ── Profile table (u) ─────────────────────────────────────────────────────
    print("\nu-velocity along vertical centreline (x = 0.5):")
    print(f"{'y/L':>8}  {'u_sim':>9}  {'u_Ghia':>9}  {'err':>8}")
    print("-" * 42)
    for yg, ug in zip(GHIA_U_Y, GHIA_U):
        u_interp = float(np.interp(yg, y_norm, u_cl))
        err = abs(u_interp - ug)
        print(f"{yg:8.4f}  {u_interp:9.5f}  {ug:9.5f}  {err:8.5f}")


if __name__ == "__main__":
    main()
