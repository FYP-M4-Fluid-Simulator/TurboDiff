"""
Poiseuille Flow Benchmark
=========================
Validates the solver against the exact analytical solution for
pressure-driven laminar channel flow.

Setup:
    - Horizontal channel: L = 5.0 m, H = 1.0 m
    - Parabolic inlet BC (Umax = 0.15 m/s)
    - Top/bottom walls: no-slip (solid)
    - Outflow: open (no boundary, pressure = 0)
    - Re = Umean * H / nu = 0.1 * 1.0 / 0.001 = 100

Analytical solution (fully-developed Poiseuille profile):
    u(y) = 4 * Umax * y * (H - y) / H^2

Expected results:
    - L2 error vs analytical < 2% on this grid (cell_size=0.025 m)
    - Max pointwise error < 3%
    - Flow should reach steady state by t ~ 50s (step ~ 10000)

Reference: Exact Navier-Stokes solution, no external citation needed.
"""

import jax
import jax.numpy as jnp
import time
import numpy as np

from turbodiff.core.fluid_grid_jax import FluidGrid, FluidState
from turbodiff.core.utils import apply_zero_velocity_at_solids

# ── Simulation parameters ────────────────────────────────────────────────────
H = 1.0  # channel height (m)
L = 5.0  # channel length (m)
cell_size = 0.025  # m/cell → grid is 40 × 200
dt = 0.005  # s
nu = 0.001  # kinematic viscosity (m²/s)
rho = 1.0  # density (kg/m³)
Umax = 0.15  # peak inlet velocity (m/s)
Umean = (2.0 / 3.0) * Umax  # mean of parabolic profile

height = int(H / cell_size)  # 40 cells
width = int(L / cell_size)  # 200 cells

Re = Umean * H / nu
print(f"Re = {Re:.1f}")


@jax.jit
def inject_inlet(grid: FluidGrid, state: FluidState) -> FluidState:
    """Parabolic inlet at left boundary."""
    i_indices = jnp.arange(grid.height)
    y = (i_indices + 0.5) * grid.cell_size
    u_profile = 4.0 * Umax * y * (H - y) / (H**2)

    u = state.velocity.u.at[:, 0:2].set(u_profile[:, None])
    v = state.velocity.v.at[:, 0:1].set(0.0)

    return state.__class__(
        density=state.density,
        velocity=state.velocity.with_values(u, v),
        pressure=state.pressure,
        solid_mask=state.solid_mask,
        sources=state.sources,
        time=state.time,
        step=state.step,
    )


@jax.jit
def step_poiseuille(grid: FluidGrid, state: FluidState) -> FluidState:
    state = grid.diffuse_velocity(state, num_iters=40)
    state = grid.advect_velocity(state)
    state = inject_inlet(grid, state)

    u, v = apply_zero_velocity_at_solids(
        state.velocity.u, state.velocity.v, state.solid_mask
    )
    state = state.__class__(
        density=state.density,
        velocity=state.velocity.with_values(u, v),
        pressure=state.pressure,
        solid_mask=state.solid_mask,
        sources=state.sources,
        time=state.time,
        step=state.step,
    )

    state = grid.solve_pressure(state, num_iters=80)
    state = grid.project_velocity(state)

    return state.__class__(
        density=state.density,
        velocity=state.velocity,
        pressure=state.pressure,
        solid_mask=state.solid_mask,
        sources=state.sources,
        time=state.time + grid.dt,
        step=state.step + 1,
    )


def analytical_u(y_arr: np.ndarray) -> np.ndarray:
    """Exact parabolic Poiseuille profile."""
    return 4.0 * Umax * y_arr * (H - y_arr) / (H**2)


def evaluate_profile(state: FluidState, grid: FluidGrid):
    """Compare mid-channel cross-section to analytical solution."""
    j_mid = width // 2  # column at x = L/2

    # Cell-centred u: average of left and right face velocities
    u_field = state.velocity.u
    u_center_col = 0.5 * (u_field[:, j_mid] + u_field[:, j_mid + 1])
    u_sim = np.array(u_center_col)

    # Physical y-coordinates of cell centres
    i_arr = np.arange(height)
    y_arr = (i_arr + 0.5) * cell_size  # metres

    u_ref = analytical_u(y_arr)

    l2_rel = np.linalg.norm(u_sim - u_ref) / (np.linalg.norm(u_ref) + 1e-12) * 100
    max_rel = np.max(np.abs(u_sim - u_ref)) / (np.max(u_ref) + 1e-12) * 100

    return y_arr, u_sim, u_ref, l2_rel, max_rel


def main():
    print("=" * 60)
    print("Poiseuille Flow Benchmark")
    print("=" * 60)
    print(f"Grid: {height} × {width}  |  cell_size = {cell_size} m")
    print(f"dt = {dt} s  |  nu = {nu} m²/s  |  Re = {Re:.1f}")
    print(f"Umax = {Umax} m/s  |  Umean = {Umean:.4f} m/s")
    print()

    # Build grid (no built-in boundary; we add walls manually)
    sim = FluidGrid(
        height=height,
        width=width,
        cell_size=cell_size,
        dt=dt,
        diffusion=0.0,
        viscosity=nu,
        boundary_type=0,  # no automatic boundary
        # visualise=True,   # Enabled for real-time visualization
        # show_velocity=True
    )

    # Solid mask: only top and bottom walls
    solid_mask = jnp.zeros((height, width), dtype=float)
    solid_mask = solid_mask.at[0, :].set(1.0)  # top wall
    solid_mask = solid_mask.at[-1, :].set(1.0)  # bottom wall
    sim.solid_mask = solid_mask

    state = sim.create_initial_state()

    max_steps = 15000
    check_every = 500
    l2_prev = None

    def callback(grid: FluidGrid, st: FluidState) -> bool:
        nonlocal l2_prev
        if st.step % check_every == 0:
            y, u_s, u_r, l2, max_e = evaluate_profile(st, grid)
            u_cl_s = u_s[height // 2]
            u_cl_r = u_r[height // 2]
            print(
                f"{st.step:6d} | {l2:10.4f} | {max_e:11.4f} | {u_cl_s:9.5f} | {u_cl_r:9.5f}"
            )

            if l2_prev is not None and abs(l2 - l2_prev) < 0.001:
                print("\n✓ Converged to steady state.")
                return True
            l2_prev = l2
        return False

    # ── JIT warm-up ──────────────────────────────────────────────────────────
    t0 = time.time()
    _ = step_poiseuille(sim, state)
    print(f"JIT compilation: {time.time() - t0:.2f} s")

    # ── Simulation loop ───────────────────────────────────────────────────────
    print(
        f"\n{'Step':>6} | {'L2 err (%)':>10} | {'Max err (%)':>11} | {'Sim Ucl':>9} | {'Ref Ucl':>9}"
    )
    print("-" * 60)

    # Use the standard simulation machine with our custom physics and callback
    state = sim.simulate(
        state, steps=max_steps, custom_step_fn=step_poiseuille, callback_fn=callback
    )

    # ── Final report ──────────────────────────────────────────────────────────
    y_arr, u_sim, u_ref, l2, max_e = evaluate_profile(state, sim)

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"L2 relative error  : {l2:.3f}%")
    print(f"Max pointwise error: {max_e:.3f}%")
    print()
    print("Expected L2 error  : < 2%   (this grid:  cell_size=0.025 m)")
    print("Expected max error : < 3%")
    if l2 < 2.0:
        print("✓ PASS — within expected tolerance")
    else:
        print("✗ FAIL — above expected tolerance")

    # ── Profile table ─────────────────────────────────────────────────────────
    print("\nVelocity profile at x = L/2:")
    print(f"{'y (m)':>8}  {'u_sim':>9}  {'u_ref':>9}  {'err (%)':>8}")
    print("-" * 40)
    for i in range(0, height, max(1, height // 10)):
        err_pct = abs(u_sim[i] - u_ref[i]) / (u_ref[i] + 1e-12) * 100
        print(f"{y_arr[i]:8.4f}  {u_sim[i]:9.5f}  {u_ref[i]:9.5f}  {err_pct:8.3f}")


if __name__ == "__main__":
    main()
