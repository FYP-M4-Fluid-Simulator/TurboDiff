"""
Spalart-Allmaras Flat Plate Boundary Layer Benchmark
====================================================
Tests the Spalart-Allmaras solver on a zero-pressure gradient flat plate.
The flow develops naturally boundary layer from an inlet free-stream.

Setup:
    - Domain: L = 1.0 m, H = 0.1 m
    - Grid: 50 (H) x 500 (W) -> cell_size = 0.002 m
    - Viscosity (nu) = 1e-4 m²/s
    - Freestream (U_inf) = 10.0 m/s
    - Re_L = U_inf * L / nu = 10^5

Goal:
    - Verify that the resulting turbulent velocity profile fits 
      the Law of the Wall (viscous sublayer u+ = y+  and 
      log-law u+ = 1/k * ln(y+) + B)
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import time

from turbodiff.core.fluid_grid_jax import FluidGrid, FluidState
from turbodiff.core.utils import apply_zero_velocity_at_solids

# Settings
L = 1.0
H = 0.1
cell_size = 0.002
height = int(H / cell_size)
width = int(L / cell_size)

nu = 2.5e-5
U_inf = 10.0
dt = 0.0001
max_steps = 8000  # 8 flow-through times (L=1.0, U=10)


@jax.jit
def apply_wall_function(
    u_P_array: jnp.ndarray, y_P: float, nu_val: float
) -> jnp.ndarray:
    """Explicitly compute u_tau for wall cells using Newton-Raphson on the Log Law."""
    kappa = 0.41
    B = 5.0
    E = jnp.exp(kappa * B)

    # Laminar guess
    u_P_clean = jnp.maximum(u_P_array, 1e-12)
    u_tau = jnp.sqrt(nu_val * u_P_clean / y_P)

    def nr_step(u_t, _):
        y_plus = y_P * u_t / nu_val
        f = u_P_clean - (u_t / kappa) * jnp.log(E * y_plus + 1e-10)
        df_du = -(1.0 / kappa) * (jnp.log(E * y_plus + 1e-10) + 1.0)
        u_t_new = u_t - f / (df_du - 1e-10)
        u_t_new = jnp.maximum(u_t_new, 1e-10)
        return u_t_new, None

    u_tau_nr, _ = jax.lax.scan(nr_step, u_tau, None, length=5)

    y_plus_final = y_P * u_tau_nr / nu_val
    valid_log = y_plus_final > 11.6

    # Return NR result if in log-layer, else laminar exact value
    return jnp.where(valid_log, u_tau_nr, jnp.sqrt(nu_val * u_P_clean / y_P))


def inject_inlet(grid: FluidGrid, state: FluidState) -> FluidState:
    """Free stream inlet on the left boundary and top boundary."""
    # Left boundary: u = U_inf, v = 0
    u = state.velocity.u.at[:, 0:2].set(U_inf)
    v = state.velocity.v.at[:, 0:1].set(0.0)

    # Top boundary: u = U_inf, v = 0
    u = u.at[-1, :].set(U_inf)
    v = v.at[-1, :].set(0.0)

    # Note: nu_tilde is explicitly injected in step_flat_plate

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


def main():
    print("=" * 60)
    print("Spalart-Allmaras Flat Plate Boundary Layer")
    print("=" * 60)
    print(f"Grid: {height}x{width} | cell_size={cell_size}m | nu={nu}")
    print(f"Re_L: {U_inf * L / nu:.1e} | dt: {dt}")

    sim = FluidGrid(
        height=height,
        width=width,
        cell_size=cell_size,
        dt=dt,
        viscosity=nu,
        boundary_type=0,
        use_sa_turbulence=True,
    )

    # Standard flat plate wall: Solid at the bottom (i=0)
    solid_mask = jnp.zeros((height, width))
    solid_mask = solid_mask.at[0, :].set(1.0)
    sim.solid_mask = solid_mask

    # Needs a wall dist update after changing mask
    sim._wall_dist = sim._compute_wall_distance(solid_mask)

    state = sim.create_initial_state()
    # Initialize flow to free-stream U_inf to speed up convergence
    u_init = jnp.full_like(state.velocity.u, U_inf)
    u_init = u_init.at[0, :].set(0.0)  # no slip exactly at wall
    v_init = jnp.zeros_like(state.velocity.v)

    # SA recommends freestream nu_tilde ~ 3 * nu. We trip the boundary layer
    # slightly stronger so it develops turbulence quickly
    nu_init_val = 5.0 * nu
    nu_tilde_vals = jnp.full((height, width), nu_init_val)

    state = state.__class__(
        density=state.density,
        velocity=state.velocity.with_values(u_init, v_init),
        pressure=state.pressure,
        solid_mask=state.solid_mask,
        sources=state.sources,
        nu_tilde=state.nu_tilde.with_values(nu_tilde_vals),
        time=state.time,
        step=state.step,
    )

    @jax.jit
    def step_flat_plate(state: FluidState) -> FluidState:
        # SA turbulence modeling step
        state = sim.step_sa_turbulence(state, sim._wall_dist, num_diff_iters=10)

        # Inject freestream nu_tilde at boundaries
        nut = state.nu_tilde.values
        nut = nut.at[:, 0:2].set(nu_init_val)  # Inlet
        nut = nut.at[-1, :].set(nu_init_val)  # Top boundary
        state = state.__class__(
            **{**state.__dict__, "nu_tilde": state.nu_tilde.with_values(nut)}
        )

        # Effective viscosity
        nu_eff = sim.compute_effective_viscosity(state)

        # --- Apply Wall Function Boundary ---
        u_field = state.velocity.u
        u_P = 0.5 * (u_field[1, :-1] + u_field[1, 1:])  # Average to cell center
        y_P = cell_size / 2.0

        u_tau = apply_wall_function(u_P, y_P, sim.viscosity)

        # Override the effective viscosity at the wall cell to force the correct shear stress
        # tau_w = u_tau^2 = nu_eff_wall * (u_P / y_P)  =>  nu_eff_wall = (u_tau^2 * y_P) / u_P
        nu_eff_wall = (u_tau**2 * y_P) / jnp.maximum(u_P, 1e-12)

        nu_eff = nu_eff.at[1, :].set(nu_eff_wall)
        # ------------------------------------

        # Fluid Dynamics steps
        state = sim.diffuse_velocity(state, num_iters=30, nu_eff_field=nu_eff)
        state = sim.advect_velocity(state)
        state = inject_inlet(sim, state)

        u, v = apply_zero_velocity_at_solids(
            state.velocity.u, state.velocity.v, state.solid_mask
        )
        state = state.__class__(
            **{**state.__dict__, "velocity": state.velocity.with_values(u, v)}
        )

        # Pressure Projection
        state = sim.solve_pressure(state, num_iters=40)
        state = sim.project_velocity(state)

        return state.__class__(
            **{**state.__dict__, "time": state.time + dt, "step": state.step + 1}
        )

    print("\nJIT compiling step function...")
    t0 = time.time()
    _ = step_flat_plate(state)
    print(f"JIT complete in {time.time() - t0:.2f}s")

    print("\nSimulating...")
    t_start = time.time()

    check_every = 200
    for step in range(1, max_steps + 1):
        state = step_flat_plate(state)

        if step % check_every == 0:
            print(f"Step {step:4d} / {max_steps} | Time simulated: {step * dt:.4f}s")

    print(f"Simulation completed in {time.time() - t_start:.2f}s")

    # ----- Extract Profile Station at x/L = 0.9 -----
    station_idx_x = int(0.9 * width)
    u_field = state.velocity.u
    u_station = 0.5 * (u_field[:, station_idx_x] + u_field[:, station_idx_x + 1])
    u_prof = np.array(u_station)

    # Effective viscosity profile at the wall to compute friction velocity
    nu_eff_jax = sim.compute_effective_viscosity(state)
    nu_eff = np.array(nu_eff_jax)

    # To compute skin friction, we calculate wall shear stress tau_w
    # tau_w = rho * nu_eff * du/dy at y=0.
    # We use the first cell near the wall
    du = u_prof[1] - 0.0  # Cell 1 velocity minus wall velocity (0.0)
    dy_wall = cell_size / 2.0  # Cell center distance

    # In pure SA, effectively kinematic nu is evaluated at the first cell
    tau_w = nu_eff[1, station_idx_x] * (du / dy_wall)
    u_tau = np.sqrt(tau_w)  # assuming rho=1

    print("\n" + "=" * 60)
    print("BOUNDARY LAYER ANALYSIS")
    print("=" * 60)
    print(f"Station: x = {0.9 * L:.2f} m")
    print(f"tau_w = {tau_w:.5f}")
    print(f"u_tau = {u_tau:.5f}")

    # Calculate y+ and u+
    y_phys = (np.arange(height) + 0.5) * cell_size
    y_plus = y_phys * u_tau / nu
    u_plus = u_prof / u_tau

    print(f"First cell y+ = {y_plus[1]:.2f}")
    if y_plus[1] > 11.6:
        print(
            "✓ First cell firmly inside Log-Layer (Wall Function properly activated!)"
        )
    elif y_plus[1] < 5:
        print("✓ First cell is safely within viscous sublayer (y+ < 5)")
    else:
        print("! WARNING: First cell might be outside viscous sublayer. Refine grid.")

    # Generate Law of the Wall plot
    plt.figure(figsize=(10, 6))

    # Ignore the solid wall cell y_plus[0], plot from index 1 upwards
    valid_indices = range(1, height)
    plot_y = y_plus[valid_indices]
    plot_u = u_plus[valid_indices]

    # Filter out values where u_plus hasn't developed fully (optional, for neat plot)
    plotFilter = plot_y < 1000

    plt.semilogx(
        plot_y[plotFilter],
        plot_u[plotFilter],
        "ro-",
        label="JAX SA Profile",
        markersize=4,
    )

    # Reference analytical log-law
    y_ref = np.logspace(-1, 3, 100)
    u_sublayer = y_ref
    u_loglaw = (1.0 / 0.41) * np.log(y_ref) + 5.0

    plt.semilogx(
        y_ref[y_ref < 11.6],
        u_sublayer[y_ref < 11.6],
        "k--",
        label="Viscous Sublayer (u+=y+)",
    )
    plt.semilogx(
        y_ref[y_ref > 11.6],
        u_loglaw[y_ref > 11.6],
        "b--",
        label="Log-Law (k=0.41, B=5.0)",
    )

    plt.title("Flat Plate Law of the Wall (SA Model)")
    plt.xlabel("$y^+$")
    plt.ylabel("$u^+$")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("sa_flat_plate_results.png", dpi=150)
    print("Saved plot to sa_flat_plate_results.png")


if __name__ == "__main__":
    main()
