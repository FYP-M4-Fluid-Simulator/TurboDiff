"""
Spalart-Allmaras Decay Benchmark
================================
Validates the destruction term of the Spalart-Allmaras (SA) turbulence model.

Setup:
    - Zero velocity everywhere (u=0, v=0) -> No production, no advection.
    - Uniform initial working variable nu_tilde = 0.01. -> No diffusion.
    - Uniform wall distance injected: d = 1.0 m.

Result:
    The only active term is the destruction term:
    D(nu_tilde)/Dt = -cw1 * (nu_tilde/d)^2
    
    This has an exact analytical solution:
    nu_tilde(t) = 1.0 / ( (cw1/d^2) * t + (1.0/nu_tilde(0)) )

We simulate this forward in time and compare the domain-average nu_tilde
to the analytical curve.
"""

import jax
import jax.numpy as jnp
import numpy as np
import time
import matplotlib.pyplot as plt

from turbodiff.core.fluid_grid_jax import FluidGrid, FluidState


def analytical_decay(t_arr, nu0, d, cw1):
    return 1.0 / ((cw1 / (d**2)) * t_arr + (1.0 / nu0))


def main():
    print("=" * 60)
    print("Spalart-Allmaras Decay Benchmark")
    print("=" * 60)

    height = 20
    width = 20
    cell_size = 0.05
    dt = 0.01
    nu_init_val = 0.01
    d_wall = 1.0

    sim = FluidGrid(
        height=height,
        width=width,
        cell_size=cell_size,
        dt=dt,
        viscosity=1e-5,
        boundary_type=0,  # Periodic/open, we don't care because u=v=0
        use_sa_turbulence=True,
    )

    # SA Constants
    cw1 = sim.sa_cw1

    # Overwrite the built-in computed wall distance with a uniform dummy value
    # to artificially activate the destruction term globally.
    wall_dist = jnp.full((height, width), d_wall)

    state = sim.create_initial_state()

    # Initialize uniform nu_tilde
    nu_tilde_vals = jnp.full((height, width), nu_init_val)
    state = state.__class__(
        density=state.density,
        velocity=state.velocity,
        pressure=state.pressure,
        solid_mask=state.solid_mask,
        sources=state.sources,
        nu_tilde=state.nu_tilde.with_values(nu_tilde_vals),
        time=state.time,
        step=state.step,
    )

    # JIT the SA step with the custom wall_dist
    @jax.jit
    def step_decay(state: FluidState):
        return sim.step_sa_turbulence(state, wall_dist)

    print("JIT compilation...")
    t0 = time.time()
    _ = step_decay(state)
    print(f"JIT done in {time.time() - t0:.2f}s\n")

    max_steps = 1000
    times = []
    sim_nuts = []
    ana_nuts = []

    state_curr = state
    for step in range(max_steps + 1):
        if step % 100 == 0:
            current_time = step * dt
            # Average nu_tilde in the domain
            sim_nut = np.mean(np.array(state_curr.nu_tilde.values))
            ana_nut = analytical_decay(current_time, nu_init_val, d_wall, cw1)

            err = abs(sim_nut - ana_nut) / ana_nut * 100
            print(
                f"Step {step:4d} | t = {current_time:5.2f}s | sim = {sim_nut:.6f} | ana = {ana_nut:.6f} | err = {err:.3f}%"
            )

            times.append(current_time)
            sim_nuts.append(sim_nut)
            ana_nuts.append(ana_nut)

        if step < max_steps:
            state_curr = step_decay(state_curr)
            # Advance state time manually since we bypass full simulator step
            state_curr = state_curr.__class__(
                **{
                    **state_curr.__dict__,
                    "time": state_curr.time + dt,
                    "step": state_curr.step + 1,
                }
            )

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    final_err = abs(sim_nuts[-1] - ana_nuts[-1]) / ana_nuts[-1] * 100
    print(f"Final error vs analytical solution: {final_err:.4f}%")
    if final_err < 1.0:
        print("✓ PASS — Destruction term precisely matches mathematical ODE.")
    else:
        print("✗ FAIL — Mismatch in destruction term.")

    # Save a plot
    plt.figure(figsize=(8, 5))
    plt.plot(times, ana_nuts, "k-", label="Analytical ODE", linewidth=2)
    plt.plot(
        times, sim_nuts, "ro", label="JAX SA Solver", markersize=6, fillstyle="none"
    )
    plt.title("Spalart-Allmaras Decay of Isotropic Turbulence")
    plt.xlabel("Time (s)")
    plt.ylabel("$\\tilde{\\nu}$ (Modified Eddy Viscosity)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("sa_decay_results.png", dpi=150)
    print("Saved plot to sa_decay_results.png")


if __name__ == "__main__":
    main()
