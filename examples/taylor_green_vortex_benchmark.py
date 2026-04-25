from dataclasses import dataclass
import jax
import jax.numpy as jnp
import numpy as np
from turbodiff.core.fluid_grid_jax import FluidGrid, FluidState
from turbodiff.core.benchmark_validations import (
    update_fluid_state,
    get_cell_centered_velocity,
)


@dataclass(frozen=True)
class TaylorGreenMetrics:
    elapsed_time: float
    energy_ratio_sim: float
    energy_ratio_analytic: float
    energy_ratio_rel_error: float
    corr_u: float
    corr_v: float
    div_l2: float
    div_max: float


def _kinetic_energy_density(state: FluidState) -> float:
    u_cc, v_cc = get_cell_centered_velocity(state)
    return 0.5 * float(np.mean(u_cc * u_cc + v_cc * v_cc))


def _build_taylor_green_initial_faces(
    n: int,
    cell_size: float,
    u0: float,
) -> tuple[jnp.ndarray, jnp.ndarray, np.ndarray, np.ndarray]:
    x = (jnp.arange(n) + 0.5) * cell_size
    y = (jnp.arange(n) + 0.5) * cell_size
    yy, xx = jnp.meshgrid(y, x, indexing="ij")

    u_cc = u0 * jnp.sin(xx) * jnp.cos(yy)
    v_cc = -u0 * jnp.cos(xx) * jnp.sin(yy)

    u_face = jnp.zeros((n, n + 1))
    u_face = u_face.at[:, 1:-1].set(0.5 * (u_cc[:, :-1] + u_cc[:, 1:]))
    u_face = u_face.at[:, 0].set(u_cc[:, 0])
    u_face = u_face.at[:, -1].set(u_cc[:, -1])

    v_face = jnp.zeros((n + 1, n))
    v_face = v_face.at[1:-1, :].set(0.5 * (v_cc[:-1, :] + v_cc[1:, :]))
    v_face = v_face.at[0, :].set(v_cc[0, :])
    v_face = v_face.at[-1, :].set(v_cc[-1, :])

    return u_face, v_face, np.array(xx), np.array(yy)


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.ravel()
    b_flat = b.ravel()
    a_std = np.std(a_flat)
    b_std = np.std(b_flat)
    if a_std < 1e-12 or b_std < 1e-12:
        return 0.0
    return float(np.corrcoef(a_flat, b_flat)[0, 1])


def run_taylor_green_vortex_benchmark(
    *,
    n: int = 40,
    dt: float = 0.002,
    viscosity: float = 0.01,
    u0: float = 0.25,
    steps: int = 120,
    pressure_iters: int = 40,
    diffusion_iters: int = 16,
    trim: int = 2,
    visualise: bool = False,
) -> TaylorGreenMetrics:
    """Run a Taylor-Green vortex benchmark on the base CFD solver.

    This benchmark checks:
    - kinetic energy decay against the analytical exponential trend
    - spatial correlation of velocity fields against the analytical shape
    - incompressibility via divergence metrics
    """
    length = 2.0 * np.pi
    cell_size = length / n

    u_face, v_face, xx, yy = _build_taylor_green_initial_faces(n, cell_size, u0)

    grid = FluidGrid(
        height=n,
        width=n,
        cell_size=cell_size,
        dt=dt,
        diffusion=0.0,
        viscosity=viscosity,
        boundary_type=0,
        visualise=visualise,
        show_velocity=True,
        use_sa_turbulence=False,
    )
    state = grid.create_initial_state(velocity_u_init=u_face, velocity_v_init=v_face)
    state_initial = state

    @jax.jit
    def step_fn(sim: FluidGrid, st: FluidState) -> FluidState:
        st = sim.diffuse_velocity(st, num_iters=diffusion_iters)
        st = sim.advect_velocity(st)
        st = sim.solve_pressure(st, num_iters=pressure_iters)
        st = sim.project_velocity(st)
        return update_fluid_state(st, time=st.time + sim.dt, step=st.step + 1)

    _ = step_fn(grid, state)  # JIT warm-up
    state = state_initial
    e0 = _kinetic_energy_density(state)

    state = grid.simulate(state, steps=steps, custom_step_fn=step_fn)

    elapsed = float(state.time)
    e_sim = _kinetic_energy_density(state)
    ratio_sim = e_sim / (e0 + 1e-12)
    ratio_ref = float(np.exp(-2.0 * viscosity * elapsed))
    ratio_rel_error = abs(ratio_sim - ratio_ref) / (ratio_ref + 1e-12)

    u_cc, v_cc = get_cell_centered_velocity(state)
    decay = ratio_ref
    u_ref = u0 * np.sin(xx) * np.cos(yy) * decay
    v_ref = -u0 * np.cos(xx) * np.sin(yy) * decay

    if trim > 0 and 2 * trim < n:
        sl = (slice(trim, -trim), slice(trim, -trim))
        corr_u = _correlation(u_cc[sl], u_ref[sl])
        corr_v = _correlation(v_cc[sl], v_ref[sl])
    else:
        corr_u = _correlation(u_cc, u_ref)
        corr_v = _correlation(v_cc, v_ref)

    u_arr = np.array(state.velocity.u)
    v_arr = np.array(state.velocity.v)
    div = (u_arr[:, 1:] - u_arr[:, :-1]) / cell_size + (
        v_arr[1:, :] - v_arr[:-1, :]
    ) / cell_size

    return TaylorGreenMetrics(
        elapsed_time=elapsed,
        energy_ratio_sim=ratio_sim,
        energy_ratio_analytic=ratio_ref,
        energy_ratio_rel_error=ratio_rel_error,
        corr_u=corr_u,
        corr_v=corr_v,
        div_l2=float(np.sqrt(np.mean(div * div))),
        div_max=float(np.max(np.abs(div))),
    )


def main() -> None:
    metrics = run_taylor_green_vortex_benchmark(visualise=False)

    print("=" * 64)
    print("Taylor-Green Vortex Benchmark (Base CFD)")
    print("=" * 64)
    print(f"elapsed time (s)                : {metrics.elapsed_time:.4f}")
    print(f"energy ratio (sim)              : {metrics.energy_ratio_sim:.5f}")
    print(f"energy ratio (analytic)         : {metrics.energy_ratio_analytic:.5f}")
    print(f"energy ratio relative error     : {metrics.energy_ratio_rel_error:.3f}")
    print(f"velocity shape correlation (u)  : {metrics.corr_u:.4f}")
    print(f"velocity shape correlation (v)  : {metrics.corr_v:.4f}")
    print(f"divergence L2                   : {metrics.div_l2:.6f}")
    print(f"divergence max                  : {metrics.div_max:.6f}")

    pass_energy = metrics.energy_ratio_rel_error <= 0.50
    pass_corr = metrics.corr_u >= 0.55 and metrics.corr_v >= 0.55
    pass_div = metrics.div_l2 <= 0.20 and metrics.div_max <= 1.00

    print("-" * 64)
    print(f"PASS energy trend  : {pass_energy}")
    print(f"PASS shape match   : {pass_corr}")
    print(f"PASS incompressible: {pass_div}")
    print(f"OVERALL PASS       : {pass_energy and pass_corr and pass_div}")


if __name__ == "__main__":
    main()
