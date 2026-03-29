"""Validation benchmarks for the base CFD solver (without SA-RANS)."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from turbodiff.core.fluid_grid_jax import FluidGrid, FluidState
from turbodiff.core.utils import apply_zero_velocity_at_solids


@dataclass(frozen=True)
class BlasiusMetrics:
    station_x: float
    rmse_u_over_uinf: float
    delta99_sim: float
    delta99_ref: float
    delta99_rel_error: float
    wall_u_over_uinf: float


def update_fluid_state(
    state: FluidState,
    *,
    u: jnp.ndarray | None = None,
    v: jnp.ndarray | None = None,
    time: float | None = None,
    step: int | None = None,
) -> FluidState:
    velocity = (
        state.velocity if u is None or v is None else state.velocity.with_values(u, v)
    )
    return state.__class__(
        density=state.density,
        velocity=velocity,
        pressure=state.pressure,
        solid_mask=state.solid_mask,
        sources=state.sources,
        nu_tilde=state.nu_tilde,
        time=state.time if time is None else time,
        step=state.step if step is None else step,
    )


def get_cell_centered_velocity(state: FluidState) -> tuple[np.ndarray, np.ndarray]:
    u_face = np.array(state.velocity.u)
    v_face = np.array(state.velocity.v)
    u_cc = 0.5 * (u_face[:, :-1] + u_face[:, 1:])
    v_cc = 0.5 * (v_face[:-1, :] + v_face[1:, :])
    return u_cc, v_cc


def _blasius_profile(
    eta_max: float = 8.0, d_eta: float = 0.01
) -> tuple[np.ndarray, np.ndarray]:
    """Numerically integrate the Blasius ODE and return (eta, f')."""

    def rhs(f: float, fp: float, fpp: float) -> tuple[float, float, float]:
        return fp, fpp, -0.5 * f * fpp

    eta_vals = np.arange(0.0, eta_max + d_eta, d_eta)
    fp_vals = np.zeros_like(eta_vals)

    # Well-known Blasius shooting value: f''(0) ≈ 0.3320573362
    f = 0.0
    fp = 0.0
    fpp = 0.332057336215

    fp_vals[0] = fp
    for idx in range(1, len(eta_vals)):
        k1 = rhs(f, fp, fpp)
        k2 = rhs(
            f + 0.5 * d_eta * k1[0],
            fp + 0.5 * d_eta * k1[1],
            fpp + 0.5 * d_eta * k1[2],
        )
        k3 = rhs(
            f + 0.5 * d_eta * k2[0],
            fp + 0.5 * d_eta * k2[1],
            fpp + 0.5 * d_eta * k2[2],
        )
        k4 = rhs(
            f + d_eta * k3[0],
            fp + d_eta * k3[1],
            fpp + d_eta * k3[2],
        )

        f += (d_eta / 6.0) * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0])
        fp += (d_eta / 6.0) * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1])
        fpp += (d_eta / 6.0) * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2])

        fp_vals[idx] = fp

    return eta_vals, np.clip(fp_vals, 0.0, 1.05)


def _delta_99(y: np.ndarray, u_by_uinf: np.ndarray) -> float:
    idx = np.argmax(u_by_uinf >= 0.99)
    if u_by_uinf[idx] < 0.99:
        return float(y[-1])
    if idx == 0:
        return float(y[0])
    u0 = u_by_uinf[idx - 1]
    u1 = u_by_uinf[idx]
    y0 = y[idx - 1]
    y1 = y[idx]
    alpha = (0.99 - u0) / (u1 - u0 + 1e-12)
    return float(y0 + alpha * (y1 - y0))


def run_blasius_boundary_layer_benchmark(
    *,
    height: int = 72,
    width: int = 170,
    cell_size: float = 0.01,
    dt: float = 0.003,
    viscosity: float = 0.01,
    u_inf: float = 1.0,
    steps: int = 500,
    station_x: float = 0.7,
    pressure_iters: int = 60,
    diffusion_iters: int = 24,
    visualise: bool = False,
) -> BlasiusMetrics:
    """Run a flat-plate laminar boundary-layer benchmark against Blasius."""

    grid = FluidGrid(
        height=height,
        width=width,
        cell_size=cell_size,
        dt=dt,
        diffusion=0.0,
        viscosity=viscosity,
        boundary_type=0,
        visualise=visualise,
        use_sa_turbulence=False,
    )

    # Bottom wall only; top is free stream and right side acts as open outflow.
    solid_mask = jnp.zeros((height, width), dtype=bool)
    solid_mask = solid_mask.at[-1, :].set(True)
    grid.solid_mask = solid_mask

    state = grid.create_initial_state()

    @jax.jit
    def step_fn(sim: FluidGrid, st: FluidState) -> FluidState:
        st = sim.diffuse_velocity(st, num_iters=diffusion_iters)
        st = sim.advect_velocity(st)

        u = st.velocity.u.at[:, 0:2].set(u_inf)
        v = st.velocity.v.at[:, 0:1].set(0.0)
        u, v = apply_zero_velocity_at_solids(u, v, st.solid_mask)
        st = update_fluid_state(st, u=u, v=v)

        st = sim.solve_pressure(st, num_iters=pressure_iters)
        st = sim.project_velocity(st)

        u = st.velocity.u.at[:, 0:2].set(u_inf)
        v = st.velocity.v.at[:, 0:1].set(0.0)
        u, v = apply_zero_velocity_at_solids(u, v, st.solid_mask)
        return update_fluid_state(st, u=u, v=v, time=st.time + sim.dt, step=st.step + 1)

    state = step_fn(grid, state)  # JIT warm-up
    state = grid.simulate(state, steps=steps, custom_step_fn=step_fn)

    x_idx = max(2, min(width - 3, int(round(station_x / cell_size))))
    x_phys = (x_idx + 0.5) * cell_size

    u_cc, _ = get_cell_centered_velocity(state)
    u_col = u_cc[:, x_idx]

    fluid_rows = np.arange(0, height - 1)  # exclude solid bottom row
    y_from_wall = (height - 1 - fluid_rows - 0.5) * cell_size
    y_from_wall = np.array(y_from_wall)
    u_by_uinf = np.clip(np.array(u_col[fluid_rows]) / (u_inf + 1e-12), 0.0, 1.2)

    # Sort low-to-high y for interpolation/metrics.
    order = np.argsort(y_from_wall)
    y = y_from_wall[order]
    u_by_uinf = u_by_uinf[order]

    eta_vals, fp_vals = _blasius_profile()
    eta = y * np.sqrt(u_inf / (viscosity * x_phys + 1e-12))
    fp_ref = np.interp(np.clip(eta, 0.0, eta_vals[-1]), eta_vals, fp_vals)

    # Focus comparison in the boundary-layer region only.
    mask = eta <= 6.0
    if np.count_nonzero(mask) < 8:
        mask = np.arange(len(y)) < min(len(y), 16)

    rmse = float(np.sqrt(np.mean((u_by_uinf[mask] - fp_ref[mask]) ** 2)))

    delta99_sim = _delta_99(y, u_by_uinf)
    delta99_ref = 5.0 * np.sqrt(viscosity * x_phys / (u_inf + 1e-12))
    delta99_rel_error = abs(delta99_sim - delta99_ref) / (delta99_ref + 1e-12)

    wall_u = float(u_by_uinf[0])

    return BlasiusMetrics(
        station_x=float(x_phys),
        rmse_u_over_uinf=rmse,
        delta99_sim=delta99_sim,
        delta99_ref=float(delta99_ref),
        delta99_rel_error=float(delta99_rel_error),
        wall_u_over_uinf=wall_u,
    )
