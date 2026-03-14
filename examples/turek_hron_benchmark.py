import jax
import jax.numpy as jnp
import time

from turbodiff.core.fluid_grid_jax import FluidGrid, FluidState
from turbodiff.core.utils import apply_zero_velocity_at_solids


@jax.jit
def inject_parabolic_inlet(grid: FluidGrid, state: FluidState) -> FluidState:
    # Inlet velocity profile: U(y) = 4 * Umax * y * (H - y) / H^2
    Umax = 0.3
    H = 0.41

    i_indices = jnp.arange(grid.height)
    y = (i_indices + 0.5) * grid.cell_size

    u_profile = 4.0 * Umax * y * (H - y) / (H**2)

    u = state.velocity.u
    v = state.velocity.v

    # Set inlet at the leftmost column of faces
    u = u.at[:, 0:2].set(u_profile[:, None])
    v = v.at[:, 0:1].set(0.0)

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
def step_turek(grid: FluidGrid, state: FluidState) -> FluidState:
    # Diffuse velocity!
    state = grid.diffuse_velocity(state, num_iters=40)

    # Advect velocity
    state = grid.advect_velocity(state)

    # Inject parabolic inlet condition
    state = inject_parabolic_inlet(grid, state)

    # Enforce solid boundaries fully before pressure solve
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

    # Pressure project
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


@jax.jit
def compute_forces(
    u: jnp.ndarray,
    v: jnp.ndarray,
    p: jnp.ndarray,
    cell_size: float,
    nu: float,
    rho: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    # Cylinder is at center (0.2, 0.2) with D=0.1
    # We create a tight CV around it, e.g., x from 0.14 to 0.26, y from 0.14 to 0.26
    # In terms of grid indices (cell_size = 0.01):
    j1, j2 = 14, 26
    i1, i2 = 14, 26

    P_left = jnp.sum(p[i1:i2, j1 - 1]) * cell_size
    P_right = jnp.sum(p[i1:i2, j2]) * cell_size
    F_pressure_x = P_left - P_right

    P_top = jnp.sum(p[i1 - 1, j1:j2]) * cell_size
    P_bottom = jnp.sum(p[i2, j1:j2]) * cell_size
    F_pressure_y = P_top - P_bottom

    u_right = u[i1:i2, j2]
    v_right = 0.5 * (v[i1:i2, j2 - 1] + v[i1:i2, j2])

    u_left = u[i1:i2, j1]
    v_left = 0.5 * (v[i1:i2, j1 - 1] + v[i1:i2, j1])

    u_bottom = 0.5 * (u[i2 - 1, j1:j2] + u[i2, j1:j2])
    v_bottom = v[i2, j1:j2]

    u_top = 0.5 * (u[i1 - 1, j1:j2] + u[i1, j1:j2])
    v_top = v[i1, j1:j2]

    Flux_out_x = (
        jnp.sum(rho * u_right * u_right) * cell_size
        - jnp.sum(rho * u_left * u_left) * cell_size
        + jnp.sum(rho * v_bottom * u_bottom) * cell_size
        - jnp.sum(rho * v_top * u_top) * cell_size
    )

    Flux_out_y = (
        jnp.sum(rho * u_right * v_right) * cell_size
        - jnp.sum(rho * u_left * v_left) * cell_size
        + jnp.sum(rho * v_bottom * v_bottom) * cell_size
        - jnp.sum(rho * v_top * v_top) * cell_size
    )

    mu = rho * nu

    du_dx_right = (u[i1:i2, j2 + 1] - u[i1:i2, j2 - 1]) / (2 * cell_size)
    du_dy_right = (u[i1 + 1 : i2 + 1, j2] - u[i1 - 1 : i2 - 1, j2]) / (2 * cell_size)
    dv_dx_right = (v[i1:i2, j2] - v[i1:i2, j2 - 1]) / cell_size
    tau_xx_right = 2 * mu * du_dx_right
    tau_xy_right = mu * (du_dy_right + dv_dx_right)

    du_dx_left = (u[i1:i2, j1 + 1] - u[i1:i2, j1 - 1]) / (2 * cell_size)
    du_dy_left = (u[i1 + 1 : i2 + 1, j1] - u[i1 - 1 : i2 - 1, j1]) / (2 * cell_size)
    dv_dx_left = (v[i1:i2, j1] - v[i1:i2, j1 - 1]) / cell_size
    tau_xx_left = 2 * mu * du_dx_left
    tau_xy_left = mu * (du_dy_left + dv_dx_left)

    du_dy_bottom = (u[i2, j1:j2] - u[i2 - 1, j1:j2]) / cell_size
    dv_dx_bottom = (v[i2, j1 + 1 : j2 + 1] - v[i2, j1 - 1 : j2 - 1]) / (2 * cell_size)
    dv_dy_bottom = (v[i2 + 1, j1:j2] - v[i2 - 1, j1:j2]) / (2 * cell_size)
    tau_xy_bottom = mu * (du_dy_bottom + dv_dx_bottom)
    tau_yy_bottom = 2 * mu * dv_dy_bottom

    du_dy_top = (u[i1, j1:j2] - u[i1 - 1, j1:j2]) / cell_size
    dv_dx_top = (v[i1, j1 + 1 : j2 + 1] - v[i1, j1 - 1 : j2 - 1]) / (2 * cell_size)
    dv_dy_top = (v[i1 + 1, j1:j2] - v[i1 - 1, j1:j2]) / (2 * cell_size)
    tau_xy_top = mu * (du_dy_top + dv_dx_top)
    tau_yy_top = 2 * mu * dv_dy_top

    F_viscous_x = (
        jnp.sum(tau_xx_right - tau_xx_left) * cell_size
        + jnp.sum(tau_xy_bottom - tau_xy_top) * cell_size
    )
    F_viscous_y = (
        jnp.sum(tau_xy_right - tau_xy_left) * cell_size
        + jnp.sum(tau_yy_bottom - tau_yy_top) * cell_size
    )

    F_drag = F_pressure_x + F_viscous_x - Flux_out_x
    F_lift = F_pressure_y + F_viscous_y - Flux_out_y

    F_lift = -F_lift

    return F_drag, F_lift


def main():
    print("Setting up Turek-Hron DFG-2D-1 benchmark (Re=20)...")
    height = 41
    width = 220
    cell_size = 0.01
    dt = 0.005
    nu = 0.001
    rho = 1.0

    sim = FluidGrid(
        height=height,
        width=width,
        cell_size=cell_size,
        dt=dt,
        diffusion=0.0,
        viscosity=nu,
        boundary_type=0,
        visualise=False,
    )

    # Solid mask setup
    solid_mask = jnp.zeros((height, width), dtype=float)

    # Top and bottom walls
    solid_mask = solid_mask.at[0, :].set(1.0)
    solid_mask = solid_mask.at[-1, :].set(1.0)

    # Cylinder mask
    i_grid, j_grid = jnp.meshgrid(jnp.arange(height), jnp.arange(width), indexing="ij")
    x = (j_grid + 0.5) * cell_size
    y = (i_grid + 0.5) * cell_size
    dist = jnp.sqrt((x - 0.2) ** 2 + (y - 0.2) ** 2)
    cylinder_mask = jnp.where(dist < 0.05, 1.0, 0.0)

    solid_mask = jnp.maximum(solid_mask, cylinder_mask)
    sim.solid_mask = solid_mask

    state = sim.create_initial_state()

    # Dynamic pressure
    Umax = 0.3
    Umean = (2.0 / 3.0) * Umax
    D = 0.1
    q = 0.5 * rho * (Umean**2) * D

    step = 0
    Cd_prev = 0.0

    print("Starting simulation loop... (Press Ctrl+C to stop)")
    print(f"{'Step':>6} | {'Cd':>8} | {'Cl':>8} | {'deltaCd':>8}")

    # Warm up JIT
    start_time = time.time()
    state = step_turek(sim, state)
    _ = compute_forces(
        state.velocity.u, state.velocity.v, state.pressure.values, cell_size, nu, rho
    )
    print(f"JIT Compilation took {time.time() - start_time:.2f} seconds")

    while step <= 15000:
        state = step_turek(sim, state)

        if step % 50 == 0 and step > 0:
            F_drag, F_lift = compute_forces(
                state.velocity.u,
                state.velocity.v,
                state.pressure.values,
                cell_size,
                nu,
                rho,
            )

            Cd = float(F_drag) / q
            Cl = float(F_lift) / q

            deltaCd = abs(Cd - Cd_prev)
            print(f"{step:06d} | {Cd:8.5f} | {Cl:8.5f} | {deltaCd:8.6f}")

            if step > 1000 and deltaCd < 1e-6:
                print("\nSteady state reached!")
                print(f"Final Cd = {Cd:.5f}")
                print(f"Final Cl = {Cl:.5f}")
                print("Benchmark reference Cd = 5.5795")
                print(
                    f"Diff                   = {abs(Cd - 5.5795):.5f} = {abs(Cd - 5.5795) / 5.5795 * 100:.2f}%"
                )
                break

            Cd_prev = Cd

        step += 1

    print("Simulation finished.")


if __name__ == "__main__":
    main()
