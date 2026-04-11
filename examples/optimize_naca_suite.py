"""
Multi-Condition Airfoil Optimization Suite — NACA 0012
======================================================
Optimizes NACA 0012 across a 3x3 matrix of (Reynolds Number x AoA).
Each case runs for a fixed number of iterations (default 30).
Validates against Xfoil after each step and tracks the best airfoil per case.

Conditions:
  - RE: 1e5, 1e6, 6e6
  - AoA: 0, 4, 8

Outputs:
  - optimization_summary.txt : results for all 9 cases.
  - shapes_re_{RE}.png       : 4-way shape comparison per RE.
  - best_airfoil_re_{RE}_aoa_{AOA}.dat : Selig format coordinates.
"""

import os
import math
import subprocess
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib

matplotlib.use("Agg")
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
from scipy.ndimage import distance_transform_edt
from scipy.special import comb as sp_comb

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL PARAMETERS (Configurable)
# ─────────────────────────────────────────────────────────────────────────────
NUM_ITERATIONS = 30
LEARNING_RATE = 0.005
N_ORDER = 5

RE_LIST = [1e5, 1e6, 6e6]
AOA_LIST = [0, 4, 8]

XFOIL_PATH = "/Users/musab/Xfoil-for-Mac/bin/xfoil"
OUTPUT_SUMMARY = "optimization_summary.txt"

# Physics/Grid (from optimize_airfoil_rans.py)
HEIGHT = 128
WIDTH = 384
CELL_SIZE = 0.01
DT = 0.0002
RHO = 1.0
U_INF = 1.0
CHORD = 1.0
AIRFOIL_X0 = 0.4
AIRFOIL_Y0 = HEIGHT * CELL_SIZE / 2.0
N_SIM_STEPS = 1000
N_SUB = 3

# NACA 0012 Approximate CST Weights (Order 5)
NACA0012_W_U = jnp.array([0.1726, 0.1269, 0.1441, 0.1080, 0.1750, 0.1471])
NACA0012_W_L = -NACA0012_W_U

# Geometry constraints (from optimize_airfoil_rans.py)
MIN_THICKNESS_RATIO = 0.10
MAX_THICKNESS_RATIO = 0.25
W_DRAG = 1.0
W_LIFT = 0.5
W_GEO = 10.0

# ─────────────────────────────────────────────────────────────────────────────
# JAX Infrastructure (Grid & RANS)
# ─────────────────────────────────────────────────────────────────────────────
_GRID_CACHE = {}


def compute_aerodynamic_forces(
    state: FluidState,
    solid_mask: jnp.ndarray,
    nu_eff: jnp.ndarray,
    rho: float,
    aoa_rad: float,
) -> tuple:
    u, v, p = state.velocity.u, state.velocity.v, state.pressure.values
    h, mu = CELL_SIZE, rho * nu_eff
    fluid = 1.0 - solid_mask
    is_right_wall = fluid[:, :-1] * solid_mask[:, 1:]
    is_left_wall = fluid[:, 1:] * solid_mask[:, :-1]
    is_bottom_wall = fluid[:-1, :] * solid_mask[1:, :]
    is_top_wall = fluid[1:, :] * solid_mask[:-1, :]
    Fpx = jnp.sum(p[:, :-1] * is_right_wall) * h - jnp.sum(p[:, 1:] * is_left_wall) * h
    Fpy = jnp.sum(p[:-1, :] * is_bottom_wall) * h - jnp.sum(p[1:, :] * is_top_wall) * h
    u_cc, v_cc = 0.5 * (u[:, :-1] + u[:, 1:]), 0.5 * (v[:-1, :] + v[1:, :])
    dudy = jnp.zeros_like(p).at[1:-1, :].set((u_cc[2:, :] - u_cc[:-2, :]) / (2.0 * h))
    dvdx = jnp.zeros_like(p).at[:, 1:-1].set((v_cc[:, 2:] - v_cc[:, :-2]) / (2.0 * h))
    dudx = jnp.zeros_like(p).at[:, 1:-1].set((u_cc[:, 2:] - u_cc[:, :-2]) / (2.0 * h))
    dvdy = jnp.zeros_like(p).at[1:-1, :].set((v_cc[2:, :] - v_cc[:-2, :]) / (2.0 * h))
    tau_xy, tau_xx, tau_yy = mu * (dudy + dvdx), 2.0 * mu * dudx, 2.0 * mu * dvdy
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
    Fx, Fy = Fpx + Fvx, Fpy + Fvy
    cos_a, sin_a = jnp.cos(aoa_rad), jnp.sin(aoa_rad)
    return Fx * cos_a + Fy * sin_a, -Fx * sin_a + Fy * cos_a


def compute_wall_dist_np(solid_mask_jax: jnp.ndarray) -> jnp.ndarray:
    solid_np = np.array(solid_mask_jax)
    solid_np_open = solid_np.copy()
    solid_np_open[1:-1, -1] = 0.0
    wall_dist_np = distance_transform_edt(solid_np_open < 0.5) * CELL_SIZE
    return jax.lax.stop_gradient(
        jnp.array(np.maximum(wall_dist_np.astype(np.float32), 1e-10))
    )


def make_rans_step(grid: FluidGrid, u_inlet: float, v_inlet: float):
    @jax.jit
    def apply_freestream_bc(state: FluidState) -> FluidState:
        u, v = state.velocity.u, state.velocity.v
        u = (
            u.at[:, :2]
            .set(u_inlet)
            .at[:, -1]
            .set(jnp.maximum(u[:, -2], u_inlet * 0.5))
            .at[-1, :]
            .set(u_inlet)
            .at[0, :]
            .set(u_inlet)
        )
        v = (
            v.at[:, :2]
            .set(v_inlet)
            .at[:, -1]
            .set(v[:, -2])
            .at[-1, :]
            .set(v_inlet)
            .at[0, :]
            .set(v_inlet)
        )
        return state.__class__(
            **{**state.__dict__, "velocity": state.velocity.with_values(u, v)}
        )

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
            **{**state.__dict__, "velocity": state.velocity.with_values(u, v)}
        )
        state = grid.solve_pressure(state, num_iters=60)
        state = grid.project_velocity(state)
        state = apply_freestream_bc(state)
        return state.__class__(
            **{**state.__dict__, "time": state.time + grid.dt, "step": state.step + 1}
        )

    return rans_step


def get_or_build_grid(u_inlet, v_inlet, nu_lam):
    key = (round(u_inlet, 8), round(v_inlet, 8), round(nu_lam, 12))
    if key not in _GRID_CACHE:
        grid = FluidGrid(HEIGHT, WIDTH, CELL_SIZE, DT, 0.0, nu_lam, RHO)
        grid.use_sa_turbulence = True
        rans_step = make_rans_step(grid, u_inlet, v_inlet)
        _GRID_CACHE[key] = (grid, rans_step)
    return _GRID_CACHE[key]


# ─────────────────────────────────────────────────────────────────────────────
# Xfoil Helper & Parser
# ─────────────────────────────────────────────────────────────────────────────
def run_xfoil(dat_file, re, aoa):
    polar_file = dat_file.replace(".dat", ".txt")
    print(f"Running Xfoil for {dat_file} with Re={re} and AoA={aoa}")
    if os.path.exists(polar_file):
        os.remove(polar_file)

    # Leveraged command structure from xfoil_validation.py
    commands = f"""LOAD {dat_file}
PANE
OPER
ITER 300
VISC {re}
PACC
{polar_file}

ALFA {aoa}
QUIT
"""

    try:
        env = os.environ.copy()
        env["DISPLAY"] = ":0"
        process = subprocess.Popen(
            XFOIL_PATH,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        process.communicate(input=commands, timeout=20)
    except Exception:
        return None
    if not os.path.exists(polar_file):
        return None
    try:
        with open(polar_file, "r") as f:
            for line in f:
                parts = line.split()
                if (
                    len(parts) >= 3
                    and parts[0].replace(".", "", 1).replace("-", "", 1).isdigit()
                ):
                    if abs(float(parts[0]) - aoa) < 0.01:
                        return float(parts[1]), float(parts[2])
    except Exception:
        pass
    return None


def write_dat(wu, wl, filepath, label="CST Airfoil", num_points=200):
    wu_j, wl_j = jnp.array(wu), jnp.array(wl)
    x_norm, y_upper, y_lower = generate_cst_coords(wu_j, wl_j, num_points=num_points)
    x_np = np.array(x_norm)
    yu = np.array(y_upper)
    yl = np.array(y_lower)
    # Selig format: upper surface TE→LE, then lower surface LE→TE
    # Both include the LE point (x=0); XFoil's PANE handles the duplicate fine.
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


# ─────────────────────────────────────────────────────────────────────────────
# Optimized Geometry Logic (Soft Mask)
# ─────────────────────────────────────────────────────────────────────────────
def build_soft_mask(weights_u, weights_l, subcells):
    sharpness = 500.0
    accum = jnp.zeros((HEIGHT, WIDTH))
    for sc in subcells:
        y_u, y_l = sc["C_g"] * (sc["B_g"] @ weights_u), sc["C_g"] * (
            sc["B_g"] @ weights_l
        )
        above = jax.nn.sigmoid(sharpness * (sc["cy_body"] - y_l))
        below = jax.nn.sigmoid(sharpness * (y_u - sc["cy_body"]))
        in_chord = jax.nn.sigmoid(sharpness * sc["xi_grid"]) * jax.nn.sigmoid(
            sharpness * (1.0 - sc["xi_grid"])
        )
        accum += above * below * in_chord
    return accum / (N_SUB * N_SUB)


def precompute_subcells():
    n_order = N_ORDER
    i_vals = np.arange(n_order + 1)
    # Keep binom as a numpy array so all sub-cell arithmetic stays in NumPy
    # (avoids the JAX bool-dtype FutureWarning from mixed numpy/JAX broadcasting)
    binom = np.array(
        [sp_comb(n_order, i, exact=True) for i in i_vals], dtype=np.float32
    )
    subcells = []
    jj, ii = np.meshgrid(np.arange(WIDTH), np.arange(HEIGHT))
    for si in range(N_SUB):
        for sj in range(N_SUB):
            cx = (jj + (sj + 0.5) / N_SUB) * CELL_SIZE
            cy = (ii + (si + 0.5) / N_SUB) * CELL_SIZE
            cx_rel = (cx - AIRFOIL_X0).astype(np.float32)
            cy_rel = (cy - AIRFOIL_Y0).astype(np.float32)
            xi_grid = cx_rel / CHORD
            xi_c = np.clip(xi_grid, 0.0, 1.0).astype(np.float32)
            C_g = (np.sqrt(np.maximum(xi_c, 0.0)) * (1.0 - xi_c)).astype(np.float32)
            xi_c_col = xi_c[..., None]  # (H, W, 1)
            B_g = (
                binom * xi_c_col**i_vals * (1.0 - xi_c_col) ** (n_order - i_vals)
            ).astype(np.float32)
            subcells.append(
                {
                    "cy_body": jnp.array(cy_rel),
                    "xi_grid": jnp.array(xi_grid),
                    "C_g": jnp.array(C_g),
                    "B_g": jnp.array(B_g),
                }
            )
    return subcells


# ─────────────────────────────────────────────────────────────────────────────
# MAIN OPTIMIZATION LOOP
# ─────────────────────────────────────────────────────────────────────────────
def run_suite():
    print(f"Starting Multi-Condition Optimization Suite ({NUM_ITERATIONS} iterations)")
    subcells = precompute_subcells()
    if os.path.exists(OUTPUT_SUMMARY):
        os.remove(OUTPUT_SUMMARY)
    final_airfoils = {}

    for re_val in RE_LIST:
        nu_lam = U_INF * CHORD / re_val
        for aoa_val in AOA_LIST:
            print(f"\n>>> Case: RE={re_val}, AoA={aoa_val}")
            u_inlet, v_inlet = U_INF * math.cos(
                math.radians(aoa_val)
            ), U_INF * math.sin(math.radians(aoa_val))
            grid, rans_step = get_or_build_grid(u_inlet, v_inlet, nu_lam)
            wu, wl = jnp.copy(NACA0012_W_U), jnp.copy(NACA0012_W_L)
            best_cl_cd, best_cl, best_cd, best_state = -1.0, 0.0, 0.0, (wu, wl)
            aoa_rad = jnp.float32(math.radians(aoa_val))
            q_inf, ref_area = 0.5 * RHO * U_INF**2, CHORD

            # Differentiable loss function closes over persistent grid and rans_step
            # but takes hard_mask and wall_dist (wd) as arguments so they can be updated.
            def loss_fn(params, hard_mask, wd):
                w_u, w_l = params[:6], params[6:]
                sm = build_soft_mask(w_u, w_l, subcells)
                sm_open = sm.at[1:-1, -1].set(0.0)
                # Geometry loss
                x_, yu_, yl_ = generate_cst_coords(w_u, w_l, 200)
                geo_loss = crossover_validity_loss(
                    yu_, yl_
                ) + thickness_constraint_loss(
                    thickness_at_x(yu_, yl_), MIN_THICKNESS_RATIO, MAX_THICKNESS_RATIO
                )

                # Simulation use the provided hard mask and wall distance (updated each iteration)
                base_state = grid.create_initial_state()

                u_init = jnp.ones((HEIGHT, WIDTH + 1), dtype=jnp.float32) * u_inlet
                v_init = jnp.ones((HEIGHT + 1, WIDTH), dtype=jnp.float32) * v_inlet
                u_init, v_init = apply_ibm_continuous_forcing(u_init, v_init, hard_mask)

                nu_init = jnp.where(
                    hard_mask > 0.5, jnp.float32(0.0), jnp.float32(5.0 * nu_lam)
                )
                nu_tilde_init = base_state.nu_tilde.with_values(nu_init)

                state = base_state.__class__(
                    **{
                        **base_state.__dict__,
                        "velocity": base_state.velocity.with_values(u_init, v_init),
                        "solid_mask": hard_mask,
                        "nu_tilde": nu_tilde_init,
                    }
                )

                final_state = jax.lax.fori_loop(
                    0, N_SIM_STEPS, lambda i, s: rans_step(s, wd), state
                )
                final_state = jax.lax.stop_gradient(final_state)
                nu_eff = jax.lax.stop_gradient(
                    grid.compute_effective_viscosity(final_state)
                )

                # Force calculation uses the CURRENT soft mask (sm_open) for gradients
                F_drag, F_lift = compute_aerodynamic_forces(
                    final_state, sm_open, nu_eff, RHO, aoa_rad
                )
                cl, cd = F_lift / (q_inf * ref_area), F_drag / (q_inf * ref_area)
                safe_cl = jnp.maximum(cl, 0.1)
                aero_loss = (
                    W_DRAG * (jnp.abs(cd) / safe_cl)
                    + W_LIFT * jnp.maximum(0.1 - cl, 0.0) * 10.0
                )
                return aero_loss + W_GEO * geo_loss, (cl, cd)

            val_and_grad = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))

            eff_b = None
            # --- Log Baseline Performance ---
            dat_path_base = f"tmp_base_re_{re_val}_aoa_{aoa_val}.dat"
            write_dat(wu, wl, dat_path_base)
            xf_base = run_xfoil(dat_path_base, re_val, aoa_val)
            if xf_base:
                cl_b, cd_b = xf_base
                eff_b = cl_b / max(cd_b, 1e-5)
                base_str = f"BASELINE RE={re_val}, AoA={aoa_val} | Cl={cl_b:.4f}, Cd={cd_b:.5f}, Cl/Cd={eff_b:.4f}\n"
                print("  " + base_str.strip())
                with open(OUTPUT_SUMMARY, "a") as f:
                    f.write(base_str)
            else:
                base_str = f"BASELINE RE={re_val}, AoA={aoa_val} | Xfoil failed\n"
                print("  " + base_str.strip())
                with open(OUTPUT_SUMMARY, "a") as f:
                    f.write(base_str)
            # --------------------------------

            m_adam = jnp.zeros_like(jnp.concatenate([wu, wl]))
            v_adam = jnp.zeros_like(jnp.concatenate([wu, wl]))
            beta1, beta2, eps_adam = 0.9, 0.999, 1e-8

            for i in range(NUM_ITERATIONS):
                # 1. Update Geometry (Hard Mask & Wall Distance) based on CURRENT CST weights
                # This ensures the RANS simulation and SA turbulence model remain accurate.
                current_sm = build_soft_mask(wu, wl, subcells)
                current_sm_open = current_sm.at[1:-1, -1].set(0.0)
                current_hard_mask = jnp.where(current_sm_open >= 0.5, 1.0, 0.0)
                current_wd = compute_wall_dist_np(current_hard_mask)

                # 2. Optimization Step
                (val, (cl_rans, cd_rans)), grads = val_and_grad(
                    jnp.concatenate([wu, wl]), current_hard_mask, current_wd
                )

                it_adam = i + 1
                m_adam = beta1 * m_adam + (1.0 - beta1) * grads
                v_adam = beta2 * v_adam + (1.0 - beta2) * grads**2
                m_hat = m_adam / (1.0 - beta1**it_adam)
                v_hat = v_adam / (1.0 - beta2**it_adam)

                update = LEARNING_RATE * m_hat / (jnp.sqrt(v_hat) + eps_adam)
                wu -= update[:6]
                wl -= update[6:]

                # 3. Validation & Tracking
                dat_path = f"tmp_re_{re_val}_aoa_{aoa_val}.dat"
                write_dat(wu, wl, dat_path)
                xf = run_xfoil(dat_path, re_val, aoa_val)
                if xf:
                    cl_x, cd_x = xf
                    eff = cl_x / max(cd_x, 1e-5)
                    if eff > best_cl_cd:
                        best_cl_cd, best_cl, best_cd, best_state = (
                            eff,
                            cl_x,
                            cd_x,
                            (jnp.copy(wu), jnp.copy(wl)),
                        )
                    print(
                        f"    Iter {i:2d}: Loss={float(val):.4f}, Cl_xf={cl_x:.4f}, Cd_xf={cd_x:.5f}, Eff_xf={eff:.2f}",
                        end="\r",
                    )
                else:
                    print(
                        f"    Iter {i:2d}: Loss={float(val):.4f}, Xfoil failed",
                        end="\r",
                    )

            final_airfoils[(re_val, aoa_val)] = best_state
            bw_u, bw_l = best_state
            x, y_u, y_l = generate_cst_coords(bw_u, bw_l, 200)
            t_max = float(max_thickness(y_u, y_l))

            if eff_b is not None and eff_b > 1e-5:
                imp = (best_cl_cd - eff_b) / eff_b * 100.0
                imp_str = f" | Imp={imp:+.1f}%"
            else:
                imp_str = ""

            res_str = f"OPT RE={re_val}, AoA={aoa_val} | Cl={best_cl:.4f}, Cd={best_cd:.5f}, Cl/Cd={best_cl_cd:.4f}{imp_str} | Max Thick={t_max:.4f}\n"
            print("\n  " + res_str.strip())
            with open(OUTPUT_SUMMARY, "a") as f:
                f.write(res_str)
            write_dat(bw_u, bw_l, f"best_airfoil_re_{re_val}_aoa_{aoa_val}.dat")
        plot_re(re_val, final_airfoils)


def plot_re(re_val, final_airfoils):
    plt.figure(figsize=(10, 4))
    # Initial
    x, yu, yl = generate_cst_coords(NACA0012_W_U, NACA0012_W_L, 200)
    plt.plot(x, yu, "k--", alpha=0.3, label="NACA 0012")
    plt.plot(x, yl, "k--", alpha=0.3)

    colors = ["#E74C3C", "#2ECC71", "#3498DB"]
    for i, aoa in enumerate(AOA_LIST):
        if (re_val, aoa) in final_airfoils:
            wu, wl = final_airfoils[(re_val, aoa)]
            x, yu, yl = generate_cst_coords(wu, wl, 200)
            plt.plot(x, yu, color=colors[i], label=f"Opt AoA {aoa}", linewidth=2)
            plt.plot(x, yl, color=colors[i], linewidth=2)

    plt.title(f"Optimized Shapes at RE = {re_val}")
    plt.xlabel("x/c")
    plt.ylabel("y/c")
    plt.legend()
    plt.axis("equal")
    plt.savefig(f"shapes_re_{re_val}.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    run_suite()
