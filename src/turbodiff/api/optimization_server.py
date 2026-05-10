"""FastAPI server for streaming airfoil optimization results.

Runs gradient-based airfoil shape optimization and streams per-iteration
results (shape, CL, CD, CL/CD, drag, loss) over WebSocket.
"""

from __future__ import annotations

import asyncio
import math
import os
import logging
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple
from uuid import uuid4

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.special import comb as sp_comb

import jax
import jax.numpy as jnp
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, Depends
from pydantic import BaseModel, Field
from turbodiff.api.auth import get_current_user, verify_websocket_token

from turbodiff.core.airfoil import generate_cst_coords, thickness_at_x
from turbodiff.core.loss_functions import (
    crossover_validity_loss,
    thickness_constraint_loss,
)
from turbodiff.core.optimization import create_optimizer
from turbodiff.core.fluid_grid_jax import FluidGrid
from turbodiff.core.utils import apply_ibm_continuous_forcing
from turbodiff.xfoil import run_xfoil, write_dat_file
from turbodiff.db.storage import (
    OptimizedAirfoilPayload,
    SessionCreatePayload,
    get_storage_repository,
)


# ---------------------------------------------------------------------------
# Fidelity presets (same as streaming_server)
# ---------------------------------------------------------------------------
# Keys match the values sent by the frontend after meshDensity → fidelity mapping.
FIDELITY_MAP: Dict[str, Tuple[int, int]] = {
    "low": (64, 192),  # Coarse, preserves suite-style 3:1 domain ratio
    "medium": (128, 384),  # Matches optimize_naca_suite.py baseline domain
    "high": (256, 768),  # Fine
    "ultra": (512, 1536),  # Ultra
}

# Default cell size (metres) for each fidelity level.
# Keeps the airfoil chord at high resolution (suite-like at medium and above).
CELL_SIZE_MAP: Dict[str, float] = {
    "low": 0.02,
    "medium": 0.01,
    "high": 0.005,
    "ultra": 0.0025,
}

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/optimize", tags=["optimization"])

_OPT_SESSIONS: Dict[str, "OptSessionConfig"] = {}
_OPT_RESULTS: Dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Request / config models
# ---------------------------------------------------------------------------


class OptSessionRequest(BaseModel):
    """Parameters to create an optimization session."""

    user_id: str | None = Field(None, description="User identifier")

    fidelity: str = Field("low", description="low | medium | high | ultra")

    # Initial CST weights
    cst_upper: List[float] = Field(
        default=[0.1726, 0.1269, 0.1441, 0.1080, 0.1750, 0.1471],
        description="Initial upper-surface CST weights",
    )
    cst_lower: List[float] = Field(
        default=[-0.1726, -0.1269, -0.1441, -0.1080, -0.1750, -0.1471],
        description="Initial lower-surface CST weights",
    )

    # Simulation physics — cell_size defaults to None so the server can derive
    # the appropriate value from the fidelity level automatically.
    cell_size: float | None = Field(
        None,
        gt=0.0,
        description="Cell size in metres; auto-derived from fidelity if omitted",
    )
    dt: float = Field(0.0002, gt=0.0)
    diffusion: float = Field(0.001, ge=0.0)
    inflow_velocity: float = Field(1.0, ge=0.0)
    num_sim_steps: int = Field(1000, ge=1, description="Sim steps per iteration")

    # Airfoil placement
    chord_length: float | None = Field(
        None, description="Chord length (cells × cell_size)"
    )
    airfoil_offset_x: float | None = Field(None)
    airfoil_offset_y: float | None = Field(None)

    # Optimization hyper-parameters
    num_iterations: int = Field(30, ge=1)
    learning_rate: float = Field(0.005, gt=0.0)
    optimizer: str = Field("adam", description="adam | sgd")
    grad_clip: float = Field(0.1, gt=0.0)

    # Geometric constraints
    min_thickness: float = Field(0.10, ge=0.0)
    max_thickness: float = Field(0.25, gt=0.0)

    # Objective weights (strictly matching optimize_naca_suite.py)
    w_lift: float = Field(0.5, ge=0.0)
    w_drag: float = Field(1.0, ge=0.0)
    w_ratio: float = Field(10.0, ge=0.0)

    # CST generation
    num_cst_points: int = Field(200, ge=10)
    mask_sharpness: float = Field(500.0, gt=0.0)

    # Angle of attack
    angle_of_attack: float = Field(
        0.0, description="Degrees; rotates airfoil, wind stays horizontal"
    )

    # Reynolds number — when provided, derives viscosity: ν = inflow_velocity × chord / Re
    reynolds_number: float | None = Field(
        None,
        gt=0.0,
        description="Reynolds number (e.g. 1e6). Overrides diffusion-based viscosity.",
    )

    # Streaming
    stream_fps: float = Field(0.0, ge=0.0, description="0 = as fast as possible")


@dataclass(frozen=True)
class OptSessionConfig:
    session_id: str
    user_id: str
    height: int
    width: int
    cst_upper: List[float]
    cst_lower: List[float]
    cell_size: float
    dt: float
    diffusion: float
    inflow_velocity: float
    num_sim_steps: int
    chord_length: float
    airfoil_offset_x: float
    airfoil_offset_y: float
    num_iterations: int
    learning_rate: float
    optimizer: str
    grad_clip: float
    min_thickness: float
    max_thickness: float
    w_lift: float
    w_drag: float
    w_ratio: float
    num_cst_points: int
    mask_sharpness: float
    angle_of_attack: float
    reynolds_number: float | None
    stream_fps: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_fidelity(fidelity: str) -> Tuple[int, int]:
    key = fidelity.lower()
    if key not in FIDELITY_MAP:
        opts = ", ".join(sorted(FIDELITY_MAP.keys()))
        raise ValueError(f"Invalid fidelity='{fidelity}'. Valid options: {opts}")
    return FIDELITY_MAP[key]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("/sessions")
def create_opt_session(
    request: OptSessionRequest, user: dict = Depends(get_current_user)
):
    """Create a new optimization session and return its ID."""
    user_id = user.get("uid")

    try:
        height, width = _get_fidelity(request.fidelity)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Resolve cell_size: honor explicit caller value, otherwise use fidelity default.
    fidelity_key = request.fidelity.lower()
    cell_size = request.cell_size or CELL_SIZE_MAP[fidelity_key]

    # Default chord length: 1.0 m (within 0.5–2 m range).
    # Users can override via request.chord_length.
    chord_length = request.chord_length or 1.0
    airfoil_offset_x = request.airfoil_offset_x or (30 * cell_size)
    airfoil_offset_y = request.airfoil_offset_y or (height // 2 * cell_size)

    # Resolve viscosity from Reynolds number when provided.
    # ν = U_inf × chord / Re  (matches optimize_naca_suite.py convention)
    resolved_diffusion = request.diffusion
    if request.reynolds_number and request.reynolds_number > 0:
        resolved_diffusion = (
            request.inflow_velocity * chord_length
        ) / request.reynolds_number

    session_id = str(uuid4())
    config = OptSessionConfig(
        session_id=session_id,
        user_id=user_id,
        height=height,
        width=width,
        cst_upper=request.cst_upper,
        cst_lower=request.cst_lower,
        cell_size=cell_size,
        dt=request.dt,
        diffusion=resolved_diffusion,
        inflow_velocity=request.inflow_velocity,
        num_sim_steps=request.num_sim_steps,
        chord_length=chord_length,
        airfoil_offset_x=airfoil_offset_x,
        airfoil_offset_y=airfoil_offset_y,
        num_iterations=request.num_iterations,
        learning_rate=request.learning_rate,
        optimizer=request.optimizer,
        grad_clip=request.grad_clip,
        min_thickness=request.min_thickness,
        max_thickness=request.max_thickness,
        w_lift=request.w_lift,
        w_drag=request.w_drag,
        w_ratio=request.w_ratio,
        num_cst_points=request.num_cst_points,
        mask_sharpness=request.mask_sharpness,
        angle_of_attack=request.angle_of_attack,
        reynolds_number=request.reynolds_number,
        stream_fps=request.stream_fps,
    )
    _OPT_SESSIONS[session_id] = config

    repo = get_storage_repository()
    parameters = {
        "request": request.dict(),
        "resolved": {
            "height": height,
            "width": width,
            "cell_size": cell_size,
            "chord_length": chord_length,
            "airfoil_offset_x": airfoil_offset_x,
            "airfoil_offset_y": airfoil_offset_y,
            "diffusion": resolved_diffusion,
            "reynolds_number": request.reynolds_number,
        },
    }
    try:
        storage = repo.create_session_with_airfoil(
            SessionCreatePayload(
                session_id=session_id,
                user_id=user_id,
                session_type="optimize",
                parameters=parameters,
                cst_weights_upper=request.cst_upper,
                cst_weights_lower=request.cst_lower,
                chord_length=chord_length,
                angle_of_attack=request.angle_of_attack,
            )
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "session_id": session_id,
        "config": asdict(config),
        "storage": {
            "airfoil_id": storage.airfoil_id,
            "cst_id": storage.cst_id,
        },
    }


def _build_optimization_fns(config: OptSessionConfig):
    """Build the loss/grad functions (pure, no async). Returns closures.

    Uses the full Spalart-Allmaras RANS pipeline (matching optimize_airfoil_rans.py):
      1. Build differentiable soft mask from CST weights
      2. Threshold → hard mask → wall distance (stop_gradient)
      3. Run N_SIM_STEPS of RANS (fori_loop, stop_gradient on final state)
      4. Compute pressure + viscous forces using the soft mask (gradients flow here)
    """

    n_weights = len(config.cst_upper)
    nu_lam = (
        config.diffusion
    )  # kinematic viscosity (derived from Re on session creation)
    u_inlet = float(
        config.inflow_velocity * math.cos(math.radians(config.angle_of_attack))
    )
    v_inlet = float(
        config.inflow_velocity * math.sin(math.radians(config.angle_of_attack))
    )
    aoa_rad = jnp.float32(math.radians(config.angle_of_attack))
    q_inf = 0.5 * 1.0 * config.inflow_velocity**2  # rho=1

    # Precompute sub-cell geometry for smooth mask gradients (suite-style).
    n_order = n_weights - 1
    n_sub = 3
    i_vals = np.arange(n_order + 1)
    binom = np.array(
        [sp_comb(n_order, i, exact=True) for i in i_vals], dtype=np.float32
    )

    subcells = []
    jj, ii = np.meshgrid(np.arange(config.width), np.arange(config.height))
    for si in range(n_sub):
        for sj in range(n_sub):
            cx = (jj + (sj + 0.5) / n_sub) * config.cell_size
            cy = (ii + (si + 0.5) / n_sub) * config.cell_size
            cx_rel = (cx - config.airfoil_offset_x).astype(np.float32)
            cy_rel = (cy - config.airfoil_offset_y).astype(np.float32)
            xi_grid = cx_rel / config.chord_length
            xi_c = np.clip(xi_grid, 0.0, 1.0).astype(np.float32)
            C_g = (np.sqrt(np.maximum(xi_c, 0.0)) * (1.0 - xi_c)).astype(np.float32)
            xi_c_col = xi_c[..., None]
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

    # Build the RANS grid once (cached across iterations)
    rans_grid = FluidGrid(
        height=config.height,
        width=config.width,
        cell_size=config.cell_size,
        dt=config.dt,
        diffusion=0.0,
        viscosity=nu_lam,
        boundary_type=0,
        use_sa_turbulence=True,
        visualise=False,
    )
    rans_grid.use_sa_turbulence = True

    # Free-stream BC helper (applied after each RANS step)
    @jax.jit
    def apply_freestream_bc(state):
        u = state.velocity.u
        v = state.velocity.v
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
    def rans_step(state, wall_dist):
        state = rans_grid.step_sa_turbulence(state, wall_dist, num_diff_iters=4)
        nu_eff = rans_grid.compute_effective_viscosity(state)
        state = rans_grid.diffuse_velocity(state, num_iters=20, nu_eff_field=nu_eff)
        state = rans_grid.advect_velocity(state)
        state = apply_freestream_bc(state)
        u, v = apply_ibm_continuous_forcing(
            state.velocity.u, state.velocity.v, state.solid_mask
        )
        state = state.__class__(
            **{**state.__dict__, "velocity": state.velocity.with_values(u, v)}
        )
        state = rans_grid.solve_pressure(state, num_iters=60)
        state = rans_grid.project_velocity(state)
        state = apply_freestream_bc(state)
        return state.__class__(
            **{
                **state.__dict__,
                "time": state.time + rans_grid.dt,
                "step": state.step + 1,
            }
        )

    def _create_mask(weights_upper, weights_lower):
        sharpness = config.mask_sharpness
        accum = jnp.zeros((config.height, config.width), dtype=jnp.float32)
        for sc in subcells:
            y_u = sc["C_g"] * (sc["B_g"] @ weights_upper)
            y_l = sc["C_g"] * (sc["B_g"] @ weights_lower)
            above = jax.nn.sigmoid(sharpness * (sc["cy_body"] - y_l))
            below = jax.nn.sigmoid(sharpness * (y_u - sc["cy_body"]))
            in_chord = jax.nn.sigmoid(sharpness * sc["xi_grid"]) * jax.nn.sigmoid(
                sharpness * (1.0 - sc["xi_grid"])
            )
            accum = accum + above * below * in_chord
        return accum / float(n_sub * n_sub)

    def _wall_dist_from_mask(hard_mask: jnp.ndarray) -> jnp.ndarray:
        """Compute wall distance via scipy EDT (not differentiated)."""
        solid_np = np.array(hard_mask)
        solid_np[1:-1, -1] = 0.0  # open outflow
        wd_np = distance_transform_edt(solid_np < 0.5) * config.cell_size
        return jax.lax.stop_gradient(
            jnp.array(np.maximum(wd_np.astype(np.float32), 1e-10))
        )

    def _compute_forces(state, soft_mask_open, nu_eff):
        """Pressure + viscous forces on the airfoil (differentiable)."""
        u, v, p = state.velocity.u, state.velocity.v, state.pressure.values
        h = config.cell_size
        mu = nu_eff  # rho=1
        fluid = 1.0 - soft_mask_open
        is_right = fluid[:, :-1] * soft_mask_open[:, 1:]
        is_left = fluid[:, 1:] * soft_mask_open[:, :-1]
        is_bot = fluid[:-1, :] * soft_mask_open[1:, :]
        is_top = fluid[1:, :] * soft_mask_open[:-1, :]
        Fpx = jnp.sum(p[:, :-1] * is_right) * h - jnp.sum(p[:, 1:] * is_left) * h
        Fpy = jnp.sum(p[:-1, :] * is_bot) * h - jnp.sum(p[1:, :] * is_top) * h
        u_cc = 0.5 * (u[:, :-1] + u[:, 1:])
        v_cc = 0.5 * (v[:-1, :] + v[1:, :])
        dudy = jnp.zeros_like(p).at[1:-1, :].set((u_cc[2:, :] - u_cc[:-2, :]) / (2 * h))
        dvdx = jnp.zeros_like(p).at[:, 1:-1].set((v_cc[:, 2:] - v_cc[:, :-2]) / (2 * h))
        dudx = jnp.zeros_like(p).at[:, 1:-1].set((u_cc[:, 2:] - u_cc[:, :-2]) / (2 * h))
        dvdy = jnp.zeros_like(p).at[1:-1, :].set((v_cc[2:, :] - v_cc[:-2, :]) / (2 * h))
        tau_xy = mu * (dudy + dvdx)
        tau_xx = 2.0 * mu * dudx
        tau_yy = 2.0 * mu * dvdy
        Fvx = (
            jnp.sum(tau_xx[:, :-1] * is_right) * h
            - jnp.sum(tau_xx[:, 1:] * is_left) * h
            + jnp.sum(tau_xy[:-1, :] * is_bot) * h
            - jnp.sum(tau_xy[1:, :] * is_top) * h
        )
        Fvy = (
            jnp.sum(tau_xy[:, :-1] * is_right) * h
            - jnp.sum(tau_xy[:, 1:] * is_left) * h
            + jnp.sum(tau_yy[:-1, :] * is_bot) * h
            - jnp.sum(tau_yy[1:, :] * is_top) * h
        )
        Fx, Fy = Fpx + Fvx, Fpy + Fvy
        cos_a2, sin_a2 = jnp.cos(aoa_rad), jnp.sin(aoa_rad)
        F_drag = Fx * cos_a2 + Fy * sin_a2
        F_lift = -Fx * sin_a2 + Fy * cos_a2
        return F_drag, F_lift

    def compute_loss_with_aux(params, hard_mask, wd):
        """Loss function matching optimize_naca_suite.py exactly.

        hard_mask and wd are computed OUTSIDE this function each iteration
        (via scipy EDT, which cannot run inside JAX's trace).
        """
        weights_upper = params[:n_weights]
        weights_lower = params[n_weights:]

        # ── 1. Geometric constraints ───────────────────────────────────────────
        x_cst, y_upper, y_lower = generate_cst_coords(weights_upper, weights_lower)
        thickness = thickness_at_x(y_upper, y_lower)
        geo_loss = crossover_validity_loss(
            y_upper, y_lower
        ) + thickness_constraint_loss(
            thickness, config.min_thickness, config.max_thickness
        )

        # ── 2. Soft mask (differentiable, for force gradients) ────────────────
        soft_mask = _create_mask(weights_upper, weights_lower)
        soft_mask_open = soft_mask.at[1:-1, -1].set(0.0)

        # ── 3. RANS initial state (uses hard_mask from caller) ───────────────
        rans_grid.solid_mask = hard_mask
        rans_grid._wall_dist = wd

        u_init = (
            jnp.ones((config.height, config.width + 1), dtype=jnp.float32) * u_inlet
        )
        v_init = (
            jnp.ones((config.height + 1, config.width), dtype=jnp.float32) * v_inlet
        )
        u_init, v_init = apply_ibm_continuous_forcing(u_init, v_init, hard_mask)

        base_state = rans_grid.create_initial_state()  # seeds nu_tilde at 5ν in fluid
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

        # ── 4. RANS time integration ────────────────────────────────────────
        final_state = jax.lax.fori_loop(
            0, config.num_sim_steps, lambda i, s: rans_step(s, wd), state
        )
        final_state = jax.lax.stop_gradient(final_state)

        # ── 5. Aerodynamic forces (differentiable via soft mask) ──────────────
        nu_eff = jax.lax.stop_gradient(
            rans_grid.compute_effective_viscosity(final_state)
        )
        F_drag, F_lift = _compute_forces(final_state, soft_mask_open, nu_eff)

        C_L = F_lift / (q_inf * config.chord_length)
        C_D = F_drag / (q_inf * config.chord_length)

        # ── 6. Composite loss (strictly matching optimize_naca_suite.py) ──────
        safe_cl = jnp.maximum(C_L, jnp.float32(0.1))
        aero_loss = (
            config.w_drag * (jnp.abs(C_D) / safe_cl)
            + config.w_lift * jnp.maximum(0.1 - C_L, jnp.float32(0.0)) * 10.0
        )
        total_loss = aero_loss + config.w_ratio * geo_loss

        return total_loss, (C_L, C_D, F_lift, F_drag, geo_loss)

    def loss_only(params, hard_mask, wd):
        loss, _ = compute_loss_with_aux(params, hard_mask, wd)
        return loss

    # jax.grad differentiates w.r.t. first arg (params) only
    grad_fn = jax.grad(loss_only)

    def compute_current_geometry(params_np):
        """Return (hard_mask, wall_dist) for current params — runs OUTSIDE JAX trace."""
        wu = jnp.array(params_np[:n_weights])
        wl = jnp.array(params_np[n_weights:])
        soft = _create_mask(wu, wl)
        soft_open = soft.at[1:-1, -1].set(0.0)
        hard = jnp.where(soft_open >= 0.5, jnp.float32(1.0), jnp.float32(0.0))
        wd = _wall_dist_from_mask(hard)
        return hard, wd

    return n_weights, compute_loss_with_aux, grad_fn, compute_current_geometry


def _run_iteration(params, hard_mask, wd, compute_loss_with_aux, grad_fn, config):
    """Run one optimization iteration (CPU-heavy, called in thread).

    Matches optimize_naca_suite.py: hard_mask and wd are computed by the
    caller BEFORE this function, so scipy EDT never runs inside JAX's trace.
    """
    loss_val, (C_L, C_D, lift_force, drag_force, geo_loss) = compute_loss_with_aux(
        params, hard_mask, wd
    )
    gradients = grad_fn(params, hard_mask, wd)

    has_nan = bool(jnp.isnan(loss_val) or jnp.any(jnp.isnan(gradients)))

    return loss_val, C_L, C_D, lift_force, drag_force, geo_loss, gradients, has_nan


@router.websocket("/ws/{session_id}")
async def stream_optimization(ws: WebSocket, session_id: str):
    """Run airfoil optimization and stream each iteration's results."""
    await ws.accept()

    user = await verify_websocket_token(ws)
    if user is None:
        await ws.send_json({"error": "Missing or invalid authentication token"})
        await ws.close(code=1008)
        return

    config = _OPT_SESSIONS.get(session_id)
    if config is None:
        await ws.send_json({"error": "unknown session_id"})
        await ws.close(code=1008)
        return

    # Build functions (fast — no actual computation yet)
    n_weights, compute_loss_with_aux, grad_fn, compute_current_geometry = (
        _build_optimization_fns(config)
    )

    # Initial parameters
    params = jnp.concatenate(
        [
            jnp.array(config.cst_upper),
            jnp.array(config.cst_lower),
        ]
    )

    # Optimizer
    opt_state, update_fn = create_optimizer(
        config.optimizer, learning_rate=config.learning_rate
    )

    print(
        f"[optimize] Session {config.session_id}: starting {config.num_iterations} iterations"
    )

    # Best-shape tracking (mirrors optimize_naca_suite.py)
    best_cl_cd = -1.0
    best_params = params

    last_cl = None
    last_cd = None
    last_lift = None
    last_drag = None
    best_cl = None
    best_cd = None
    best_lift = None
    best_drag = None
    best_max_thickness = None

    try:
        for iteration in range(config.num_iterations):
            # 1. Compute geometry (hard mask + wall dist) OUTSIDE loss/grad
            #    Mirrors suite lines 428-431: scipy EDT must not be inside JAX trace.
            hard_mask, wd = await asyncio.to_thread(compute_current_geometry, params)

            # 2. Offload heavy JAX compute to thread pool
            loss_val, C_L, C_D, lift_force, drag_force, geo_loss, gradients, has_nan = (
                await asyncio.to_thread(
                    _run_iteration,
                    params,
                    hard_mask,
                    wd,
                    compute_loss_with_aux,
                    grad_fn,
                    config,
                )
            )

            if has_nan:
                await ws.send_json(
                    {
                        "type": "warning",
                        "iteration": iteration + 1,
                        "message": "NaN detected, skipping update",
                    }
                )
                continue

            # 3. Update params first, then validate the updated shape (suite behavior).
            params, opt_state = update_fn(params, gradients, opt_state)

            # --- extract updated shape for FE and validation ---
            cur_upper = params[:n_weights]
            cur_lower = params[n_weights:]
            x_cst, y_upper_cst, y_lower_cst = generate_cst_coords(
                cur_upper, cur_lower, num_points=config.num_cst_points
            )

            # 4. Validation & Tracking (mirrors optimize_naca_suite.py)
            # Resolve re_val from config, falling back to simulation-derived Re
            re_val = config.reynolds_number or (
                (config.inflow_velocity * config.chord_length) / config.diffusion
            )
            aoa_val = config.angle_of_attack

            # Use the Airfoils/ directory for temporary files
            project_root = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..", "..")
            )
            _AIRFOILS_DIR = os.path.join(project_root, "Airfoils")
            os.makedirs(_AIRFOILS_DIR, exist_ok=True)

            dat_path = os.path.join(
                _AIRFOILS_DIR, f"tmp_opt_{config.session_id}_{iteration}.dat"
            )
            polar_path = dat_path.replace(".dat", ".txt")

            try:
                # Use thread to not block event loop when writing/running subprocess
                await asyncio.to_thread(
                    write_dat_file, cur_upper, cur_lower, dat_path, num_points=200
                )
                logger.info(
                    "Running XFoil for session %s, iteration %d, Re=%f, AoA=%f",
                    config.session_id,
                    iteration + 1,
                    re_val,
                    aoa_val,
                )
                xf = await asyncio.to_thread(run_xfoil, dat_path, re_val, aoa_val)
                if xf:
                    logger.info(
                        "XFoil for session %s, iteration %d completed: Cl=%f, Cd=%f, Eff=%f",
                        config.session_id,
                        iteration + 1,
                        xf[0],
                        xf[1],
                        xf[0] / max(xf[1], 1e-5),
                    )
                else:
                    logger.warning(
                        "XFoil did not converge for session %s, iteration %d (Re=%f, AoA=%f)",
                        config.session_id,
                        iteration + 1,
                        re_val,
                        aoa_val,
                    )
            finally:
                # Clean up the dat and txt files immediately
                for path in [dat_path, polar_path]:
                    if os.path.exists(path):
                        try:
                            os.remove(path)
                            # print("NHI KRRHA DELETE!!")
                        except Exception:
                            pass

            # ── Tracking & display (mirrors optimize_naca_suite.py exactly) ──────
            # Best shape is ONLY promoted when XFoil converges (suite lines 452-461).
            # A failed XFoil call just supplies display values for this iteration.
            if xf:
                cl_x, cd_x = xf
                eff = cl_x / max(cd_x, 1e-5)
                # Update best only on XFoil success (suite behavior)
                if eff > best_cl_cd:
                    best_cl_cd = eff
                    best_params = jnp.array(params)
                    best_cl = float(cl_x)
                    best_cd = float(cd_x)
                    best_lift = float(lift_force)
                    best_drag = float(drag_force)
                    best_max_thickness = float(jnp.max(y_upper_cst - y_lower_cst))

                display_cl = float(cl_x)
                display_cd = float(cd_x)
                cl_cd = eff
                print(
                    f"    Iter {iteration+1:2d}: Loss={float(loss_val):.4f}, Cl_xf={cl_x:.4f}, Cd_xf={cd_x:.5f}, Eff_xf={eff:.2f}"
                )
                logger.info(
                    f"Session {config.session_id} Iter {iteration+1}: XFoil converged Cl={cl_x:.4f}, Cd={cd_x:.5f}, Eff={eff:.2f}"
                )
            else:
                # XFoil failed — do NOT update best; send null for cl, cd, cl_cd
                # so the frontend only plots XFoil-verified points.
                display_cl = None
                display_cd = None
                cl_cd = None

                print(
                    f"    Iter {iteration+1:2d}: Loss={float(loss_val):.4f}, Xfoil failed — no results recorded"
                )
                logger.warning(
                    f"Session {config.session_id} Iter {iteration+1}: XFoil failed (Re={re_val}, AoA={aoa_val}); best not updated."
                )

            last_lift = float(lift_force)
            last_drag = float(drag_force)
            last_cl = display_cl
            last_cd = display_cd

            payload = {
                "type": "iteration",
                "meta": {
                    "iteration": iteration + 1,
                    "total_iterations": config.num_iterations,
                    "loss": float(loss_val),
                    "cl": display_cl,
                    "cd": display_cd,
                    "cl_cd": cl_cd,
                    "lift_force": last_lift,
                    "drag_force": last_drag,
                    "max_thickness": float(jnp.max(y_upper_cst - y_lower_cst)),
                },
                "shape": {
                    "cst_upper": cur_upper.tolist(),
                    "cst_lower": cur_lower.tolist(),
                    "airfoil_x": x_cst.tolist(),
                    "airfoil_y_upper": y_upper_cst.tolist(),
                    "airfoil_y_lower": y_lower_cst.tolist(),
                },
            }

            await ws.send_json(payload)

            # Yield control so the WS frame can be sent
            if config.stream_fps > 0:
                await asyncio.sleep(1.0 / config.stream_fps)
            else:
                await asyncio.sleep(0)

        # ----- send final summary (use best L/D shape, mirrors suite's best_state) -----
        final_upper = best_params[:n_weights]
        final_lower = best_params[n_weights:]
        x_final, y_upper_final, y_lower_final = generate_cst_coords(
            final_upper, final_lower, num_points=config.num_cst_points
        )

        final_payload = {
            "type": "complete",
            "meta": {
                "total_iterations": config.num_iterations,
                "final_cl": (
                    best_cl
                    if best_cl is not None
                    else (last_cl if last_cl is not None else float(C_L))
                ),
                "final_cd": (
                    best_cd
                    if best_cd is not None
                    else (last_cd if last_cd is not None else float(C_D))
                ),
                "final_cl_cd": best_cl_cd,
                "final_drag": best_drag if best_drag is not None else float(drag_force),
                "final_loss": float(loss_val),
                "final_max_thickness": (
                    best_max_thickness
                    if best_max_thickness is not None
                    else float(jnp.max(y_upper_final - y_lower_final))
                ),
            },
            "shape": {
                "cst_upper": final_upper.tolist(),
                "cst_lower": final_lower.tolist(),
                "airfoil_x": x_final.tolist(),
                "airfoil_y_upper": y_upper_final.tolist(),
                "airfoil_y_lower": y_lower_final.tolist(),
            },
            "initial_shape": {
                "cst_upper": config.cst_upper,
                "cst_lower": config.cst_lower,
            },
        }
        await ws.send_json(final_payload)

        _OPT_RESULTS[session_id] = final_payload

        print(f"[optimize] Session {config.session_id}: optimization complete")
        await ws.close()

    except WebSocketDisconnect:
        print(f"[optimize] Session {config.session_id}: client disconnected")
    finally:
        if best_cl is not None and best_cd is not None:
            best_upper = best_params[:n_weights].tolist()
            best_lower = best_params[n_weights:].tolist()
            print(
                f"Saving final optimized airfoil for session {config.session_id}: cl={best_cl:.4f}, cd={best_cd:.4f}"
            )
            try:
                repo = get_storage_repository()
                repo.save_optimized_airfoil(
                    OptimizedAirfoilPayload(
                        session_id=session_id,
                        user_id=config.user_id,
                        cst_weights_upper=best_upper,
                        cst_weights_lower=best_lower,
                        chord_length=config.chord_length,
                        angle_of_attack=config.angle_of_attack,
                        cl=best_cl,
                        cd=best_cd,
                        lift=best_lift if best_lift is not None else last_lift,
                        drag=best_drag if best_drag is not None else last_drag,
                    )
                )
            except Exception as e:
                print(
                    f"Failed to auto-save optimization metrics for session {session_id}: {e}"
                )


@router.get("/sessions/{session_id}/result")
def get_optimization_result(session_id: str, user: dict = Depends(get_current_user)):
    """Get the optimization result from cache or db."""
    user_id = user.get("uid")

    if session_id in _OPT_RESULTS:
        # We could check ownership here if cache has user_id, but skipping for simplicity
        # or checking the config:
        config = _OPT_SESSIONS.get(session_id)
        if config and config.user_id != user_id:
            raise HTTPException(
                status_code=403, detail="Not authorized to access this session"
            )
        return _OPT_RESULTS[session_id]

    repo = get_storage_repository()

    # Verify this is an optimization session
    session_record = repo.get_session(session_id)
    if not session_record:
        raise HTTPException(status_code=404, detail="Session not found")

    if session_record.session_type != "optimize":
        raise HTTPException(
            status_code=400, detail="This session is not an optimization session"
        )

    airfoil = repo.get_latest_airfoil(session_id, is_optimized=True)
    if not airfoil:
        raise HTTPException(
            status_code=404, detail="Optimization result not found for this session"
        )

    if str(airfoil.created_by_user_id) != str(user_id):
        raise HTTPException(
            status_code=403, detail="Not authorized to access this session"
        )

    cst = repo.get_cst(airfoil.cst_id)
    if not cst:
        raise HTTPException(status_code=404, detail="CST data not found")

    final_upper = jnp.array(cst.weights_upper)
    final_lower = jnp.array(cst.weights_lower)

    num_cst_points = 200
    config = _OPT_SESSIONS.get(session_id)
    if config:
        num_cst_points = config.num_cst_points
        initial_upper = config.cst_upper
        initial_lower = config.cst_lower
        total_iterations = config.num_iterations
    else:
        initial_upper = cst.weights_upper
        initial_lower = cst.weights_lower
        total_iterations = 0

    x_final, y_upper_final, y_lower_final = generate_cst_coords(
        final_upper, final_lower, num_points=num_cst_points
    )

    cl_cd = airfoil.cl / airfoil.cd if airfoil.cd and abs(airfoil.cd) > 1e-12 else 0.0

    final_payload = {
        "type": "complete",
        "meta": {
            "total_iterations": total_iterations,
            "final_cl": airfoil.cl,
            "final_cd": airfoil.cd,
            "final_cl_cd": cl_cd,
            "final_drag": airfoil.drag,
            "final_loss": 0.0,
        },
        "shape": {
            "cst_upper": cst.weights_upper,
            "cst_lower": cst.weights_lower,
            "airfoil_x": x_final.tolist(),
            "airfoil_y_upper": y_upper_final.tolist(),
            "airfoil_y_lower": y_lower_final.tolist(),
        },
        "initial_shape": {
            "cst_upper": initial_upper,
            "cst_lower": initial_lower,
        },
    }

    _OPT_RESULTS[session_id] = final_payload
    return final_payload
