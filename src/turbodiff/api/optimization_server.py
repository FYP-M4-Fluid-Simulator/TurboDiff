"""FastAPI server for streaming airfoil optimization results.

Runs gradient-based airfoil shape optimization and streams per-iteration
results (shape, CL, CD, CL/CD, drag, loss) over WebSocket.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
from itertools import count
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from turbodiff.core.airfoil import generate_cst_coords, thickness_at_x
from turbodiff.core.airfoil_optimization import (
    compute_grid_coordinates,
    create_airfoil_solid_mask,
)
from turbodiff.core.loss_functions import (
    crossover_validity_loss,
    thickness_constraint_loss,
)
from turbodiff.core.optimization import create_optimizer
from turbodiff.core.fluid_grid_jax import FluidGrid, FluidState


# ---------------------------------------------------------------------------
# Fidelity presets (same as streaming_server)
# ---------------------------------------------------------------------------
FIDELITY_MAP: Dict[str, Tuple[int, int]] = {
    "low": (64, 128),
    "medium": (128, 256),
    "coarse": (256, 512),
}

router = APIRouter(prefix="/optimize", tags=["optimization"])

_OPT_SESSION_ID = count(1)
_OPT_SESSIONS: Dict[str, "OptSessionConfig"] = {}


# ---------------------------------------------------------------------------
# Request / config models
# ---------------------------------------------------------------------------

class OptSessionRequest(BaseModel):
    """Parameters to create an optimization session."""

    fidelity: str = Field("low", description="low | medium | coarse")

    # Initial CST weights
    cst_upper: List[float] = Field(
        default=[0.18, 0.22, 0.20, 0.18, 0.15, 0.12],
        description="Initial upper-surface CST weights",
    )
    cst_lower: List[float] = Field(
        default=[-0.10, -0.08, -0.06, -0.05, -0.04, -0.03],
        description="Initial lower-surface CST weights",
    )

    # Simulation physics
    cell_size: float = Field(0.1, gt=0.0)
    dt: float = Field(0.05, gt=0.0)
    diffusion: float = Field(0.001, ge=0.0)
    inflow_velocity: float = Field(1.0, ge=0.0)
    num_sim_steps: int = Field(80, ge=1, description="Sim steps per iteration")

    # Airfoil placement
    chord_length: float | None = Field(None, description="Chord length (cells × cell_size)")
    airfoil_offset_x: float | None = Field(None)
    airfoil_offset_y: float | None = Field(None)

    # Optimization hyper-parameters
    num_iterations: int = Field(30, ge=1)
    learning_rate: float = Field(0.005, gt=0.0)
    optimizer: str = Field("adam", description="adam | sgd")
    grad_clip: float = Field(0.1, gt=0.0)

    # Geometric constraints
    min_thickness: float = Field(0.06, ge=0.0)
    max_thickness: float = Field(0.25, gt=0.0)

    # Objective weights
    w_lift: float = Field(1000.0, ge=0.0)
    w_drag: float = Field(100.0, ge=0.0)
    w_ratio: float = Field(100.0, ge=0.0)

    # CST generation
    num_cst_points: int = Field(100, ge=10)
    mask_sharpness: float = Field(50.0, gt=0.0)

    # Streaming
    stream_fps: float = Field(0.0, ge=0.0, description="0 = as fast as possible")


@dataclass(frozen=True)
class OptSessionConfig:
    session_id: str
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
    stream_fps: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_fidelity(fidelity: str) -> Tuple[int, int]:
    key = fidelity.lower()
    if key not in FIDELITY_MAP:
        opts = ", ".join(sorted(FIDELITY_MAP.keys()))
        raise ValueError(f"Invalid fidelity={fidelity}. Options: {opts}")
    return FIDELITY_MAP[key]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/sessions")
def create_opt_session(request: OptSessionRequest):
    """Create a new optimization session and return its ID."""

    try:
        height, width = _get_fidelity(request.fidelity)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    chord_length = request.chord_length or 3.0
    airfoil_offset_x = request.airfoil_offset_x or 2.0
    airfoil_offset_y = request.airfoil_offset_y or (height * request.cell_size / 2)

    session_id = str(next(_OPT_SESSION_ID))
    config = OptSessionConfig(
        session_id=session_id,
        height=height,
        width=width,
        cst_upper=request.cst_upper,
        cst_lower=request.cst_lower,
        cell_size=request.cell_size,
        dt=request.dt,
        diffusion=request.diffusion,
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
        stream_fps=request.stream_fps,
    )
    _OPT_SESSIONS[session_id] = config
    return {"session_id": session_id, "config": asdict(config)}


def _build_optimization_fns(config: OptSessionConfig):
    """Build the loss/grad functions (pure, no async). Returns closures."""

    n_weights = len(config.cst_upper)

    grid_x, grid_y = compute_grid_coordinates(
        config.height, config.width, config.cell_size
    )

    def _create_mask(weights_upper, weights_lower):
        return create_airfoil_solid_mask(
            weights_upper,
            weights_lower,
            grid_x,
            grid_y,
            config.airfoil_offset_x,
            config.airfoil_offset_y,
            config.chord_length,
            num_cst_points=config.num_cst_points,
            sharpness=config.mask_sharpness,
        )

    def compute_loss_with_aux(params):
        weights_upper = params[:n_weights]
        weights_lower = params[n_weights:]

        # Geometric constraints (mirrors optimize_airfoil.py exactly)
        x_cst, y_upper, y_lower = generate_cst_coords(weights_upper, weights_lower)
        thickness = thickness_at_x(y_upper, y_lower)
        geo_loss = crossover_validity_loss(y_upper, y_lower) + thickness_constraint_loss(
            thickness, config.min_thickness, config.max_thickness
        )

        obstacle_mask = _create_mask(weights_upper, weights_lower)

        sim = FluidGrid(
            height=config.height,
            width=config.width,
            cell_size=config.cell_size,
            dt=config.dt,
            diffusion=config.diffusion,
            boundary_type=2,
            visualise=False,
        )

        state = sim.create_initial_state()
        state = sim.set_velocity_field(state, "wind tunnel")

        def step_fn(i, s):
            return sim.step(s)

        state = jax.lax.fori_loop(0, config.num_sim_steps, step_fn, state)

        pressure = state.pressure.values
        cell_volume = config.cell_size ** 2

        grad_mask_x = jnp.zeros_like(obstacle_mask)
        grad_mask_y = jnp.zeros_like(obstacle_mask)
        grad_mask_x = grad_mask_x.at[:, 1:-1].set(
            (obstacle_mask[:, 2:] - obstacle_mask[:, :-2]) / (2 * config.cell_size)
        )
        grad_mask_y = grad_mask_y.at[1:-1, :].set(
            (obstacle_mask[2:, :] - obstacle_mask[:-2, :]) / (2 * config.cell_size)
        )

        drag_force = jnp.sum(pressure * grad_mask_x) * cell_volume
        lift_force = jnp.sum(pressure * grad_mask_y) * cell_volume

        q = 0.5 * config.inflow_velocity ** 2
        C_D = drag_force / (q * config.chord_length)
        C_L = lift_force / (q * config.chord_length)

        # Loss: minimize |drag| + geometric constraints (same as optimize_airfoil.py)
        total_loss = jnp.abs(drag_force) + geo_loss

        return total_loss, (C_L, C_D, lift_force, drag_force, geo_loss)

    def loss_only(params):
        loss, _ = compute_loss_with_aux(params)
        return loss

    grad_fn = jax.grad(loss_only)

    return n_weights, compute_loss_with_aux, grad_fn


def _run_iteration(params, compute_loss_with_aux, grad_fn, config):
    """Run one optimization iteration (CPU-heavy, called in thread)."""

    loss_val, (C_L, C_D, lift_force, drag_force, geo_loss) = compute_loss_with_aux(params)
    gradients = grad_fn(params)

    # No gradient clipping (matches optimize_airfoil.py)

    has_nan = bool(jnp.isnan(loss_val) or jnp.any(jnp.isnan(gradients)))

    return loss_val, C_L, C_D, lift_force, drag_force, geo_loss, gradients, has_nan


@router.websocket("/ws/{session_id}")
async def stream_optimization(ws: WebSocket, session_id: str):
    """Run airfoil optimization and stream each iteration's results."""

    config = _OPT_SESSIONS.get(session_id)
    if config is None:
        await ws.accept()
        await ws.send_json({"error": "unknown session_id"})
        await ws.close(code=1008)
        return

    await ws.accept()

    # Build functions (fast — no actual computation yet)
    n_weights, compute_loss_with_aux, grad_fn = _build_optimization_fns(config)

    # Initial parameters
    params = jnp.concatenate([
        jnp.array(config.cst_upper),
        jnp.array(config.cst_lower),
    ])

    # Optimizer
    opt_state, update_fn = create_optimizer(
        config.optimizer, learning_rate=config.learning_rate
    )

    print(f"[optimize] Session {config.session_id}: starting {config.num_iterations} iterations")

    try:
        for iteration in range(config.num_iterations):
            # --- Offload heavy JAX compute to thread pool ---
            loss_val, C_L, C_D, lift_force, drag_force, geo_loss, gradients, has_nan = (
                await asyncio.to_thread(
                    _run_iteration, params, compute_loss_with_aux, grad_fn, config
                )
            )

            if has_nan:
                await ws.send_json({
                    "type": "warning",
                    "iteration": iteration + 1,
                    "message": "NaN detected, skipping update",
                })
                continue

            # --- extract current shape for FE ---
            cur_upper = params[:n_weights]
            cur_lower = params[n_weights:]
            x_cst, y_upper_cst, y_lower_cst = generate_cst_coords(
                cur_upper, cur_lower, num_points=config.num_cst_points
            )

            cl_cd = float(C_L) / float(C_D) if abs(float(C_D)) > 1e-12 else 0.0

            payload = {
                "type": "iteration",
                "meta": {
                    "iteration": iteration + 1,
                    "total_iterations": config.num_iterations,
                    "loss": float(loss_val),
                    "cl": float(C_L),
                    "cd": float(C_D),
                    "cl_cd": cl_cd,
                    "lift_force": float(lift_force),
                    "drag_force": float(drag_force),
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

            print(
                f"[optimize] Iter {iteration+1:3d}/{config.num_iterations} | "
                f"Loss: {float(loss_val):10.4f} | CL: {float(C_L):.6f} | "
                f"CD: {float(C_D):.6f} | L/D: {cl_cd:.2f}"
            )

            # --- update params ---
            params, opt_state = update_fn(params, gradients, opt_state)

            # Yield control so the WS frame can be sent
            if config.stream_fps > 0:
                await asyncio.sleep(1.0 / config.stream_fps)
            else:
                await asyncio.sleep(0)

        # ----- send final summary -----
        final_upper = params[:n_weights]
        final_lower = params[n_weights:]
        x_final, y_upper_final, y_lower_final = generate_cst_coords(
            final_upper, final_lower, num_points=config.num_cst_points
        )

        await ws.send_json({
            "type": "complete",
            "meta": {
                "total_iterations": config.num_iterations,
                "final_cl": float(C_L),
                "final_cd": float(C_D),
                "final_cl_cd": cl_cd,
                "final_drag": float(drag_force),
                "final_loss": float(loss_val),
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
        })

        print(f"[optimize] Session {config.session_id}: optimization complete")
        await ws.close()

    except WebSocketDisconnect:
        print(f"[optimize] Session {config.session_id}: client disconnected")
        return
