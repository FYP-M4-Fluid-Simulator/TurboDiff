"""FastAPI server for streaming TurboDiff grid data."""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple
from uuid import uuid4

import jax.numpy as jnp
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from turbodiff.core.airfoil_optimization import (
    compute_grid_coordinates,
    create_airfoil_solid_mask,
    compute_force_coefficients,
)
from turbodiff.core.fluid_grid_jax import FluidGrid
from turbodiff.db.storage import (
    SessionCreatePayload,
    SimulationMetricsUpdate,
    get_storage_repository,
)


# Grid fidelity presets — keys match the frontend "meshDensity" values after mapping.
# (height, width) in grid cells.
FIDELITY_MAP: Dict[str, Tuple[int, int]] = {
    "low": (64, 128),  # Coarse  (Fast)
    "medium": (128, 256),  # Medium  (Balanced)
    "high": (256, 512),  # Fine    (Detailed)
    "ultra": (512, 1024),  # Ultra   (Precise)
}

# Default cell size (metres) for each fidelity level.
# Keeps the airfoil chord at ≈ 40 cells regardless of resolution.
CELL_SIZE_MAP: Dict[str, float] = {
    "low": 0.04,
    "medium": 0.02,
    "high": 0.01,
    "ultra": 0.005,
}

router = APIRouter()
_SESSIONS: Dict[str, "SessionConfig"] = {}


class SessionRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")
    fidelity: str = Field("medium", description="low | medium | coarse")
    sim_time: float = Field(0.0, ge=0.0, description="Seconds; 0 for infinite")
    dt: float = Field(0.01, gt=0.0)
    cell_size: float = Field(0.01, gt=0.0)
    diffusion: float = Field(0.01, ge=0.0)
    viscosity: float = Field(0.01, ge=0.0)
    boundary_type: int = Field(2, ge=0, le=2)
    inflow_velocity: float = Field(2.0, ge=0.0)
    stream_fps: float = Field(30.0, ge=0.0)
    stream_every: int = Field(1, ge=1)
    angle_of_attack: float | None = Field(None, description="Degrees")
    cst_upper: List[float] | None = Field(None, description="CST upper surface weights")
    cst_lower: List[float] | None = Field(None, description="CST lower surface weights")
    airfoil_offset_x: float | None = Field(
        None, description="Leading edge x position in meters"
    )
    airfoil_offset_y: float | None = Field(
        None, description="Centerline y position in meters"
    )
    chord_length: float | None = Field(None, description="Chord length in meters")
    num_cst_points: int = Field(100, ge=10)
    mask_sharpness: float = Field(500.0, gt=0.0)


@dataclass(frozen=True)
class SessionConfig:
    session_id: str
    user_id: str
    height: int
    width: int
    sim_time: float
    dt: float
    cell_size: float
    diffusion: float
    viscosity: float
    boundary_type: int
    inflow_velocity: float
    stream_fps: float
    stream_every: int
    angle_of_attack: float | None
    cst_upper: List[float] | None
    cst_lower: List[float] | None
    airfoil_offset_x: float
    airfoil_offset_y: float
    chord_length: float
    num_cst_points: int
    mask_sharpness: float


def _get_fidelity(fidelity: str) -> Tuple[int, int]:
    fidelity_key = fidelity.lower()
    if fidelity_key not in FIDELITY_MAP:
        options = ", ".join(sorted(FIDELITY_MAP.keys()))
        raise ValueError(f"Invalid fidelity='{fidelity}'. Valid options: {options}")
    return FIDELITY_MAP[fidelity_key]


def _compute_curl(u, v, cell_size):
    # Centered difference curl calculation
    height, width = u.shape[0], v.shape[1]
    curl = jnp.zeros((height, width))

    # v terms (dv/dx)
    curl = curl.at[:, 1:].add((v[:-1, :-1] + v[1:, :-1]) / 2)
    curl = curl.at[:, :-1].add(-(v[:-1, 1:] + v[1:, 1:]) / 2)

    # u terms (du/dy)
    curl = curl.at[1:, :].add(-(u[:-1, :-1] + u[:-1, 1:]) / 2)
    curl = curl.at[:-1, :].add((u[1:, :-1] + u[1:, 1:]) / 2)

    return curl / cell_size


def _extract_cell_fields(state, cell_size):
    u = state.velocity.u
    v = state.velocity.v
    u_center = 0.5 * (u[:, :-1] + u[:, 1:])
    v_center = 0.5 * (v[:-1, :] + v[1:, :])
    curl = _compute_curl(u, v, cell_size)
    return u_center, v_center, curl, state.pressure.values, state.solid_mask


@router.get("/health")
def health_check():
    return {"status": "ok"}


@router.post("/sessions")
def create_session(request: SessionRequest):
    try:
        height, width = _get_fidelity(request.fidelity)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if (request.cst_upper is None) != (request.cst_lower is None):
        raise HTTPException(
            status_code=400,
            detail="Both cst_upper and cst_lower must be provided together, or both be null.",
        )

    # Resolve cell_size: use the per-fidelity default when the caller sends the
    # generic default (0.01) so the airfoil always covers ~40 cells of chord.
    fidelity_key = request.fidelity.lower()
    cell_size = (
        CELL_SIZE_MAP.get(fidelity_key, request.cell_size)
        if request.cell_size == 0.01  # unchanged from schema default
        else request.cell_size
    )

    # Default chord length: 1.0 m, giving 25–200 cells/chord across fidelity levels.
    # Users can override via request.chord_length.
    chord_length = request.chord_length or 1.0
    airfoil_offset_x = request.airfoil_offset_x or (30 * cell_size)
    airfoil_offset_y = request.airfoil_offset_y or (height // 2 * cell_size)

    session_id = str(uuid4())
    config = SessionConfig(
        session_id=session_id,
        user_id=request.user_id,
        height=height,
        width=width,
        sim_time=request.sim_time,
        dt=request.dt,
        cell_size=cell_size,
        diffusion=request.diffusion,
        viscosity=request.viscosity,
        boundary_type=request.boundary_type,
        inflow_velocity=request.inflow_velocity,
        stream_fps=request.stream_fps,
        stream_every=request.stream_every,
        angle_of_attack=request.angle_of_attack,
        cst_upper=request.cst_upper,
        cst_lower=request.cst_lower,
        airfoil_offset_x=airfoil_offset_x,
        airfoil_offset_y=airfoil_offset_y,
        chord_length=chord_length,
        num_cst_points=request.num_cst_points,
        mask_sharpness=request.mask_sharpness,
    )
    _SESSIONS[session_id] = config

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
        },
    }
    try:
        storage = repo.create_session_with_airfoil(
            SessionCreatePayload(
                session_id=session_id,
                user_id=request.user_id,
                session_type="simulate",
                parameters=parameters,
                cst_weights_upper=request.cst_upper or [],
                cst_weights_lower=request.cst_lower or [],
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


class SimulationSaveRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")
    cl: float | None = None
    cd: float | None = None
    lift: float | None = None
    drag: float | None = None
    angle_of_attack: float | None = None


@router.post("/sessions/{session_id}/save")
def save_simulation_metrics(session_id: str, request: SimulationSaveRequest):
    repo = get_storage_repository()
    try:
        airfoil = repo.update_simulation_metrics(
            SimulationMetricsUpdate(
                session_id=session_id,
                user_id=request.user_id,
                cl=request.cl,
                cd=request.cd,
                lift=request.lift,
                drag=request.drag,
                angle_of_attack=request.angle_of_attack,
            )
        )
    except ValueError as exc:
        detail = str(exc)
        status = 404 if "no matching airfoil" in detail else 400
        raise HTTPException(status_code=status, detail=detail) from exc

    return {"airfoil_id": airfoil.id}


@router.websocket("/ws/{session_id}")
async def stream_state(ws: WebSocket, session_id: str):
    config = _SESSIONS.get(session_id)
    if config is None:
        await ws.accept()
        await ws.send_json({"error": "unknown session_id"})
        await ws.close(code=1008)
        return

    await ws.accept()

    if config.cst_upper is None or config.cst_lower is None:
        await ws.send_json({"error": "Missing weights in session config"})
        await ws.close(code=1003)
        return
    else:
        cst_upper = config.cst_upper
        cst_lower = config.cst_lower

    grid = FluidGrid(
        height=config.height,
        width=config.width,
        cell_size=config.cell_size,
        dt=config.dt,
        diffusion=config.diffusion,
        viscosity=config.viscosity,
        boundary_type=config.boundary_type,
        visualise=False,
        sdf=None,  # We inject mask manually
    )

    grid.is_wind_tunnel = True
    grid.inlet_velocity = config.inflow_velocity
    angle_deg = config.angle_of_attack or 0.0
    # Wind stays horizontal; the airfoil is rotated instead.

    state = grid.create_initial_state()

    grid_x, grid_y = compute_grid_coordinates(
        config.height, config.width, config.cell_size
    )

    # Rotate the airfoil by rotating the grid sample coords in the opposite
    # direction around the airfoil's midpoint (leading-edge + half-chord).
    pivot_x = config.airfoil_offset_x + config.chord_length / 2.0
    pivot_y = config.airfoil_offset_y
    angle_rad = float(jnp.deg2rad(angle_deg))
    cos_a, sin_a = jnp.cos(angle_rad), jnp.sin(angle_rad)
    dx = grid_x - pivot_x
    dy = grid_y - pivot_y
    # Rotate grid points by +angle (equivalent to rotating airfoil by -angle)
    grid_x_rot = cos_a * dx + sin_a * dy + pivot_x
    grid_y_rot = -sin_a * dx + cos_a * dy + pivot_y

    airfoil_mask = create_airfoil_solid_mask(
        jnp.asarray(cst_upper),
        jnp.asarray(cst_lower),
        grid_x_rot,
        grid_y_rot,
        config.airfoil_offset_x,
        config.airfoil_offset_y,
        config.chord_length,
        num_cst_points=config.num_cst_points,
        sharpness=config.mask_sharpness,
    )

    # Threshold fix for soft mask
    airfoil_mask = jnp.where(airfoil_mask < 0.05, 0.0, airfoil_mask)

    # Combine with boundary
    combined_mask = jnp.maximum(grid.solid_mask, airfoil_mask)
    grid.solid_mask = combined_mask

    state = state.__class__(
        density=state.density,
        velocity=state.velocity,
        pressure=state.pressure,
        solid_mask=combined_mask,
        sources=state.sources,
        time=state.time,
        step=state.step,
    )

    state = grid.set_velocity_field(state, field_type="wind tunnel")

    # Add smoke sources (same visualization as validation_server_cst.py)
    source_positions = []
    for i in range(config.height):
        if i % 8 < 4:
            source_positions.append((i, 5, 2.0))
    state = grid.set_sources(state, source_positions)

    max_steps = int(config.sim_time / config.dt) if config.sim_time > 0 else -1
    step = 0

    print(f"   Starting simulation for session {config.session_id}")
    print(f"   sim_time={config.sim_time}s, dt={config.dt}s, max_steps={max_steps}")

    try:
        while True:
            # Log every 100 steps
            if step % 100 == 0:
                print(f"   Progress: step={step}/{max_steps}")

            # Check if we should stop
            if max_steps > 0 and step >= max_steps:
                print(
                    f"   Simulation complete: reached {step} steps (max: {max_steps})"
                )
                print("   Breaking out of loop...")
                break

            state = grid.step(state)
            step += 1

            if step % config.stream_every == 0:
                # Calculate aerodynamic coefficients
                cl, cd = compute_force_coefficients(
                    state,
                    airfoil_mask,
                    inflow_velocity=config.inflow_velocity,
                    chord_length=config.chord_length,
                )

                # Avoid division by zero for L/D
                l_d = cl / cd if abs(cd) > 1e-9 else 0.0

                u_center, v_center, curl, pressure, solid = _extract_cell_fields(
                    state, config.cell_size
                )
                payload = {
                    "meta": {
                        "session_id": config.session_id,
                        "height": int(config.height),
                        "width": int(config.width),
                        "cell_size": float(config.cell_size),
                        "chord_length": float(config.chord_length),
                        "airfoil_offset_x": float(config.airfoil_offset_x),
                        "airfoil_offset_y": float(config.airfoil_offset_y),
                        "time": float(state.time),
                        "step": int(state.step),
                        "cl": float(cl),
                        "cd": float(cd),
                        "l_d": float(l_d),
                    },
                    "fields": {
                        "u": jnp.asarray(u_center).tolist(),
                        "v": jnp.asarray(v_center).tolist(),
                        "curl": jnp.asarray(curl).tolist(),
                        "pressure": jnp.asarray(pressure).tolist(),
                        "solid": jnp.asarray(solid).astype(int).tolist(),
                    },
                }
                await ws.send_json(payload)

            if config.stream_fps > 0:
                await asyncio.sleep(1.0 / config.stream_fps)
            else:
                await asyncio.sleep(0)

        # Simulation completed, close the connection
        print(f"Closing WebSocket for session {config.session_id}")
        await ws.close()

    except WebSocketDisconnect:
        print(f"WebSocket disconnected for session {config.session_id}")
        return
