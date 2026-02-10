"""FastAPI server for streaming TurboDiff grid data."""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
from itertools import count
from typing import Dict, List, Tuple

import jax.numpy as jnp
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from turbodiff.core.airfoil_optimization import (
    compute_grid_coordinates,
    create_airfoil_solid_mask,
)
from turbodiff.core.fluid_grid_jax import FluidGrid


FIDELITY_MAP: Dict[str, Tuple[int, int]] = {
    "low": (64, 128),
    "medium": (128, 256),
    "coarse": (256, 512),
}


app = FastAPI(title="TurboDiff Streaming API")

_SESSION_ID = count(1)
_SESSIONS: Dict[str, "SessionConfig"] = {}


class SessionRequest(BaseModel):
    fidelity: str = Field("medium", description="low | medium | coarse")
    sim_time: float = Field(1.0, ge=0.0, description="Seconds; 0 for infinite")
    dt: float = Field(0.01, gt=0.0)
    cell_size: float = Field(0.01, gt=0.0)
    diffusion: float = Field(0.001, ge=0.0)
    viscosity: float = Field(0.0, ge=0.0)
    boundary_type: int = Field(1, ge=0, le=2)
    inflow_velocity: float = Field(2.0, ge=0.0)
    stream_fps: float = Field(30.0, ge=0.0)
    stream_every: int = Field(1, ge=1)
    angle_of_attack: float | None = Field(None, description="Degrees")
    cst_upper: List[float] | None = Field(
        None, description="CST upper surface weights"
    )
    cst_lower: List[float] | None = Field(
        None, description="CST lower surface weights"
    )
    airfoil_offset_x: float | None = Field(
        None, description="Leading edge x position in meters"
    )
    airfoil_offset_y: float | None = Field(
        None, description="Centerline y position in meters"
    )
    chord_length: float | None = Field(
        None, description="Chord length in meters"
    )
    num_cst_points: int = Field(100, ge=10)
    mask_sharpness: float = Field(50.0, gt=0.0)


@dataclass(frozen=True)
class SessionConfig:
    session_id: str
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
        raise ValueError(f"Invalid fidelity={fidelity}. Options: {options}")
    return FIDELITY_MAP[fidelity_key]


def _extract_cell_fields(state):
    u = state.velocity.u
    v = state.velocity.v
    u_center = 0.5 * (u[:, :-1] + u[:, 1:])
    v_center = 0.5 * (v[:-1, :] + v[1:, :])
    return u_center, v_center, state.pressure.values, state.solid_mask


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/sessions")
def create_session(request: SessionRequest):
    try:
        height, width = _get_fidelity(request.fidelity)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    has_upper = request.cst_upper is not None
    has_lower = request.cst_lower is not None
    if has_upper != has_lower:
        raise HTTPException(
            status_code=400,
            detail="Both cst_upper and cst_lower must be provided together.",
        )

    domain_width = width * request.cell_size
    domain_height = height * request.cell_size
    chord_length = request.chord_length or (0.25 * domain_width)
    airfoil_offset_x = request.airfoil_offset_x or (0.2 * domain_width)
    airfoil_offset_y = request.airfoil_offset_y or (0.5 * domain_height)

    session_id = str(next(_SESSION_ID))
    config = SessionConfig(
        session_id=session_id,
        height=height,
        width=width,
        sim_time=request.sim_time,
        dt=request.dt,
        cell_size=request.cell_size,
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
    return {"session_id": session_id, "config": asdict(config)}


@app.websocket("/ws/{session_id}")
async def stream_state(ws: WebSocket, session_id: str):
    config = _SESSIONS.get(session_id)
    if config is None:
        await ws.accept()
        await ws.send_json({"error": "unknown session_id"})
        await ws.close(code=1008)
        return

    await ws.accept()

    grid = FluidGrid(
        height=config.height,
        width=config.width,
        cell_size=config.cell_size,
        dt=config.dt,
        diffusion=config.diffusion,
        viscosity=config.viscosity,
        boundary_type=config.boundary_type,
        visualise=False,
    )
    grid.is_wind_tunnel = True
    grid.inlet_velocity = config.inflow_velocity
    angle_deg = config.angle_of_attack or 0.0
    grid.inlet_angle_rad = float(jnp.deg2rad(angle_deg))
    grid.solid_mask = grid.solid_mask.at[1:-1, -1].set(False)

    state = grid.create_initial_state()

    if config.cst_upper is not None and config.cst_lower is not None:
        grid_x, grid_y = compute_grid_coordinates(
            config.height, config.width, config.cell_size
        )
        airfoil_mask = create_airfoil_solid_mask(
            jnp.asarray(config.cst_upper),
            jnp.asarray(config.cst_lower),
            grid_x,
            grid_y,
            config.airfoil_offset_x,
            config.airfoil_offset_y,
            config.chord_length,
            num_cst_points=config.num_cst_points,
            sharpness=config.mask_sharpness,
        )
        combined_mask = jnp.maximum(state.solid_mask.astype(jnp.float32), airfoil_mask)
        state = state.__class__(
            density=state.density,
            velocity=state.velocity,
            pressure=state.pressure,
            solid_mask=combined_mask,
            sources=state.sources,
            time=state.time,
            step=state.step,
        )

    max_steps = int(config.sim_time / config.dt) if config.sim_time > 0 else -1
    step = 0

    try:
        while max_steps < 0 or step < max_steps:
            state = grid.step(state)

            if step % config.stream_every == 0:
                u_center, v_center, pressure, solid = _extract_cell_fields(state)
                payload = {
                    "meta": {
                        "session_id": config.session_id,
                        "height": int(config.height),
                        "width": int(config.width),
                        "cell_size": float(config.cell_size),
                        "time": float(state.time),
                        "step": int(state.step),
                    },
                    "fields": {
                        "u": jnp.asarray(u_center).tolist(),
                        "v": jnp.asarray(v_center).tolist(),
                        "pressure": jnp.asarray(pressure).tolist(),
                        "solid": jnp.asarray(solid).astype(int).tolist(),
                    },
                }
                await ws.send_json(payload)

            step += 1

            if config.stream_fps > 0:
                await asyncio.sleep(1.0 / config.stream_fps)
            else:
                await asyncio.sleep(0)

    except WebSocketDisconnect:
        return
