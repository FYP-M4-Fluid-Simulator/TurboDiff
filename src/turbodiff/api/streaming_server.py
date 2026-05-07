"""FastAPI server for streaming TurboDiff grid data."""

from __future__ import annotations

import asyncio
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple
from uuid import uuid4

import jax
import jax.numpy as jnp
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, Depends
from pydantic import BaseModel, Field
from turbodiff.api.auth import get_current_user, verify_websocket_token

from turbodiff.core.airfoil_optimization import (
    compute_grid_coordinates,
    create_airfoil_solid_mask,
)
from turbodiff.core.fluid_grid_jax import FluidGrid, FluidState
from turbodiff.db.storage import (
    SessionCreatePayload,
    SimulationMetricsUpdate,
    get_storage_repository,
)
from turbodiff.xfoil.runner import run_xfoil, write_dat_file


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
    "low": 0.08,
    "medium": 0.04,
    "high": 0.02,
    "ultra": 0.01,
}

router = APIRouter()
_SESSIONS: Dict[str, "SessionConfig"] = {}
_SIMULATION_RESULTS: Dict[str, dict] = {}


class SessionRequest(BaseModel):
    user_id: str | None = Field(None, description="User identifier")
    fidelity: str = Field("medium", description="low | medium | coarse")
    sim_time: float = Field(0.0, ge=0.0, description="Seconds; 0 for infinite")
    dt: float = Field(0.01, gt=0.0)
    diffusion: float = Field(0.01, ge=0.0)
    viscosity: float = Field(0.01, ge=0.0)
    boundary_type: int = Field(2, ge=0, le=2)
    inflow_velocity: float = Field(2.0, ge=0.0)
    stream_fps: float = Field(30.0, ge=0.0)
    stream_every: int = Field(1, ge=1)
    angle_of_attack: float | None = Field(None, description="Degrees")
    reynolds_number: float | None = Field(None, description="RE")
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
    reynolds_number: float | None
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
    return (
        u_center,
        v_center,
        curl,
        state.pressure.values,
        state.solid_mask,
        state.density.values,
    )


@jax.jit
def compute_aerodynamic_forces(
    state: FluidState,
    solid_mask: jnp.ndarray,
    cell_size: float,
    nu_eff: jnp.ndarray,
    rho: float,
    aoa_rad: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Integrate pressure + viscous stresses on the airfoil surface.
    Returns (F_drag, F_lift) in wind axes (N/m span).
    """
    u = state.velocity.u
    v = state.velocity.v
    p = state.pressure.values
    h = cell_size
    mu = rho * nu_eff  # dynamic viscosity field (h, w)

    fluid = 1.0 - solid_mask  # 1 in fluid, 0 in solid

    # Pressure force via surface integration
    is_right_wall = fluid[:, :-1] * solid_mask[:, 1:]
    is_left_wall = fluid[:, 1:] * solid_mask[:, :-1]
    is_bottom_wall = fluid[:-1, :] * solid_mask[1:, :]
    is_top_wall = fluid[1:, :] * solid_mask[:-1, :]

    Fpx = (
        jnp.sum(p[:, :-1] * is_right_wall) * h  # +x on right wall
        - jnp.sum(p[:, 1:] * is_left_wall) * h  # -x on left wall
    )
    Fpy = (
        jnp.sum(p[:-1, :] * is_bottom_wall) * h  # +y on bottom wall
        - jnp.sum(p[1:, :] * is_top_wall) * h  # -y on top wall
    )

    # Viscous force (shear stress at walls)
    u_cc = 0.5 * (u[:, :-1] + u[:, 1:])
    v_cc = 0.5 * (v[:-1, :] + v[1:, :])

    dudy_int = jnp.zeros_like(p)
    dvdx_int = jnp.zeros_like(p)
    dudx_int = jnp.zeros_like(p)
    dvdy_int = jnp.zeros_like(p)

    dudy_int = dudy_int.at[1:-1, :].set((u_cc[2:, :] - u_cc[:-2, :]) / (2.0 * h))
    dvdx_int = dvdx_int.at[:, 1:-1].set((v_cc[:, 2:] - v_cc[:, :-2]) / (2.0 * h))
    dudx_int = dudx_int.at[:, 1:-1].set((u_cc[:, 2:] - u_cc[:, :-2]) / (2.0 * h))
    dvdy_int = dvdy_int.at[1:-1, :].set((v_cc[2:, :] - v_cc[:-2, :]) / (2.0 * h))

    tau_xy = mu * (dudy_int + dvdx_int)
    tau_xx = 2.0 * mu * dudx_int
    tau_yy = 2.0 * mu * dvdy_int

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

    Fx = Fpx + Fvx
    Fy = Fpy + Fvy

    cos_a = jnp.cos(aoa_rad)
    sin_a = jnp.sin(aoa_rad)
    F_drag = Fx * cos_a + Fy * sin_a
    F_lift = -Fx * sin_a + Fy * cos_a

    return F_drag, F_lift


@router.get("/health")
def health_check():
    return {"status": "healthy"}


@router.post("/sessions")
def create_session(request: SessionRequest, user: dict = Depends(get_current_user)):
    user_id = user.get("uid")
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
    cell_size = CELL_SIZE_MAP[fidelity_key]

    # Default chord length: 1.0 m, giving 25–200 cells/chord across fidelity levels.
    # Users can override via request.chord_length.
    chord_length = request.chord_length or 1.0
    airfoil_offset_x = request.airfoil_offset_x or (30 * cell_size)
    airfoil_offset_y = request.airfoil_offset_y or (height // 2 * cell_size)

    session_id = str(uuid4())
    config = SessionConfig(
        session_id=session_id,
        user_id=user_id,
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
        reynolds_number=request.reynolds_number,
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
                user_id=user_id,
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


@router.websocket("/ws/{session_id}")
async def stream_state(ws: WebSocket, session_id: str):
    await ws.accept()

    user = await verify_websocket_token(ws)
    if user is None:
        await ws.send_json({"error": "Missing or invalid authentication token"})
        await ws.close(code=1008)
        return

    config = _SESSIONS.get(session_id)
    if config is None:
        await ws.send_json({"error": "unknown session_id"})
        await ws.close(code=1008)
        return

    if config.cst_upper is None or config.cst_lower is None:
        await ws.send_json({"error": "Missing weights in session config"})
        await ws.close(code=1003)
        return
    else:
        cst_upper = config.cst_upper
        cst_lower = config.cst_lower

    # Resolve viscosity and Reynolds number
    if config.reynolds_number:
        viscosity = (
            config.inflow_velocity * config.chord_length
        ) / config.reynolds_number
    else:
        viscosity = config.viscosity

    grid = FluidGrid(
        height=config.height,
        width=config.width,
        cell_size=config.cell_size,
        dt=config.dt,
        diffusion=0.0,  # Typically 0 for RANS density tracer
        viscosity=viscosity,
        boundary_type=0,  # We manage boundaries manually like the benchmark
        use_sa_turbulence=True,
        visualise=False,
        sdf=None,
    )

    # Configure wind tunnel
    grid.inlet_velocity = config.inflow_velocity
    angle_deg = config.angle_of_attack or 0.0
    angle_rad = float(jnp.deg2rad(angle_deg))
    grid.inlet_angle_rad = -angle_rad  # Rotate the wind

    state = grid.create_initial_state()

    grid_x, grid_y = compute_grid_coordinates(
        config.height, config.width, config.cell_size
    )

    # Use static airfoil (no rotation in mask creation)
    airfoil_mask = create_airfoil_solid_mask(
        jnp.asarray(cst_upper),
        jnp.asarray(cst_lower),
        grid_x,
        grid_y,
        config.airfoil_offset_x,
        config.airfoil_offset_y,
        config.chord_length,
        num_cst_points=config.num_cst_points,
        sharpness=config.mask_sharpness,
    )

    # Threshold fix for soft mask
    airfoil_mask = jnp.where(airfoil_mask < 0.05, 0.0, airfoil_mask)

    # Combine with wind tunnel boundaries (solid top/bottom, open right)
    combined_mask = airfoil_mask.copy()
    combined_mask = combined_mask.at[0, :].set(0.0)  # Top wall
    combined_mask = combined_mask.at[-1, :].set(0.0)  # Bottom wall
    # Right wall remains 0.0 (open)

    grid.solid_mask = combined_mask

    # Update wall distances for SA model
    grid._wall_dist = grid._compute_wall_distance(combined_mask)

    state = state.__class__(
        density=state.density,
        velocity=state.velocity,
        pressure=state.pressure,
        solid_mask=combined_mask,
        sources=state.sources,
        nu_tilde=state.nu_tilde,
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
    last_cl: float | None = None
    last_cd: float | None = None

    print(f"   Starting RANS simulation for session {config.session_id}")
    print(f"   Re={config.reynolds_number}, nu={viscosity:.2e}")
    print(f"   sim_time={config.sim_time}s, dt={config.dt}s, max_steps={max_steps}")

    # Number of simulation steps to run in a single executor call.
    # Larger chunks = less asyncio overhead; smaller = more responsive WebSocket.
    # 20 steps is a good balance: keeps each executor call under ~1 second
    # and lets the event loop service keepalive pings between chunks.
    _CHUNK_SIZE = 20

    def _run_chunk(current_state: "FluidState", n_steps: int) -> "FluidState":
        """Run n_steps of the RANS simulation synchronously in a threadpool worker.
        Defined outside the loop to avoid re-creating the closure on every iteration.
        """
        s = current_state
        for _ in range(n_steps):
            # 1. SA turbulence step
            s = grid.step_sa_turbulence(s, grid._wall_dist, num_diff_iters=4)
            nu_eff = grid.compute_effective_viscosity(s)

            # 2. Velocity step
            s = grid.diffuse_velocity(s, num_iters=20, nu_eff_field=nu_eff)
            s = grid.advect_velocity(s)

            # 3. Density step
            s = grid.add_sources_to_density(s)
            s = grid.advect_density(s)

            # 4. Boundary enforcement and pressure
            s = grid.inject_wind_tunnel_velocity(s)
            s = grid.solve_pressure(s, num_iters=60)
            s = grid.project_velocity(s)
            s = grid.inject_wind_tunnel_velocity(s)

            s = s.__class__(
                density=s.density,
                velocity=s.velocity,
                pressure=s.pressure,
                solid_mask=s.solid_mask,
                sources=s.sources,
                nu_tilde=s.nu_tilde,
                time=s.time + grid.dt,
                step=s.step + 1,
            )
        return s

    loop = asyncio.get_running_loop()

    try:
        while True:
            # Check if we should stop before running the next chunk
            if max_steps > 0 and step >= max_steps:
                print(
                    f"   Simulation complete: reached {step} steps (max: {max_steps})"
                )
                print("   Breaking out of loop...")
                break

            # Determine how many steps to run in this chunk
            if max_steps > 0:
                chunk = min(_CHUNK_SIZE, max_steps - step)
            else:
                chunk = _CHUNK_SIZE

            # ── Offload the heavy JAX computation to a threadpool. ──────────────
            # This yields control back to the asyncio event loop between chunks,
            # allowing WebSocket keepalive pings to be serviced and preventing
            # the "keepalive ping failed / AssertionError" crash.
            state = await loop.run_in_executor(None, _run_chunk, state, chunk)
            step += chunk

            # Log progress every 100 steps
            if (step // chunk) % (100 // _CHUNK_SIZE or 1) == 0:
                print(f"   Progress: step={step}/{max_steps}")

            if step % config.stream_every == 0:
                # Calculate aerodynamic coefficients using the benchmark method
                nu_eff_field = grid.compute_effective_viscosity(state)
                F_drag, F_lift = compute_aerodynamic_forces(
                    state,
                    airfoil_mask,  # Use only the airfoil mask for forces
                    config.cell_size,
                    nu_eff_field,
                    grid.rho,
                    angle_rad,
                )

                q_inf = 0.5 * grid.rho * config.inflow_velocity**2
                ref_area = config.chord_length

                last_cl = float(F_lift / (q_inf * ref_area))
                last_cd = float(F_drag / (q_inf * ref_area))

                # Avoid division by zero for L/D
                l_d = last_cl / last_cd if abs(last_cd) > 1e-9 else 0.0

                u_center, v_center, curl, pressure, solid, density = (
                    _extract_cell_fields(state, config.cell_size)
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
                        "cl": last_cl,
                        "cd": last_cd,
                        "l_d": float(l_d),
                    },
                    "fields": {
                        "u": jnp.asarray(u_center).tolist(),
                        "v": jnp.asarray(v_center).tolist(),
                        "curl": jnp.asarray(curl).tolist(),
                        "pressure": jnp.asarray(pressure).tolist(),
                        "solid": jnp.asarray(solid).astype(int).tolist(),
                        "tracer": jnp.asarray(density).tolist(),
                    },
                }
                await ws.send_json(payload)
                _SIMULATION_RESULTS[session_id] = payload

            if config.stream_fps > 0:
                await asyncio.sleep(1.0 / config.stream_fps)
            else:
                await asyncio.sleep(0)

        # ── XFoil validation on the completed airfoil ─────────────────────
        xfoil_cl: float | None = None
        xfoil_cd: float | None = None
        xfoil_l_d: float | None = None
        xfoil_status: str = "not_run"

        re = config.reynolds_number or (
            (config.inflow_velocity * config.chord_length) / viscosity
        )
        aoa_deg = config.angle_of_attack or 0.0

        # Create temporary DAT and TXT files for XFoil validation.
        # These are cleaned up in the finally block below.
        # Persist the DAT file in the project's Airfoils/ directory for inspection.
        # The project root is three levels up from this file (api/turbodiff/src).
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..")
        )
        _AIRFOILS_DIR = os.path.join(project_root, "Airfoils")
        os.makedirs(_AIRFOILS_DIR, exist_ok=True)
        dat_path = os.path.join(_AIRFOILS_DIR, f"{session_id}.dat")

        # work_dir = "/tmp/airfoils"
        # os.makedirs(work_dir, exist_ok=True)

        # dat_path = os.path.join(work_dir, f"{session_id}.dat")
    
        print(f"   Running XFoil validation: Re={re:.2e}, AoA={aoa_deg}°")
        print(f"   DAT file → {dat_path}")
        try:
            loop = asyncio.get_event_loop()

            # write_dat_file is CPU-bound and fast — run in thread pool so we
            # don't block the event loop.
            await loop.run_in_executor(
                None,
                write_dat_file,
                cst_upper,
                cst_lower,
                dat_path,
                f"TurboDiff session {session_id}",
                200,
            )

            # run_xfoil spawns a subprocess — also run off the event loop.
            # The polar .txt lands next to the .dat (same directory).
            xfoil_result = await loop.run_in_executor(
                None,
                lambda: run_xfoil(dat_path, re, aoa_deg),
            )

            if xfoil_result is not None:
                xfoil_cl, xfoil_cd = xfoil_result
                xfoil_l_d = xfoil_cl / xfoil_cd if abs(xfoil_cd) > 1e-9 else 0.0
                xfoil_status = "converged"
                print(
                    f"   XFoil: Cl={xfoil_cl:.4f}, Cd={xfoil_cd:.5f}, L/D={xfoil_l_d:.2f}"
                )
                print(f"   Polar → {dat_path.replace('.dat', '.txt')}")
            else:
                xfoil_status = "failed"
                print(f"   XFoil did not converge for Re={re:.2e}, AoA={aoa_deg}°")
        except Exception as xf_err:
            xfoil_status = "error"
            print(f"   XFoil exception: {xf_err}")
        finally:
            # Clean up temporary files to keep the directory clean
            for path in [dat_path, dat_path.replace(".dat", ".txt")]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except Exception:
                        pass

        # Send a final "results" message with XFoil data before closing
        if last_cl is not None and last_cd is not None:
            final_l_d = last_cl / last_cd if abs(last_cd) > 1e-9 else 0.0
            final_payload = {
                "type": "final_results",
                "meta": {
                    "session_id": config.session_id,
                    "height": int(config.height),
                    "width": int(config.width),
                    "cell_size": float(config.cell_size),
                    "chord_length": float(config.chord_length),
                    "time": float(state.time),
                    "step": int(state.step),
                    "cl": last_cl,
                    "cd": last_cd,
                    "l_d": float(final_l_d),
                    "xfoil_cl": xfoil_cl,
                    "xfoil_cd": xfoil_cd,
                    "xfoil_l_d": xfoil_l_d,
                    "xfoil_status": xfoil_status,
                },
            }
            try:
                await ws.send_json(final_payload)
            except Exception:
                pass  # Client may have disconnected; that's fine

            # Update the in-memory cache with XFoil results
            if session_id in _SIMULATION_RESULTS:
                cached = _SIMULATION_RESULTS[session_id]
                cached_meta = dict(cached.get("meta", {}))
                cached_meta["xfoil_cl"] = xfoil_cl
                cached_meta["xfoil_cd"] = xfoil_cd
                cached_meta["xfoil_l_d"] = xfoil_l_d
                cached_meta["xfoil_status"] = xfoil_status
                _SIMULATION_RESULTS[session_id] = {**cached, "meta": cached_meta}

        # Simulation completed, close the connection
        print(f"Closing WebSocket for session {config.session_id}")
        await ws.close()

    except WebSocketDisconnect:
        print(f"WebSocket disconnected for session {config.session_id}")
    finally:
        if last_cl is not None and last_cd is not None:
            print(
                f"Saving final metrics for session {config.session_id}: cl={last_cl:.4f}, cd={last_cd:.4f}"
            )
            try:
                repo = get_storage_repository()
                repo.update_simulation_metrics(
                    SimulationMetricsUpdate(
                        session_id=session_id,
                        user_id=config.user_id,
                        cl=last_cl,
                        cd=last_cd,
                        lift=None,
                        drag=None,
                        angle_of_attack=config.angle_of_attack,
                    )
                )
            except Exception as e:
                print(
                    f"Failed to auto-save simulation metrics for session {session_id}: {e}"
                )


@router.get("/sessions/{session_id}/result")
def get_simulation_result(session_id: str, user: dict = Depends(get_current_user)):
    """Get the simulation result from cache or db."""
    user_id = user.get("uid")

    # 1. Check in-memory cache first
    if session_id in _SIMULATION_RESULTS:
        config = _SESSIONS.get(session_id)
        if config and config.user_id != user_id:
            raise HTTPException(
                status_code=403, detail="Not authorized to access this session"
            )
        return _SIMULATION_RESULTS[session_id]

    # 2. Fallback to database
    repo = get_storage_repository()

    session_record = repo.get_session(session_id)
    if not session_record:
        raise HTTPException(status_code=404, detail="Session not found")
    if session_record.session_type != "simulate":
        raise HTTPException(
            status_code=400, detail="This session is not a simulation session"
        )

    airfoil = repo.get_latest_airfoil(session_id, is_optimized=False)
    if not airfoil:
        raise HTTPException(
            status_code=404, detail="Simulation result not found for this session"
        )

    if str(airfoil.created_by_user_id) != str(user_id):
        raise HTTPException(
            status_code=403, detail="Not authorized to access this session"
        )

    # 3. Resolve grid parameters: prefer in-memory config, fallback to DB session parameters
    config = _SESSIONS.get(session_id)
    if config:
        height = config.height
        width = config.width
        cell_size = config.cell_size
        chord_length = config.chord_length
        airfoil_offset_x = config.airfoil_offset_x
        airfoil_offset_y = config.airfoil_offset_y
    else:
        # Server was restarted — extract from the stored session parameters
        resolved = (session_record.parameters or {}).get("resolved", {})
        height = resolved.get("height", 0)
        width = resolved.get("width", 0)
        cell_size = resolved.get("cell_size", 0.0)
        chord_length = resolved.get("chord_length", 0.0)
        airfoil_offset_x = resolved.get("airfoil_offset_x", 0.0)
        airfoil_offset_y = resolved.get("airfoil_offset_y", 0.0)

    cl = airfoil.cl if airfoil.cl is not None else 0.0
    cd = airfoil.cd if airfoil.cd is not None else 0.0
    l_d = cl / cd if abs(cd) > 1e-9 else 0.0

    payload = {
        "meta": {
            "session_id": session_id,
            "height": height,
            "width": width,
            "cell_size": cell_size,
            "chord_length": chord_length,
            "airfoil_offset_x": airfoil_offset_x,
            "airfoil_offset_y": airfoil_offset_y,
            "time": 0.0,
            "step": 0,
            "cl": cl,
            "cd": cd,
            "l_d": l_d,
        },
        "fields": {
            "u": [],
            "v": [],
            "curl": [],
            "pressure": [],
            "solid": [],
            "tracer": [],
        },
    }

    _SIMULATION_RESULTS[session_id] = payload
    print("Payload of   session " + session_id + " is: ", payload)
    return payload
