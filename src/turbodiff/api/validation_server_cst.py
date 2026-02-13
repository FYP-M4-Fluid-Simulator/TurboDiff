import asyncio
import jax.numpy as jnp
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from turbodiff import FluidGrid, FluidState
from turbodiff.core.airfoil_optimization import create_airfoil_solid_mask

router = APIRouter()


@router.websocket("/ws/validate/cst")
async def validate_cst(websocket: WebSocket):
    await websocket.accept()

    # Default grid settings
    height = 80
    width = 200
    cell_size = 0.01
    dt = 0.01

    # Airfoil positioning (physical units)
    chord_length = 40 * cell_size
    offset_x = 30 * cell_size
    offset_y = (height // 2) * cell_size

    # Grid coordinates for mask generation
    j_coords = jnp.arange(width, dtype=jnp.float32)
    i_coords = jnp.arange(height, dtype=jnp.float32)
    j_grid, i_grid = jnp.meshgrid(j_coords, i_coords, indexing="xy")
    grid_x = (j_grid + 0.5) * cell_size
    grid_y = (i_grid + 0.5) * cell_size

    def to_list(arr):
        return jnp.asarray(arr).tolist()

    def compute_curl(u, v, height, width, cell_size):
        curl = jnp.zeros((height, width))
        # v terms (dv/dx)
        curl = curl.at[:, 1:].add((v[:-1, :-1] + v[1:, :-1]) / 2)
        curl = curl.at[:, :-1].add(-(v[:-1, 1:] + v[1:, 1:]) / 2)
        # u terms (du/dy)
        curl = curl.at[1:, :].add(-(u[:-1, :-1] + u[:-1, 1:]) / 2)
        curl = curl.at[:-1, :].add((u[1:, :-1] + u[1:, 1:]) / 2)
        return curl / cell_size

    try:
        # Wait for initial configuration (CST weights)
        data = await websocket.receive_json()
        weights_upper = jnp.array(data.get("weights_upper", []))
        weights_lower = jnp.array(data.get("weights_lower", []))

        if len(weights_upper) == 0 or len(weights_lower) == 0:
            await websocket.close(code=1003, reason="Invalid weights")
            return

        print(f"Received weights: Upper={weights_upper}, Lower={weights_lower}")

        # Generate airfoil mask
        solid_mask = create_airfoil_solid_mask(
            weights_upper,
            weights_lower,
            grid_x,
            grid_y,
            offset_x,
            offset_y,
            chord=chord_length,
            num_cst_points=200,
            sharpness=50.0,
        )

        # Threshold the mask for simulation stability and visualization
        solid_mask = jnp.where(solid_mask < 0.05, 0.0, solid_mask)

        # Initialize Simulator
        sim = FluidGrid(
            height=height,
            width=width,
            cell_size=cell_size,
            diffusion=0.01,
            viscosity=0.01,
            dt=dt,
            boundary_type=2,  # No right boundary
            visualise=False,
            sdf=None,  # We inject mask manually
        )

        sim.is_wind_tunnel = True

        # Setup initial state
        state = sim.create_initial_state()

        # Combine boundary mask with airfoil mask
        combined_mask = jnp.maximum(sim.solid_mask, solid_mask)
        sim.solid_mask = combined_mask

        state = FluidState(
            density=state.density,
            velocity=state.velocity,
            pressure=state.pressure,
            solid_mask=combined_mask,
            sources=state.sources,
            time=state.time,
            step=state.step,
        )

        state = sim.set_velocity_field(state, field_type="wind tunnel")

        # Add smoke sources
        source_positions = []
        for i in range(height):
            if i % 8 < 4:
                source_positions.append((i, 5, 2.0))
        state = sim.set_sources(state, source_positions)

        # Simulation Loop
        while True:
            state = sim.step(state)

            curl_field = compute_curl(
                state.velocity.u, state.velocity.v, height, width, sim.cell_size
            )

            payload = {
                "step": int(state.step),
                "time": float(state.time),
                "height": height,
                "width": width,
                "display_size": 1,
                "solid_mask": to_list(state.solid_mask.astype(int)),
                "pressure": to_list(state.pressure.values),
                "u": to_list(state.velocity.u),
                "v": to_list(state.velocity.v),
                "curl": to_list(curl_field),
            }

            await websocket.send_json(payload)
            # Control simulation speed
            await asyncio.sleep(0.01)

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Error in validation stream: {e}")
        await websocket.close(code=1011)
