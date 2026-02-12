import os
import asyncio
import jax.numpy as jnp
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from turbodiff import FluidGrid
from turbodiff.utils.sdf_generator import create_sdf_function

router = APIRouter()


@router.websocket("/ws/validate/rae2822")
async def validate_rae2822(websocket: WebSocket):
    await websocket.accept()

    height = 80
    width = 200
    chord_length = 40
    offset_x = 30
    offset_y = height // 2

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
    dat_filepath = os.path.join(project_root, "Airfoils", "RAE2822.dat")

    if not os.path.exists(dat_filepath):
        print(f"Error: Airfoil file not found at {dat_filepath}")
        await websocket.close(code=1011, reason="Airfoil file not found")
        return

    sdf = create_sdf_function(dat_filepath, chord_length, offset_x, offset_y)

    sim = FluidGrid(
        height=height,
        width=width,
        cell_size=0.01,
        diffusion=0.01,
        viscosity=0.01,
        dt=0.01,
        boundary_type=2,
        visualise=False,
        sdf=sdf,
    )

    sim.is_wind_tunnel = True
    state = sim.create_initial_state()
    state = sim.set_velocity_field(state, field_type="wind tunnel")

    def to_list(arr):
        return jnp.asarray(arr).tolist()

    def compute_curl(u, v, height, width, cell_size):
        curl = jnp.zeros((height, width))

        # v terms (dv/dx)
        curl = curl.at[:, 1:].add((v[:-1, :-1] + v[1:, :-1]) / 2)  # j > 0: down on left
        curl = curl.at[:, :-1].add(
            -(v[:-1, 1:] + v[1:, 1:]) / 2
        )  # j < W-1: down on right

        # u terms (du/dy)
        curl = curl.at[1:, :].add(-(u[:-1, :-1] + u[:-1, 1:]) / 2)  # i > 0: right on up
        curl = curl.at[:-1, :].add(
            (u[1:, :-1] + u[1:, 1:]) / 2
        )  # i < H-1: right on down

        return curl / cell_size

    try:
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
            await asyncio.sleep(0.01)

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Error in validation stream: {e}")
        await websocket.close(code=1011)
