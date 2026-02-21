import asyncio
import json
import pygame
import websockets
import numpy as np
import requests
from turbodiff.core.airfoil import generate_cst_coords
import jax.numpy as jnp


SERVER_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000"

HEIGHT = 128  # Matches 'medium' fidelity
WIDTH = 256
DISPLAY_SIZE = 1000 // max(HEIGHT, WIDTH)

SHOW_VELOCITY = True
SHOW_CELL_PROPERTY = "curl"

# Simulation parameters (must match server defaults)
CELL_SIZE = 0.01  # meters
CHORD_LENGTH = 40 * CELL_SIZE  # 0.4 meters (server default)

# CST coefficients
cst_upper = [0.18, 0.22, 0.20, 0.18, 0.15, 0.12]
cst_lower = [-0.10, -0.08, -0.06, -0.05, -0.04, -0.03]

# Airfoil offsets in meters (world coordinates, matching server defaults)
airfoil_offset_x = 30 * CELL_SIZE  # 0.3 meters (30 cells from left edge)
airfoil_offset_y = (HEIGHT // 2) * CELL_SIZE  # 0.64 meters (vertically centered)


async def run_client():
    pygame.init()
    pygame.display.set_caption("TurboDiff - Streaming Server Validation")
    screen = pygame.display.set_mode((WIDTH * DISPLAY_SIZE, HEIGHT * DISPLAY_SIZE))

    # 1. Create Session
    print("Creating session...")
    try:
        response = requests.post(
            f"{SERVER_URL}/sessions",
            json={
                "user_id": "demo-user",
                "fidelity": "medium",
                "sim_time": 10,  # Infinite
                "cst_upper": cst_upper,  # Default RAE2822
                "cst_lower": cst_lower,  # Default RAE2822
                "stream_every": 1,
                "stream_fps": 30.0,
                "airfoil_offset_x": airfoil_offset_x,
                "airfoil_offset_y": airfoil_offset_y,
            },
        )
        response.raise_for_status()
        session_data = response.json()
        session_id = session_data["session_id"]
        print(f"Session created: {session_id}")
    except Exception as e:
        print(f"Failed to create session: {e}")
        return

    # 2. Connect via WebSocket
    uri = f"{WS_URL}/ws/{session_id}"
    print(f"Connecting to {uri}...")

    # try:
    if True:
        async with websockets.connect(uri, max_size=None) as websocket:
            print("Connected! Waiting for stream...")

            while True:
                message = await websocket.recv()
                payload = json.loads(message)

                if "error" in payload:
                    print(f"Server error: {payload['error']}")
                    break

                step = payload["meta"]["step"]
                cl = payload["meta"].get("cl", 0.0)
                cd = payload["meta"].get("cd", 0.0)
                l_d = payload["meta"].get("l_d", 0.0)

                print(f"Step {step}: CL={cl:.4f}, CD={cd:.4f}, L/D={l_d:.2f}")

                fields = payload.get("fields", {})

                u = np.array(fields["u"])
                v = np.array(fields["v"])
                solid_mask = np.array(fields["solid"])
                curl_field = np.array(fields["curl"])

                screen.fill((0, 0, 0))

                # Draw cells (solid or curl)
                for i in range(HEIGHT):
                    for j in range(WIDTH):
                        if solid_mask[i][j]:
                            # Solid cells
                            shade = 128
                            color = (shade, shade, shade)
                        else:
                            # Fluid cells - visualize curl
                            curl = curl_field[i][j]
                            color = (
                                max(
                                    0, min(255, int(curl * 5))
                                ),  # Scale curl for visibility
                                0,
                                max(0, min(255, int(-curl * 5))),
                            )

                        rect = pygame.Rect(
                            j * DISPLAY_SIZE,
                            i * DISPLAY_SIZE,
                            DISPLAY_SIZE,
                            DISPLAY_SIZE,
                        )
                        pygame.draw.rect(screen, color, rect)

                # Draw velocity arrows
                if SHOW_VELOCITY:
                    # u, v are already cell-centered from this server
                    mags = np.sqrt(u**2 + v**2)
                    mag_safe = np.maximum(mags, 1e-6)
                    u_dir = u / mag_safe
                    v_dir = v / mag_safe

                    for i in range(HEIGHT):
                        for j in range(WIDTH):
                            mag = min(mags[i, j], 1.0)

                            if mag < 0.1:
                                continue

                            r = int(255 * mag)
                            g = 0
                            b = int(255 * (1 - mag))
                            color = (r, g, b)

                            scale = DISPLAY_SIZE * 0.4
                            start_x = (j + 0.5) * DISPLAY_SIZE
                            start_y = (i + 0.5) * DISPLAY_SIZE
                            end_x = start_x + scale * u_dir[i, j]
                            end_y = start_y + scale * v_dir[i, j]

                            pygame.draw.aaline(
                                screen, color, (start_x, start_y), (end_x, end_y)
                            )

                # Draw CST airfoil outline with white border
                # Generate airfoil coordinates in normalized space [0, 1]
                x, y_upper, y_lower = generate_cst_coords(
                    jnp.array(cst_upper),
                    jnp.array(cst_lower),
                    num_points=200,
                    cosine_spacing=True,
                )

                # Convert to numpy arrays
                x = np.array(x)
                y_upper = np.array(y_upper)
                y_lower = np.array(y_lower)

                # Convert from normalized airfoil coordinates [0,1] to world coordinates (meters)
                # x ranges from offset_x to offset_x + chord_length
                # y ranges from offset_y + y_surface (where y_surface is scaled by chord)
                world_x = airfoil_offset_x + x * CHORD_LENGTH
                world_y_upper = airfoil_offset_y + y_upper * CHORD_LENGTH
                world_y_lower = airfoil_offset_y + y_lower * CHORD_LENGTH

                # Convert from world coordinates (meters) to grid coordinates (cell indices)
                grid_x = world_x / CELL_SIZE
                grid_y_upper = world_y_upper / CELL_SIZE
                grid_y_lower = world_y_lower / CELL_SIZE

                # Convert from grid coordinates to screen coordinates (pixels)
                screen_x = grid_x * DISPLAY_SIZE
                screen_y_upper = grid_y_upper * DISPLAY_SIZE
                screen_y_lower = grid_y_lower * DISPLAY_SIZE

                # Create point lists for upper and lower surfaces
                upper_points = [
                    (float(screen_x[i]), float(screen_y_upper[i]))
                    for i in range(len(x))
                ]
                lower_points = [
                    (float(screen_x[i]), float(screen_y_lower[i]))
                    for i in range(len(x))
                ]

                # Draw the airfoil outline with white border (thickness=2)
                if len(upper_points) > 1:
                    pygame.draw.lines(screen, (255, 255, 255), False, upper_points, 2)
                if len(lower_points) > 1:
                    pygame.draw.lines(screen, (255, 255, 255), False, lower_points, 2)

                pygame.display.flip()

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return

    # except Exception as e:
    #     print(f"Connection error: {e}")
    # finally:
    #     pygame.quit()


if __name__ == "__main__":
    asyncio.run(run_client())
