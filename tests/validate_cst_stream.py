import asyncio
import json
import pygame
import websockets
import numpy as np

HEIGHT = 80
WIDTH = 200
DISPLAY_SIZE = 1000 // max(HEIGHT, WIDTH)

SHOW_VELOCITY = True
SHOW_CELL_PROPERTY = "curl"
SHOW_CELL_CENTERED_VELOCITY = True


async def run_client():
    pygame.init()
    pygame.display.set_caption("TurboDiff - CST Validation Stream Client")
    screen = pygame.display.set_mode((WIDTH * DISPLAY_SIZE, HEIGHT * DISPLAY_SIZE))

    uri = "ws://localhost:8001/ws/validate/cst"
    print(f"Connecting to {uri}...")

    # Default RAE2822-like weights
    weights = {
        "weights_upper": [0.18, 0.22, 0.20, 0.18, 0.15, 0.12],
        "weights_lower": [-0.10, -0.08, -0.06, -0.05, -0.04, -0.03],
    }

    try:
        async with websockets.connect(uri, max_size=None) as websocket:
            print("Connected! Sending weights...")
            await websocket.send(json.dumps(weights))
            print("Weights sent. Waiting for stream...")

            while True:
                message = await websocket.recv()
                data = json.loads(message)

                u = np.array(data["u"])
                v = np.array(data["v"])
                solid_mask = np.array(data["solid_mask"])
                curl_field = np.array(data["curl"])

                screen.fill((0, 0, 0))

                # Draw cells (solid or curl)
                for i in range(HEIGHT):
                    for j in range(WIDTH):
                        if solid_mask[i][j]:
                            shade = solid_mask[i][j] * 255
                            color = (shade, shade, shade)
                        else:
                            curl = curl_field[i][j]
                            # Simple diverging color map for curl
                            color = (
                                max(0, min(255, int(curl))),
                                0,
                                max(0, min(255, int(-curl))),
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
                    # Compute cell centers for better visualization
                    u_center = (u[:, :-1] + u[:, 1:]) / 2.0
                    v_center = (v[:-1, :] + v[1:, :]) / 2.0
                    mags = np.sqrt(u_center**2 + v_center**2)

                    mag_safe = np.maximum(mags, 1e-6)
                    u_dir = u_center / mag_safe
                    v_dir = v_center / mag_safe

                    for i in range(HEIGHT):
                        for j in range(WIDTH):
                            mag = min(mags[i, j], 1.0)

                            # Skip drawing very small velocities
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

                pygame.display.flip()

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return

    except Exception as e:
        print(f"Connection error: {e}")
    finally:
        pygame.quit()


if __name__ == "__main__":
    asyncio.run(run_client())
