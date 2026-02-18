"""Validation client for the optimization streaming API.

Creates an optimization session, connects via WebSocket, and
draws the evolving airfoil shape in a Pygame window as each
iteration's results arrive.
"""

import asyncio
import json

import numpy as np
import pygame
import requests
import websockets

SERVER_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000"

# Display settings
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 500

# Colors
BG_COLOR = (18, 18, 24)
GRID_COLOR = (40, 40, 55)
INITIAL_FILL = (180, 60, 60, 80)
INITIAL_OUTLINE = (220, 80, 80)
CURRENT_FILL = (60, 180, 90, 80)
CURRENT_OUTLINE = (80, 220, 110)
TEXT_COLOR = (220, 220, 230)
LABEL_COLOR = (140, 140, 160)
ACCENT_COLOR = (100, 160, 255)


def world_to_screen(x, y, x_range, y_range, margin=60):
    """Map normalized airfoil coords to screen pixels."""
    plot_w = WINDOW_WIDTH - 2 * margin
    plot_h = WINDOW_HEIGHT - 2 * margin - 100  # leave room for metrics

    sx = margin + (x - x_range[0]) / (x_range[1] - x_range[0]) * plot_w
    sy = margin + 50 + (1.0 - (y - y_range[0]) / (y_range[1] - y_range[0])) * plot_h
    return float(sx), float(sy)


async def run_client():
    pygame.init()
    pygame.display.set_caption("TurboDiff – Optimization Viewer")
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    font_large = pygame.font.SysFont("monospace", 18, bold=True)
    font_small = pygame.font.SysFont("monospace", 14)

    # ---- 1. Create optimisation session ----
    print("Creating optimization session …")
    try:
        resp = requests.post(
            f"{SERVER_URL}/optimize/sessions",
            json={
                "fidelity": "low",
                "num_iterations": 30,
                "learning_rate": 0.005,
                "num_sim_steps": 80,
                "cst_upper": [0.18, 0.22, 0.20, 0.18, 0.15, 0.12],
                "cst_lower": [-0.10, -0.08, -0.06, -0.05, -0.04, -0.03],
            },
        )
        resp.raise_for_status()
        data = resp.json()
        session_id = data["session_id"]
        print(f"Session created: {session_id}")
    except Exception as e:
        print(f"Failed to create session: {e}")
        return

    # Save initial shape for comparison
    initial_upper = [0.18, 0.22, 0.20, 0.18, 0.15, 0.12]
    initial_lower = [-0.10, -0.08, -0.06, -0.05, -0.04, -0.03]

    # Pre-generate initial shape coords (normalized, 100 points)
    t = np.linspace(0, np.pi, 100)
    init_x = 0.5 * (1 - np.cos(t))  # cosine spacing
    # We'll receive the actual shape from the first iteration anyway

    uri = f"{WS_URL}/optimize/ws/{session_id}"
    print(f"Connecting to {uri} …")

    # Metrics history for sparkline
    loss_history = []
    cl_cd_history = []

    async with websockets.connect(uri, max_size=None) as websocket:
        print("Connected! Waiting for iterations …")

        while True:
            try:
                message = await websocket.recv()
            except websockets.exceptions.ConnectionClosed:
                print("Connection closed by server (optimization complete).")
                break

            payload = json.loads(message)
            msg_type = payload.get("type", "")

            if msg_type == "warning":
                print(f"  ⚠  {payload.get('message', '')}")
                continue

            if msg_type == "complete":
                print("\n✅ Optimization complete!")
                meta = payload["meta"]
                print(f"   Final CL: {meta['final_cl']:.6f}")
                print(f"   Final CD: {meta['final_cd']:.6f}")
                print(f"   Final L/D: {meta['final_cl_cd']:.4f}")
                # Keep window open
                waiting = True
                while waiting:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            waiting = False
                break

            if msg_type != "iteration":
                continue

            # ---- Parse iteration data ----
            meta = payload["meta"]
            shape = payload["shape"]

            iteration = meta["iteration"]
            total = meta["total_iterations"]
            loss = meta["loss"]
            cl = meta["cl"]
            cd = meta["cd"]
            cl_cd = meta["cl_cd"]
            drag = meta["drag_force"]
            lift = meta["lift_force"]

            loss_history.append(loss)
            cl_cd_history.append(cl_cd)

            airfoil_x = np.array(shape["airfoil_x"])
            airfoil_y_upper = np.array(shape["airfoil_y_upper"])
            airfoil_y_lower = np.array(shape["airfoil_y_lower"])

            print(
                f"Iter {iteration:3d}/{total} | Loss: {loss:10.4f} | "
                f"CL: {cl:.6f} | CD: {cd:.6f} | L/D: {cl_cd:.4f}"
            )

            # ---- Draw ----
            screen.fill(BG_COLOR)

            # Coordinate ranges for mapping
            x_range = (-0.05, 1.05)
            y_range = (-0.15, 0.15)

            # Grid lines
            for gx in np.arange(0, 1.1, 0.1):
                sx, sy_top = world_to_screen(gx, y_range[1], x_range, y_range)
                _, sy_bot = world_to_screen(gx, y_range[0], x_range, y_range)
                pygame.draw.line(screen, GRID_COLOR, (sx, sy_top), (sx, sy_bot), 1)
            for gy in np.arange(-0.15, 0.16, 0.05):
                sx_l, sy = world_to_screen(x_range[0], gy, x_range, y_range)
                sx_r, _ = world_to_screen(x_range[1], gy, x_range, y_range)
                pygame.draw.line(screen, GRID_COLOR, (sx_l, sy), (sx_r, sy), 1)

            # Draw initial shape (if we have first iteration data to use as reference)
            if iteration == 1:
                # Save the first iteration shape as "initial"
                init_airfoil_x = airfoil_x.copy()
                init_y_upper = airfoil_y_upper.copy()
                init_y_lower = airfoil_y_lower.copy()

            if "init_airfoil_x" in dir():
                # Fill
                init_pts_upper = [
                    world_to_screen(init_airfoil_x[i], init_y_upper[i], x_range, y_range)
                    for i in range(len(init_airfoil_x))
                ]
                init_pts_lower = [
                    world_to_screen(init_airfoil_x[i], init_y_lower[i], x_range, y_range)
                    for i in range(len(init_airfoil_x))
                ]
                init_polygon = init_pts_upper + init_pts_lower[::-1]
                if len(init_polygon) > 2:
                    init_surf = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
                    pygame.draw.polygon(init_surf, (180, 60, 60, 50), init_polygon)
                    screen.blit(init_surf, (0, 0))
                    pygame.draw.lines(screen, INITIAL_OUTLINE, False, init_pts_upper, 1)
                    pygame.draw.lines(screen, INITIAL_OUTLINE, False, init_pts_lower, 1)

            # Draw current shape
            cur_pts_upper = [
                world_to_screen(airfoil_x[i], airfoil_y_upper[i], x_range, y_range)
                for i in range(len(airfoil_x))
            ]
            cur_pts_lower = [
                world_to_screen(airfoil_x[i], airfoil_y_lower[i], x_range, y_range)
                for i in range(len(airfoil_x))
            ]
            cur_polygon = cur_pts_upper + cur_pts_lower[::-1]
            if len(cur_polygon) > 2:
                cur_surf = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
                pygame.draw.polygon(cur_surf, (60, 180, 90, 60), cur_polygon)
                screen.blit(cur_surf, (0, 0))
                pygame.draw.lines(screen, CURRENT_OUTLINE, False, cur_pts_upper, 2)
                pygame.draw.lines(screen, CURRENT_OUTLINE, False, cur_pts_lower, 2)

            # ---- Metrics panel (bottom) ----
            y_metrics = WINDOW_HEIGHT - 90
            pygame.draw.line(
                screen, GRID_COLOR, (10, y_metrics - 5), (WINDOW_WIDTH - 10, y_metrics - 5), 1
            )

            # Title
            title = font_large.render(
                f"Iteration {iteration}/{total}", True, ACCENT_COLOR
            )
            screen.blit(title, (15, y_metrics))

            # Metrics row
            metrics_text = (
                f"CL: {cl:+.6f}  CD: {cd:+.6f}  L/D: {cl_cd:+.4f}  "
                f"Drag: {drag:.4e}  Loss: {loss:.4f}"
            )
            met_surf = font_small.render(metrics_text, True, TEXT_COLOR)
            screen.blit(met_surf, (15, y_metrics + 25))

            # Progress bar
            bar_x, bar_y, bar_w, bar_h = 15, y_metrics + 50, WINDOW_WIDTH - 30, 10
            pygame.draw.rect(screen, GRID_COLOR, (bar_x, bar_y, bar_w, bar_h))
            fill_w = int(bar_w * iteration / total)
            pygame.draw.rect(screen, ACCENT_COLOR, (bar_x, bar_y, fill_w, bar_h))

            # Legend
            legend_y = 10
            pygame.draw.line(screen, INITIAL_OUTLINE, (15, legend_y + 7), (35, legend_y + 7), 2)
            screen.blit(font_small.render("Initial", True, INITIAL_OUTLINE), (40, legend_y))
            pygame.draw.line(screen, CURRENT_OUTLINE, (130, legend_y + 7), (150, legend_y + 7), 2)
            screen.blit(font_small.render("Current", True, CURRENT_OUTLINE), (155, legend_y))

            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

    pygame.quit()


if __name__ == "__main__":
    asyncio.run(run_client())
