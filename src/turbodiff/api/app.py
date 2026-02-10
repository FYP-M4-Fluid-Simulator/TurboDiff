import asyncio
import os
from typing import Dict, Tuple
from __future__ import annotations

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import jax.numpy as jnp
from turbodiff.core.fluid_grid_jax import FluidGrid

FIDELITY_MAP: Dict[str, Tuple[int, int]] = {
    "low": (64, 128),
    "medium": (128, 256),
    "coarse": (256, 512),
}

app = FastAPI(title="TurboDiff Streaming API")

