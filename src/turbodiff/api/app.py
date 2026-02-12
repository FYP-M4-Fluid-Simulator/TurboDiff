from __future__ import annotations

from typing import Dict, Tuple

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from turbodiff.api import validation_server

FIDELITY_MAP: Dict[str, Tuple[int, int]] = {
    "low": (64, 128),
    "medium": (128, 256),
    "coarse": (256, 512),
}

app = FastAPI(title="TurboDiff Streaming API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(validation_server.router)
