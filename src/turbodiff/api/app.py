from __future__ import annotations

from typing import Dict, Tuple

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware

from turbodiff.api import (
    streaming_server,
    cst_routes,
    optimization_server,
)
from turbodiff.api.auth import get_current_user
from turbodiff.db.storage import configure_storage_from_env

FIDELITY_MAP: Dict[str, Tuple[int, int]] = {
    "low": (64, 128),
    "medium": (128, 256),
    "coarse": (256, 512),
}

app = FastAPI(
    title="TurboDiff Streaming API",
    dependencies=[Depends(get_current_user)]
)


@app.on_event("startup")
def configure_storage() -> None:
    configure_storage_from_env()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(cst_routes.router)
app.include_router(streaming_server.router)
app.include_router(optimization_server.router)
