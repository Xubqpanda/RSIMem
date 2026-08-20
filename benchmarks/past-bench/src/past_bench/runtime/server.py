"""HTTP server for the generic decoupled runtime container."""

from __future__ import annotations

import os

import uvicorn
from fastapi import FastAPI

from .manager import RuntimeSessionManager
from .protocol import (
    BootstrapRequest,
    CloseSessionRequest,
    InterruptRequest,
    StartSessionRequest,
    StepRequest,
)


def _make_manager() -> RuntimeSessionManager:
    return RuntimeSessionManager(
        registry_path=os.environ.get("PAST_BENCH_AGENT_REGISTRY"),
        cache_dir=os.environ.get("PAST_BENCH_RUNTIME_CACHE"),
    )


app = FastAPI(title="past-bench runtime")
_MANAGER = _make_manager()


@app.get("/health")
def health():
    return _MANAGER.health().model_dump()


@app.post("/bootstrap")
def bootstrap(req: BootstrapRequest):
    return _MANAGER.bootstrap(req).model_dump()


@app.post("/start_session")
def start_session(req: StartSessionRequest):
    return _MANAGER.start_session(req).model_dump()


@app.post("/step")
def step(req: StepRequest):
    return _MANAGER.step(req).model_dump(mode="json")


@app.post("/interrupt")
def interrupt(req: InterruptRequest):
    _MANAGER.interrupt(req)
    return {"ok": True}


@app.post("/close_session")
def close_session(req: CloseSessionRequest):
    _MANAGER.close_session(req)
    return {"ok": True}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="past-bench runtime server")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)

