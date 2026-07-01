from __future__ import annotations

import os
import secrets
from dataclasses import asdict
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import Response

from controller import metrics as _metrics  # noqa: F401  # registers adaptive_* gauges
from controller.collectors.slurm import SlurmCollector
from controller.config import ControllerConfig

app = FastAPI(title="adaptive-wlm-controller", version="0.1.0")
cfg = ControllerConfig()
collector = SlurmCollector(
    cfg.compose_file,
    cfg.slurm_service,
    exec_mode=cfg.slurm_exec_mode,
    ssh_host=cfg.slurm_ssh_host,
    ssh_user=cfg.slurm_ssh_user,
    ssh_key_file=cfg.slurm_ssh_key_file,
    exec_timeout_sec=cfg.exec_timeout_sec,
    ssh_strict_host_key=cfg.ssh_strict_host_key,
)

app.mount("/ui", StaticFiles(directory="apps/dashboard", html=True), name="ui")


def require_api_token(request: Request) -> None:
    """Optional bearer-token auth for data endpoints.

    If CONTROLLER_API_TOKEN is unset/empty, this is a no-op (endpoints stay open,
    preserving current behavior). If set, requests must carry a matching
    ``Authorization: Bearer <token>`` header or get a 401.
    """
    token = os.getenv("CONTROLLER_API_TOKEN", "")
    if not token:
        return
    auth = request.headers.get("Authorization", "")
    scheme, _, provided = auth.partition(" ")
    if scheme.lower() != "bearer" or not secrets.compare_digest(provided, token):
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "adaptive-wlm-controller",
        "dashboard": "/ui",
        "health": "/healthz",
        "cluster": "/cluster/snapshot",
    }


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/cluster/snapshot", dependencies=[Depends(require_api_token)])
def cluster_snapshot() -> dict:
    snap = collector.snapshot()
    return asdict(snap)


@app.get("/metrics", dependencies=[Depends(require_api_token)])
def metrics() -> Response:
    payload = generate_latest()
    return Response(content=payload, media_type=CONTENT_TYPE_LATEST)


@app.get("/favicon.ico")
def favicon() -> FileResponse:
    return FileResponse("apps/dashboard/favicon.ico")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "apps.api.main:app",
        host=os.getenv("API_BIND_HOST", "127.0.0.1"),
        port=8080,
        reload=False,
    )
