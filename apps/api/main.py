from __future__ import annotations

from dataclasses import asdict
from fastapi import FastAPI
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
)

app.mount("/ui", StaticFiles(directory="apps/dashboard", html=True), name="ui")


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


@app.get("/cluster/snapshot")
def cluster_snapshot() -> dict:
    snap = collector.snapshot()
    return asdict(snap)


@app.get("/metrics")
def metrics() -> Response:
    payload = generate_latest()
    return Response(content=payload, media_type=CONTENT_TYPE_LATEST)


@app.get("/favicon.ico")
def favicon() -> FileResponse:
    return FileResponse("apps/dashboard/favicon.ico")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("apps.api.main:app", host="0.0.0.0", port=8080, reload=False)
