from __future__ import annotations

import logging
import os
import secrets
import time
from dataclasses import asdict
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse, Response

from controller import metrics as _metrics  # noqa: F401  # registers adaptive_* gauges
from controller.collectors.slurm import SlurmCollector
from controller.config import ControllerConfig

LOG = logging.getLogger("adaptive-wlm-controller.api")

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

# Per-client-IP fixed-window request counters backing RateLimitMiddleware, keyed
# by client host -> (window_start_epoch_minute, count_in_window). Deliberately a
# module-level dict (not middleware-instance state): state is in-memory and
# per-process only — fine for the single-process deployment this MVP targets
# (see README "Security and deployment"), but NOT shared across multiple
# uvicorn workers/replicas if someone scales this out. That tradeoff is
# documented, not hidden.
_RATE_LIMIT_WINDOWS: dict[str, tuple[int, int]] = {}
_RATE_LIMIT_WINDOWS_MAX_ENTRIES = 10_000


def _rate_limit_per_min() -> int:
    """Fail-safe env parse: garbage/unset falls back to a sane default, never raises.

    Mirrors the _parse_bool fail-safe philosophy in controller/config.py: a typo'd
    env var must not silently disable protection or crash the process. <= 0 means
    "disabled" (explicit opt-out), matching CONTROLLER_API_TOKEN's empty-string
    opt-out convention for require_api_token.
    """
    raw = os.getenv("CONTROLLER_API_RATE_LIMIT_PER_MIN", "120")
    try:
        return int(raw)
    except ValueError:
        return 120


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Per-client, fixed-window rate limit applied to every request.

    Addresses the "no rate limiting" gap called out in README.md's "Security and
    deployment" section — the unauthenticated-by-default /cluster/snapshot
    endpoint shells out to sinfo/squeue/sacct on every call, so an unthrottled
    client can cheaply trigger repeated subprocess spawns (DoS). Also throttles
    brute-forcing CONTROLLER_API_TOKEN.

    Set CONTROLLER_API_RATE_LIMIT_PER_MIN=0 (or negative) to disable.
    """

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        limit = _rate_limit_per_min()
        if limit <= 0:
            return await call_next(request)

        client = request.client.host if request.client else "unknown"
        now_min = int(time.time() // 60)
        window_start, count = _RATE_LIMIT_WINDOWS.get(client, (now_min, 0))
        if window_start != now_min:
            window_start, count = now_min, 0
        count += 1
        _RATE_LIMIT_WINDOWS[client] = (window_start, count)

        # Opportunistic cleanup so long-lived processes with many distinct
        # clients don't grow this dict unbounded.
        if len(_RATE_LIMIT_WINDOWS) > _RATE_LIMIT_WINDOWS_MAX_ENTRIES:
            for key, (window, _count) in list(_RATE_LIMIT_WINDOWS.items()):
                if window != now_min:
                    del _RATE_LIMIT_WINDOWS[key]

        if count > limit:
            return JSONResponse(
                {"detail": "rate limit exceeded"},
                status_code=429,
                headers={"Retry-After": "60"},
            )
        return await call_next(request)


def _allowed_origins() -> list[str]:
    """Parse CONTROLLER_API_ALLOWED_ORIGINS (comma-separated) fail-safe.

    Mirrors the fail-safe env-parsing style used by _rate_limit_per_min / _parse_bool:
    never raises, and an unset/blank value yields the SAFE default — here an empty
    list, meaning same-origin only. Whitespace around each origin is stripped and
    empty entries are dropped, so "https://a.example ,, https://b.example" ->
    ["https://a.example", "https://b.example"].
    """
    raw = os.getenv("CONTROLLER_API_ALLOWED_ORIGINS", "")
    return [o.strip() for o in raw.split(",") if o.strip()]


def configure_cors(fastapi_app: FastAPI) -> None:
    """Attach CORS from the CONTROLLER_API_ALLOWED_ORIGINS allowlist, fail-safe.

    Closes the "no CORS policy" gap previously flagged in README's "Security and
    deployment" section.

    - Empty/unset allowlist -> NO CORSMiddleware is attached at all: no
      Access-Control-Allow-Origin header is ever emitted, so browsers block every
      cross-origin read. This is the safe same-origin-only default (the prior
      behavior), not a silent wildcard.
    - When set, only the listed origins are echoed back; any other Origin is
      rejected (Starlette answers a disallowed preflight with 400 and omits the
      allow-origin header on simple requests).
    - allow_credentials is hardcoded False. This API authenticates with a bearer
      token in the Authorization header, never cookies, so credentialed CORS is
      never needed — and pinning it False makes the forbidden "Allow-Origin: * with
      Allow-Credentials: true" combination unreachable even if an operator sets
      CONTROLLER_API_ALLOWED_ORIGINS=* (the CORS spec bans that pairing; browsers
      reject it, and it would broadcast credentialed responses to any site).

    Unlike the rate limiter, the allowlist is read once here at app construction
    (Starlette's CORSMiddleware captures it), not per request — origins are
    deployment config that does not change at runtime.
    """
    origins = _allowed_origins()
    if not origins:
        return
    fastapi_app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["GET", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
    )


app.add_middleware(RateLimitMiddleware)
# CORS is added after the rate limiter so it wraps outermost: a valid preflight is
# answered before the request reaches the rate limiter / route handlers.
configure_cors(app)

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
    """Live cluster snapshot; 503 (not a raw 500) when the scheduler is unreachable.

    The collector deliberately raises on Slurm CLI failure (fail-loud — see the
    harden/gaps-security "collector no longer swallows Slurm failures" fix), so a
    scheduler outage used to surface here as an unhandled exception: HTTP 500 with
    a traceback dumped to the server log and FastAPI's opaque "Internal Server
    Error" body. Map it to a structured 503 Service Unavailable instead — the
    semantically correct status for a dependency outage — with a GENERIC detail
    string. The underlying exception text (command lines, hostnames, stderr) is
    logged server-side only, never echoed to the client, so no scheduler
    internals leak through the unauthenticated-by-default endpoint.
    """
    try:
        snap = collector.snapshot()
    except Exception:
        LOG.exception("cluster snapshot failed: scheduler query error")
        raise HTTPException(
            status_code=503,
            detail="cluster snapshot unavailable: scheduler query failed",
        ) from None
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
