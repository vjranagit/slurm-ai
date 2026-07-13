"""End-to-end API tests: boot the REAL server, speak REAL HTTP.

Everything in test_api.py runs in-process through Starlette's TestClient, which
(a) never exercises the module-level wiring in apps/api/main.py — the
    import-time ``configure_cors(app)`` / ``app.add_middleware(...)`` path that
    reads CONTROLLER_API_ALLOWED_ORIGINS et al. from the environment — and
(b) never proves the app actually BOOTS under uvicorn (broken imports, bad
    middleware ordering, or a missing static dir would still pass TestClient).

These tests close that gap: each fixture spawns ``python -m uvicorn
apps.api.main:app`` in a subprocess with a purpose-built environment and talks
to it over a real TCP socket. The Slurm CLI boundary is satisfied by STUB
``sinfo``/``squeue``/``sacct`` executables on the child's PATH with
SLURM_EXEC_MODE=local, so /cluster/snapshot exercises the entire pipeline
end-to-end: HTTP -> FastAPI -> SlurmCollector -> SlurmCommandRunner ->
``bash -lc`` -> stub binaries -> parser -> JSON. No real Slurm cluster is
touched.

Note on ``bash -lc``: a login shell sources /etc/profile, which on some distros
RESETS PATH (clobbering the env we pass the child). Each fixture therefore also
points HOME at a scratch dir whose ``.bash_profile`` re-prepends the stub dir,
so the stubs win regardless of the host's profile behavior.
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterator

import httpx
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BOOT_DEADLINE_SEC = 30.0
HTTP_TIMEOUT_SEC = 10.0

E2E_TOKEN = "e2e-test-token-not-a-real-secret"  # test-only value, never a live credential
ALLOWED_ORIGIN = "https://dash.example"
SECOND_ORIGIN = "https://ops.example"
DISALLOWED_ORIGIN = "https://evil.example"

# Stub scheduler output -> expected snapshot fields.
SINFO_LINES = "idle\nidle\nalloc\ndown\n"          # 4 nodes: 2 idle, 1 alloc, 1 down
SQUEUE_LINES = "PENDING|Resources\nPENDING|Priority\nRUNNING|None\n"  # 2 pending, 1 running
SACCT_LINES = "COMPLETED\nCOMPLETED\nFAILED\n"     # 2 completed in last 5m


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _write_stub(stub_dir: Path, name: str, body: str) -> None:
    path = stub_dir / name
    path.write_text(body)
    path.chmod(0o755)


def _make_slurm_stubs(stub_dir: Path, *, healthy: bool) -> None:
    """Create fake sinfo/squeue/sacct. healthy=False simulates a scheduler outage
    (non-zero exit + error on stderr), which the collector must surface."""
    stub_dir.mkdir(parents=True, exist_ok=True)
    if healthy:
        _write_stub(stub_dir, "sinfo", f"#!/bin/sh\nprintf '{SINFO_LINES}'\n")
        _write_stub(stub_dir, "squeue", f"#!/bin/sh\nprintf '{SQUEUE_LINES}'\n")
        _write_stub(stub_dir, "sacct", f"#!/bin/sh\nprintf '{SACCT_LINES}'\n")
    else:
        down = (
            "#!/bin/sh\n"
            "echo 'slurm_load_partitions: Unable to contact slurm controller' >&2\n"
            "exit 1\n"
        )
        for name in ("sinfo", "squeue", "sacct"):
            _write_stub(stub_dir, name, down)


def _hermetic_env(stub_dir: Path, fake_home: Path, **overrides: str) -> dict[str, str]:
    """Child env: outer CONTROLLER_*/SLURM_*/API_* knobs stripped, stubs on PATH,
    HOME pointed at a scratch dir whose .bash_profile re-prepends the stub dir."""
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("CONTROLLER_", "SLURM_", "API_"))
    }
    env["PATH"] = f"{stub_dir}{os.pathsep}{env.get('PATH', '')}"
    fake_home.mkdir(parents=True, exist_ok=True)
    (fake_home / ".bash_profile").write_text(
        f'export PATH="{stub_dir}:$PATH"\n'
    )
    env["HOME"] = str(fake_home)
    env["SLURM_EXEC_MODE"] = "local"
    env.update(overrides)
    return env


class LiveServer:
    def __init__(self, base_url: str, proc: subprocess.Popen, log_path: Path) -> None:
        self.base_url = base_url
        self.proc = proc
        self.log_path = log_path

    def client(self, **kwargs) -> httpx.Client:
        # trust_env=False: ignore any http_proxy/https_proxy in the outer env so
        # requests really hit 127.0.0.1 directly.
        return httpx.Client(
            base_url=self.base_url,
            trust_env=False,
            timeout=HTTP_TIMEOUT_SEC,
            follow_redirects=True,
            **kwargs,
        )


def _boot_server(tmp_dir: Path, *, healthy_stubs: bool, **env_overrides: str) -> LiveServer:
    stub_dir = tmp_dir / "stub-bin"
    fake_home = tmp_dir / "home"
    _make_slurm_stubs(stub_dir, healthy=healthy_stubs)
    env = _hermetic_env(stub_dir, fake_home, **env_overrides)

    port = _free_port()
    log_path = tmp_dir / "uvicorn.log"
    log_fh = open(log_path, "w")
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "apps.api.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        cwd=str(REPO_ROOT),  # StaticFiles("apps/dashboard") + module import are cwd-relative
        env=env,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
    )

    base_url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + BOOT_DEADLINE_SEC
    last_err: Exception | None = None
    with httpx.Client(trust_env=False, timeout=2.0) as probe:
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                break  # process died — fail fast with its log below
            try:
                if probe.get(f"{base_url}/healthz").status_code == 200:
                    return LiveServer(base_url, proc, log_path)
            except httpx.HTTPError as exc:  # not yet listening
                last_err = exc
            time.sleep(0.15)

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    log_tail = log_path.read_text()[-2000:] if log_path.exists() else "<no log>"
    raise RuntimeError(
        f"uvicorn failed to become healthy within {BOOT_DEADLINE_SEC}s "
        f"(last error: {last_err!r}); server log tail:\n{log_tail}"
    )


def _shutdown(server: LiveServer) -> None:
    server.proc.terminate()
    try:
        server.proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        server.proc.kill()
        server.proc.wait(timeout=10)


@pytest.fixture(scope="module")
def live_server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[LiveServer]:
    """Healthy stubs, CORS allowlist set, rate limiting disabled, no auth token."""
    server = _boot_server(
        tmp_path_factory.mktemp("e2e-live"),
        healthy_stubs=True,
        CONTROLLER_API_ALLOWED_ORIGINS=f"{ALLOWED_ORIGIN}, {SECOND_ORIGIN}",
        CONTROLLER_API_RATE_LIMIT_PER_MIN="0",  # deterministic: no test here trips a limit
    )
    yield server
    _shutdown(server)


@pytest.fixture(scope="module")
def secured_server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[LiveServer]:
    """Healthy stubs + bearer-token auth required; rate limiting disabled."""
    server = _boot_server(
        tmp_path_factory.mktemp("e2e-secured"),
        healthy_stubs=True,
        CONTROLLER_API_TOKEN=E2E_TOKEN,
        CONTROLLER_API_RATE_LIMIT_PER_MIN="0",
    )
    yield server
    _shutdown(server)


@pytest.fixture(scope="module")
def throttled_server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[LiveServer]:
    """Healthy stubs + a tiny rate limit so the 429 path is provable over HTTP."""
    server = _boot_server(
        tmp_path_factory.mktemp("e2e-throttled"),
        healthy_stubs=True,
        CONTROLLER_API_RATE_LIMIT_PER_MIN="3",
    )
    yield server
    _shutdown(server)


@pytest.fixture(scope="module")
def outage_server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[LiveServer]:
    """Stubs that FAIL like a dead slurmctld: /cluster/snapshot must 503, not 500."""
    server = _boot_server(
        tmp_path_factory.mktemp("e2e-outage"),
        healthy_stubs=False,
        CONTROLLER_API_RATE_LIMIT_PER_MIN="0",
    )
    yield server
    _shutdown(server)


# ---------------------------------------------------------------------------
# Boot + core endpoints (live server, real HTTP)
# ---------------------------------------------------------------------------


def test_e2e_server_boots_and_healthz_ok(live_server: LiveServer) -> None:
    with live_server.client() as client:
        resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_e2e_root_lists_endpoints(live_server: LiveServer) -> None:
    with live_server.client() as client:
        body = client.get("/").json()
    assert body["service"] == "adaptive-wlm-controller"
    assert body["cluster"] == "/cluster/snapshot"


def test_e2e_snapshot_full_pipeline_from_stub_slurm(live_server: LiveServer) -> None:
    """HTTP -> collector -> bash -lc -> stub sinfo/squeue/sacct -> parsed JSON."""
    with live_server.client() as client:
        resp = client.get("/cluster/snapshot")
    assert resp.status_code == 200
    snap = resp.json()
    assert snap["total_nodes"] == 4
    assert snap["idle_nodes"] == 2
    assert snap["alloc_nodes"] == 1
    assert snap["down_nodes"] == 1
    assert snap["pending_jobs"] == 2
    assert snap["running_jobs"] == 1
    assert snap["completed_jobs_5m"] == 2
    assert snap["pending_reasons"] == {"Resources": 1, "Priority": 1}
    assert abs(snap["cpu_pressure"] - 2 / 3) < 1e-6


def test_e2e_metrics_exposes_adaptive_gauges(live_server: LiveServer) -> None:
    with live_server.client() as client:
        resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "adaptive_saturation_score" in resp.text
    assert "adaptive_pending_jobs" in resp.text


def test_e2e_dashboard_served_at_ui(live_server: LiveServer) -> None:
    with live_server.client() as client:
        resp = client.get("/ui/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers.get("content-type", "")


def test_e2e_favicon_served(live_server: LiveServer) -> None:
    with live_server.client() as client:
        resp = client.get("/favicon.ico")
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# CORS over real HTTP — exercises the MODULE-LEVEL configure_cors(app) wiring
# (env read at import time in the child process), which TestClient tests
# structurally cannot reach.
# ---------------------------------------------------------------------------


def test_e2e_cors_preflight_allowed_origin(live_server: LiveServer) -> None:
    with live_server.client() as client:
        resp = client.options(
            "/cluster/snapshot",
            headers={
                "Origin": ALLOWED_ORIGIN,
                "Access-Control-Request-Method": "GET",
            },
        )
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == ALLOWED_ORIGIN
    # Safety invariant from the CORS hardening: never credentialed.
    assert "access-control-allow-credentials" not in resp.headers


def test_e2e_cors_preflight_second_origin_also_allowed(live_server: LiveServer) -> None:
    with live_server.client() as client:
        resp = client.options(
            "/cluster/snapshot",
            headers={
                "Origin": SECOND_ORIGIN,
                "Access-Control-Request-Method": "GET",
            },
        )
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == SECOND_ORIGIN


def test_e2e_cors_preflight_disallowed_origin_rejected(live_server: LiveServer) -> None:
    with live_server.client() as client:
        resp = client.options(
            "/cluster/snapshot",
            headers={
                "Origin": DISALLOWED_ORIGIN,
                "Access-Control-Request-Method": "GET",
            },
        )
    assert resp.status_code == 400
    assert "access-control-allow-origin" not in resp.headers


def test_e2e_cors_simple_get_allowed_origin_gets_header(live_server: LiveServer) -> None:
    with live_server.client() as client:
        resp = client.get("/healthz", headers={"Origin": ALLOWED_ORIGIN})
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == ALLOWED_ORIGIN


def test_e2e_cors_simple_get_disallowed_origin_no_header(live_server: LiveServer) -> None:
    with live_server.client() as client:
        resp = client.get("/healthz", headers={"Origin": DISALLOWED_ORIGIN})
    assert resp.status_code == 200  # simple requests still succeed server-side...
    assert "access-control-allow-origin" not in resp.headers  # ...but the browser gets no grant


# ---------------------------------------------------------------------------
# Bearer-token auth over real HTTP (import-time token read in the child env)
# ---------------------------------------------------------------------------


def test_e2e_auth_snapshot_requires_token(secured_server: LiveServer) -> None:
    with secured_server.client() as client:
        assert client.get("/cluster/snapshot").status_code == 401


def test_e2e_auth_metrics_requires_token(secured_server: LiveServer) -> None:
    with secured_server.client() as client:
        assert client.get("/metrics").status_code == 401


def test_e2e_auth_wrong_token_rejected(secured_server: LiveServer) -> None:
    with secured_server.client() as client:
        resp = client.get(
            "/cluster/snapshot",
            headers={"Authorization": "Bearer wrong-token"},
        )
    assert resp.status_code == 401


def test_e2e_auth_correct_token_serves_snapshot(secured_server: LiveServer) -> None:
    with secured_server.client() as client:
        resp = client.get(
            "/cluster/snapshot",
            headers={"Authorization": f"Bearer {E2E_TOKEN}"},
        )
    assert resp.status_code == 200
    assert resp.json()["pending_jobs"] == 2  # auth + full collector pipeline together


def test_e2e_auth_healthz_stays_open(secured_server: LiveServer) -> None:
    with secured_server.client() as client:
        assert client.get("/healthz").status_code == 200


# ---------------------------------------------------------------------------
# Rate limiting over real HTTP
# ---------------------------------------------------------------------------


def test_e2e_rate_limit_burst_gets_429_with_retry_after(
    throttled_server: LiveServer,
) -> None:
    """Limit is 3/min. A burst of 7 spans at most two fixed windows (2*3=6 grants),
    so at least one request MUST be throttled regardless of minute-boundary timing.
    """
    responses = []
    with throttled_server.client() as client:
        for _ in range(7):
            responses.append(client.get("/healthz"))
    throttled = [r for r in responses if r.status_code == 429]
    assert throttled, "burst of 7 against limit 3/min produced no 429"
    assert throttled[0].headers.get("Retry-After") == "60"
    assert throttled[0].json() == {"detail": "rate limit exceeded"}
    # And the ones that got through are genuine 200s, not errors.
    assert all(r.status_code in (200, 429) for r in responses)


# ---------------------------------------------------------------------------
# Scheduler outage over real HTTP — the structured-503 error handling
# ---------------------------------------------------------------------------


def test_e2e_scheduler_outage_returns_structured_503(outage_server: LiveServer) -> None:
    with outage_server.client() as client:
        resp = client.get("/cluster/snapshot")
    assert resp.status_code == 503
    assert resp.json() == {"detail": "cluster snapshot unavailable: scheduler query failed"}


def test_e2e_scheduler_outage_does_not_leak_cli_stderr(outage_server: LiveServer) -> None:
    """The stub prints 'Unable to contact slurm controller' on stderr; none of
    that internal detail may reach the HTTP client."""
    with outage_server.client() as client:
        resp = client.get("/cluster/snapshot")
    assert "slurm_load_partitions" not in resp.text
    assert "Unable to contact" not in resp.text


def test_e2e_scheduler_outage_healthz_still_ok(outage_server: LiveServer) -> None:
    """Scheduler down != API down: liveness endpoint keeps answering 200."""
    with outage_server.client() as client:
        assert client.get("/healthz").status_code == 200
