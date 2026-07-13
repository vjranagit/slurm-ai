"""Tests for apps/api/main.py — FastAPI endpoints with mocked collector."""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import controller.metrics  # ensure gauges are registered before generate_latest() is called  # noqa: F401
import pytest
from fastapi.testclient import TestClient

from controller.types import ClusterSnapshot


def make_fixed_snapshot() -> ClusterSnapshot:
    return ClusterSnapshot(
        timestamp=datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        total_nodes=8,
        idle_nodes=4,
        alloc_nodes=3,
        down_nodes=1,
        pending_jobs=10,
        running_jobs=5,
        completed_jobs_5m=3,
        cpu_pressure=0.667,
        mem_pressure=0.3,
        io_pressure=0.225,
        pending_reasons={"Resources": 6, "Priority": 4},
    )


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Create test client with collector.snapshot monkeypatched."""
    import apps.api.main as api_module

    snap = make_fixed_snapshot()
    mock_snapshot = MagicMock(return_value=snap)
    monkeypatch.setattr(api_module.collector, "snapshot", mock_snapshot)

    from fastapi.testclient import TestClient as TC

    return TC(api_module.app)


@pytest.fixture(autouse=True)
def _reset_rate_limit_state() -> None:
    """RateLimitMiddleware's window counters are a process-global dict (module-level,
    not per-TestClient), so every test in this file shares them via TestClient's
    fixed synthetic client host. Reset before and after each test so tests stay
    order-independent and one test's requests never count against another's limit.
    """
    import apps.api.main as api_module

    api_module._RATE_LIMIT_WINDOWS.clear()
    yield
    api_module._RATE_LIMIT_WINDOWS.clear()


# ---------------------------------------------------------------------------
# /healthz
# ---------------------------------------------------------------------------


def test_healthz_returns_200(client: TestClient) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200


def test_healthz_returns_status_ok(client: TestClient) -> None:
    response = client.get("/healthz")
    assert response.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# / (root)
# ---------------------------------------------------------------------------


def test_root_returns_200(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200


def test_root_contains_service_key(client: TestClient) -> None:
    data = client.get("/").json()
    assert "service" in data


def test_root_contains_health_key(client: TestClient) -> None:
    data = client.get("/").json()
    assert "health" in data


def test_root_contains_cluster_key(client: TestClient) -> None:
    data = client.get("/").json()
    assert "cluster" in data


def test_root_service_name_is_correct(client: TestClient) -> None:
    data = client.get("/").json()
    assert data["service"] == "adaptive-wlm-controller"


# ---------------------------------------------------------------------------
# /cluster/snapshot
# ---------------------------------------------------------------------------


def test_cluster_snapshot_returns_200(client: TestClient) -> None:
    response = client.get("/cluster/snapshot")
    assert response.status_code == 200


def test_cluster_snapshot_returns_json(client: TestClient) -> None:
    response = client.get("/cluster/snapshot")
    assert response.headers["content-type"].startswith("application/json")


def test_cluster_snapshot_has_total_nodes_field(client: TestClient) -> None:
    data = client.get("/cluster/snapshot").json()
    assert "total_nodes" in data


def test_cluster_snapshot_has_pending_jobs_field(client: TestClient) -> None:
    data = client.get("/cluster/snapshot").json()
    assert "pending_jobs" in data


def test_cluster_snapshot_has_running_jobs_field(client: TestClient) -> None:
    data = client.get("/cluster/snapshot").json()
    assert "running_jobs" in data


def test_cluster_snapshot_has_cpu_pressure_field(client: TestClient) -> None:
    data = client.get("/cluster/snapshot").json()
    assert "cpu_pressure" in data


def test_cluster_snapshot_has_mem_pressure_field(client: TestClient) -> None:
    data = client.get("/cluster/snapshot").json()
    assert "mem_pressure" in data


def test_cluster_snapshot_has_io_pressure_field(client: TestClient) -> None:
    data = client.get("/cluster/snapshot").json()
    assert "io_pressure" in data


def test_cluster_snapshot_has_pending_reasons_field(client: TestClient) -> None:
    data = client.get("/cluster/snapshot").json()
    assert "pending_reasons" in data


def test_cluster_snapshot_total_nodes_value_correct(client: TestClient) -> None:
    data = client.get("/cluster/snapshot").json()
    assert data["total_nodes"] == 8


def test_cluster_snapshot_pending_jobs_value_correct(client: TestClient) -> None:
    data = client.get("/cluster/snapshot").json()
    assert data["pending_jobs"] == 10


def test_cluster_snapshot_running_jobs_value_correct(client: TestClient) -> None:
    data = client.get("/cluster/snapshot").json()
    assert data["running_jobs"] == 5


def test_cluster_snapshot_uses_collector_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import apps.api.main as api_module
    from fastapi.testclient import TestClient as TC

    call_count = 0

    def counting_snapshot() -> ClusterSnapshot:
        nonlocal call_count
        call_count += 1
        return make_fixed_snapshot()

    monkeypatch.setattr(api_module.collector, "snapshot", counting_snapshot)
    tc = TC(api_module.app)
    tc.get("/cluster/snapshot")
    assert call_count == 1


# ---------------------------------------------------------------------------
# /metrics
# ---------------------------------------------------------------------------


def test_metrics_returns_200(client: TestClient) -> None:
    response = client.get("/metrics")
    assert response.status_code == 200


def test_metrics_content_type_is_prometheus_compatible(client: TestClient) -> None:
    response = client.get("/metrics")
    ct = response.headers.get("content-type", "")
    # prometheus text format or text/plain
    assert "text/" in ct or "application/" in ct


def test_metrics_body_contains_adaptive_metric_names(client: TestClient) -> None:
    response = client.get("/metrics")
    body = response.text
    assert "adaptive_" in body


def test_metrics_body_contains_adaptive_saturation_score(client: TestClient) -> None:
    response = client.get("/metrics")
    assert "adaptive_saturation_score" in response.text


def test_metrics_body_contains_adaptive_pending_jobs(client: TestClient) -> None:
    response = client.get("/metrics")
    assert "adaptive_pending_jobs" in response.text


# ---------------------------------------------------------------------------
# Optional bearer-token auth (CONTROLLER_API_TOKEN)
# ---------------------------------------------------------------------------


def test_no_token_set_snapshot_endpoint_stays_open(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Preserve existing behavior: unset CONTROLLER_API_TOKEN -> endpoints open."""
    monkeypatch.delenv("CONTROLLER_API_TOKEN", raising=False)
    response = client.get("/cluster/snapshot")
    assert response.status_code == 200


def test_no_token_set_metrics_endpoint_stays_open(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CONTROLLER_API_TOKEN", raising=False)
    response = client.get("/metrics")
    assert response.status_code == 200


def test_token_set_snapshot_without_header_returns_401(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CONTROLLER_API_TOKEN", "s3cret")
    response = client.get("/cluster/snapshot")
    assert response.status_code == 401


def test_token_set_snapshot_with_wrong_header_returns_401(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CONTROLLER_API_TOKEN", "s3cret")
    response = client.get(
        "/cluster/snapshot", headers={"Authorization": "Bearer wrong-token"}
    )
    assert response.status_code == 401


def test_token_set_snapshot_with_correct_header_returns_200(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CONTROLLER_API_TOKEN", "s3cret")
    response = client.get(
        "/cluster/snapshot", headers={"Authorization": "Bearer s3cret"}
    )
    assert response.status_code == 200


def test_token_set_metrics_without_header_returns_401(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CONTROLLER_API_TOKEN", "s3cret")
    response = client.get("/metrics")
    assert response.status_code == 401


def test_token_set_metrics_with_correct_header_returns_200(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CONTROLLER_API_TOKEN", "s3cret")
    response = client.get("/metrics", headers={"Authorization": "Bearer s3cret"})
    assert response.status_code == 200


def test_token_set_healthz_stays_open_always(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """/healthz and / must stay open even when a token is configured."""
    monkeypatch.setenv("CONTROLLER_API_TOKEN", "s3cret")
    response = client.get("/healthz")
    assert response.status_code == 200


def test_token_set_root_stays_open_always(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CONTROLLER_API_TOKEN", "s3cret")
    response = client.get("/")
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# RateLimitMiddleware (CONTROLLER_API_RATE_LIMIT_PER_MIN) — addresses the
# "no rate limiting" gap documented in README.md's "Security and deployment"
# section. Applies to every route (global middleware), keyed per client host.
# ---------------------------------------------------------------------------


def test_requests_within_limit_all_succeed(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CONTROLLER_API_RATE_LIMIT_PER_MIN", "5")
    for _ in range(5):
        response = client.get("/healthz")
        assert response.status_code == 200


def test_request_past_limit_returns_429(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CONTROLLER_API_RATE_LIMIT_PER_MIN", "3")
    for _ in range(3):
        assert client.get("/healthz").status_code == 200
    response = client.get("/healthz")
    assert response.status_code == 429


def test_429_response_includes_retry_after_header(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CONTROLLER_API_RATE_LIMIT_PER_MIN", "1")
    client.get("/healthz")
    response = client.get("/healthz")
    assert response.status_code == 429
    assert "Retry-After" in response.headers


def test_429_response_has_detail_field(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CONTROLLER_API_RATE_LIMIT_PER_MIN", "1")
    client.get("/healthz")
    response = client.get("/healthz")
    assert response.json() == {"detail": "rate limit exceeded"}


def test_rate_limit_is_global_across_all_routes_not_per_endpoint(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exhausting the limit on one route blocks a different route from the same client."""
    monkeypatch.setenv("CONTROLLER_API_RATE_LIMIT_PER_MIN", "2")
    assert client.get("/healthz").status_code == 200
    assert client.get("/").status_code == 200
    response = client.get("/cluster/snapshot")
    assert response.status_code == 429


def test_rate_limit_zero_disables_limiting(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CONTROLLER_API_RATE_LIMIT_PER_MIN", "0")
    for _ in range(25):
        assert client.get("/healthz").status_code == 200


def test_rate_limit_negative_disables_limiting(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CONTROLLER_API_RATE_LIMIT_PER_MIN", "-1")
    for _ in range(25):
        assert client.get("/healthz").status_code == 200


def test_rate_limit_garbage_env_falls_back_to_default_never_raises(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fail-safe like _parse_bool: a typo'd env var must not crash the app."""
    monkeypatch.setenv("CONTROLLER_API_RATE_LIMIT_PER_MIN", "not-a-number")
    response = client.get("/healthz")
    assert response.status_code == 200


def test_rate_limit_unset_uses_default_120_per_min(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CONTROLLER_API_RATE_LIMIT_PER_MIN", raising=False)
    for _ in range(10):
        assert client.get("/healthz").status_code == 200


def test_rate_limit_tracks_clients_independently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A client hitting its limit must not affect a different client's quota."""
    import apps.api.main as api_module
    from fastapi.testclient import TestClient as TC

    monkeypatch.setenv("CONTROLLER_API_RATE_LIMIT_PER_MIN", "1")
    snap = make_fixed_snapshot()
    monkeypatch.setattr(api_module.collector, "snapshot", MagicMock(return_value=snap))

    client_a = TC(api_module.app, client=("10.0.0.1", 12345))
    client_b = TC(api_module.app, client=("10.0.0.2", 12345))

    assert client_a.get("/healthz").status_code == 200
    assert client_a.get("/healthz").status_code == 429  # client A now over its limit

    # client B is a distinct source IP — must still have its own fresh quota.
    assert client_b.get("/healthz").status_code == 200


# ---------------------------------------------------------------------------
# CORS allowlist (CONTROLLER_API_ALLOWED_ORIGINS) — configure_cors().
# Closes the "no CORS policy" gap the README used to flag. CORSMiddleware
# captures the allowlist at construction, so each test sets the env, builds a
# fresh app + trivial route, and calls configure_cors() against it.
# ---------------------------------------------------------------------------


def _build_cors_app(monkeypatch: pytest.MonkeyPatch, origins_env: str | None) -> TestClient:
    import apps.api.main as api_module
    from fastapi import FastAPI
    from fastapi.testclient import TestClient as TC

    if origins_env is None:
        monkeypatch.delenv("CONTROLLER_API_ALLOWED_ORIGINS", raising=False)
    else:
        monkeypatch.setenv("CONTROLLER_API_ALLOWED_ORIGINS", origins_env)

    test_app = FastAPI()

    @test_app.get("/ping")
    def ping() -> dict[str, bool]:
        return {"ok": True}

    api_module.configure_cors(test_app)
    return TC(test_app)


# --- _allowed_origins() fail-safe parsing (mirrors _rate_limit_per_min style) ---


def test_allowed_origins_unset_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    import apps.api.main as api_module

    monkeypatch.delenv("CONTROLLER_API_ALLOWED_ORIGINS", raising=False)
    assert api_module._allowed_origins() == []


def test_allowed_origins_blank_and_comma_only_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import apps.api.main as api_module

    monkeypatch.setenv("CONTROLLER_API_ALLOWED_ORIGINS", "  , ,")
    assert api_module._allowed_origins() == []


def test_allowed_origins_strips_whitespace_and_drops_empty_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import apps.api.main as api_module

    monkeypatch.setenv(
        "CONTROLLER_API_ALLOWED_ORIGINS", " https://a.example ,, https://b.example "
    )
    assert api_module._allowed_origins() == ["https://a.example", "https://b.example"]


# --- Preflight (OPTIONS) behavior ---


def test_preflight_options_allowed_origin_returns_allow_origin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _build_cors_app(monkeypatch, "https://dash.example")
    resp = client.options(
        "/ping",
        headers={
            "Origin": "https://dash.example",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert resp.status_code == 200
    assert resp.headers["access-control-allow-origin"] == "https://dash.example"


def test_preflight_options_disallowed_origin_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A preflight from an origin not on the allowlist must be rejected (400) with
    no allow-origin echo — the browser will then block the real request."""
    client = _build_cors_app(monkeypatch, "https://dash.example")
    resp = client.options(
        "/ping",
        headers={
            "Origin": "https://evil.example",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert resp.status_code == 400
    assert "access-control-allow-origin" not in resp.headers


def test_preflight_allows_authorization_header(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cross-origin callers of a token-protected endpoint must be able to send
    the Authorization header, so it has to be in the preflight allow-headers."""
    client = _build_cors_app(monkeypatch, "https://dash.example")
    resp = client.options(
        "/ping",
        headers={
            "Origin": "https://dash.example",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "authorization",
        },
    )
    assert resp.status_code == 200
    assert "authorization" in resp.headers.get("access-control-allow-headers", "").lower()


# --- Simple (non-preflight) request behavior ---


def test_simple_get_allowed_origin_has_allow_origin_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _build_cors_app(monkeypatch, "https://dash.example")
    resp = client.get("/ping", headers={"Origin": "https://dash.example"})
    assert resp.status_code == 200
    assert resp.headers["access-control-allow-origin"] == "https://dash.example"


def test_simple_get_disallowed_origin_omits_allow_origin_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _build_cors_app(monkeypatch, "https://dash.example")
    resp = client.get("/ping", headers={"Origin": "https://evil.example"})
    # The request itself still succeeds server-side; the browser blocks the read
    # because no matching Access-Control-Allow-Origin header comes back.
    assert resp.status_code == 200
    assert "access-control-allow-origin" not in resp.headers


def test_multiple_origins_each_individually_allowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _build_cors_app(monkeypatch, "https://a.example, https://b.example")
    for origin in ("https://a.example", "https://b.example"):
        resp = client.get("/ping", headers={"Origin": origin})
        assert resp.headers["access-control-allow-origin"] == origin


# --- Same-origin default: unset/blank allowlist attaches no CORS at all ---


def test_unset_allowlist_emits_no_cors_header(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _build_cors_app(monkeypatch, None)
    resp = client.get("/ping", headers={"Origin": "https://dash.example"})
    assert resp.status_code == 200
    assert "access-control-allow-origin" not in resp.headers


def test_blank_allowlist_emits_no_cors_header(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _build_cors_app(monkeypatch, "   ")
    resp = client.get("/ping", headers={"Origin": "https://dash.example"})
    assert "access-control-allow-origin" not in resp.headers


# --- Safety invariant: never wildcard-origin WITH credentials ---


def test_wildcard_allowlist_never_allows_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A '*' allowlist may echo '*' but must NEVER also set
    Access-Control-Allow-Credentials: true (the CORS-spec-forbidden pairing)."""
    client = _build_cors_app(monkeypatch, "*")
    resp = client.get("/ping", headers={"Origin": "https://anything.example"})
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == "*"
    assert "access-control-allow-credentials" not in resp.headers


def test_specific_origin_never_allows_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _build_cors_app(monkeypatch, "https://dash.example")
    resp = client.get("/ping", headers={"Origin": "https://dash.example"})
    assert "access-control-allow-credentials" not in resp.headers
