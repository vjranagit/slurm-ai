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
