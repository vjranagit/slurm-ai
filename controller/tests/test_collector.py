"""Tests for collectors/slurm.py — snapshot() with mocked runner."""
from __future__ import annotations

import pytest

from controller.collectors.slurm import SlurmCollector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SINFO_TYPICAL = "\n".join(["idle", "idle", "alloc", "down"])
SQUEUE_TYPICAL = "PENDING|Resources\nPENDING|Priority\nRUNNING|\nRUNNING|\nRUNNING|"
SACCT_TYPICAL = "COMPLETED\nCOMPLETED\nFAILED"


def make_collector() -> SlurmCollector:
    return SlurmCollector(
        compose_file="infra/docker/docker-compose.yml",
        slurm_service="slurm",
        exec_mode="local",
    )


def patch_runner(monkeypatch: pytest.MonkeyPatch, collector: SlurmCollector, responses: list[str]) -> None:
    """Replace runner.run to return successive canned outputs."""
    call_iter = iter(responses)

    def fake_run(command: str, check: bool = True) -> tuple[bool, str]:
        return True, next(call_iter)

    monkeypatch.setattr(collector.runner, "run", fake_run)


# ---------------------------------------------------------------------------
# Typical case: mixed node states and job queues
# ---------------------------------------------------------------------------


def test_snapshot_total_nodes_equals_sinfo_line_count(monkeypatch: pytest.MonkeyPatch) -> None:
    collector = make_collector()
    patch_runner(monkeypatch, collector, [SINFO_TYPICAL, SQUEUE_TYPICAL, SACCT_TYPICAL])
    snap = collector.snapshot()
    assert snap.total_nodes == 4


def test_snapshot_idle_nodes_counted_correctly(monkeypatch: pytest.MonkeyPatch) -> None:
    collector = make_collector()
    patch_runner(monkeypatch, collector, [SINFO_TYPICAL, SQUEUE_TYPICAL, SACCT_TYPICAL])
    snap = collector.snapshot()
    assert snap.idle_nodes == 2


def test_snapshot_alloc_nodes_counted_correctly(monkeypatch: pytest.MonkeyPatch) -> None:
    collector = make_collector()
    patch_runner(monkeypatch, collector, [SINFO_TYPICAL, SQUEUE_TYPICAL, SACCT_TYPICAL])
    snap = collector.snapshot()
    assert snap.alloc_nodes == 1


def test_snapshot_down_nodes_counted_correctly(monkeypatch: pytest.MonkeyPatch) -> None:
    collector = make_collector()
    patch_runner(monkeypatch, collector, [SINFO_TYPICAL, SQUEUE_TYPICAL, SACCT_TYPICAL])
    snap = collector.snapshot()
    assert snap.down_nodes == 1


def test_snapshot_pending_jobs_counted_correctly(monkeypatch: pytest.MonkeyPatch) -> None:
    collector = make_collector()
    patch_runner(monkeypatch, collector, [SINFO_TYPICAL, SQUEUE_TYPICAL, SACCT_TYPICAL])
    snap = collector.snapshot()
    assert snap.pending_jobs == 2


def test_snapshot_running_jobs_counted_correctly(monkeypatch: pytest.MonkeyPatch) -> None:
    collector = make_collector()
    patch_runner(monkeypatch, collector, [SINFO_TYPICAL, SQUEUE_TYPICAL, SACCT_TYPICAL])
    snap = collector.snapshot()
    assert snap.running_jobs == 3


def test_snapshot_completed_5m_counted_from_sacct(monkeypatch: pytest.MonkeyPatch) -> None:
    collector = make_collector()
    patch_runner(monkeypatch, collector, [SINFO_TYPICAL, SQUEUE_TYPICAL, SACCT_TYPICAL])
    snap = collector.snapshot()
    assert snap.completed_jobs_5m == 2


def test_snapshot_pending_reasons_aggregated(monkeypatch: pytest.MonkeyPatch) -> None:
    collector = make_collector()
    patch_runner(monkeypatch, collector, [SINFO_TYPICAL, SQUEUE_TYPICAL, SACCT_TYPICAL])
    snap = collector.snapshot()
    assert snap.pending_reasons == {"Resources": 1, "Priority": 1}


# ---------------------------------------------------------------------------
# Pressure math
# ---------------------------------------------------------------------------


def test_snapshot_cpu_pressure_equals_pending_over_total_jobs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collector = make_collector()
    patch_runner(monkeypatch, collector, [SINFO_TYPICAL, SQUEUE_TYPICAL, SACCT_TYPICAL])
    snap = collector.snapshot()
    # pending=2, running=3 -> 2/5 = 0.4
    assert snap.cpu_pressure == pytest.approx(2 / 5)


def test_snapshot_mem_pressure_formula(monkeypatch: pytest.MonkeyPatch) -> None:
    collector = make_collector()
    patch_runner(monkeypatch, collector, [SINFO_TYPICAL, SQUEUE_TYPICAL, SACCT_TYPICAL])
    snap = collector.snapshot()
    # min(1.0, (2 * 0.2) / max(1, 3 + 1)) = min(1.0, 0.4/4) = 0.1
    assert snap.mem_pressure == pytest.approx(0.1)


def test_snapshot_io_pressure_formula(monkeypatch: pytest.MonkeyPatch) -> None:
    collector = make_collector()
    patch_runner(monkeypatch, collector, [SINFO_TYPICAL, SQUEUE_TYPICAL, SACCT_TYPICAL])
    snap = collector.snapshot()
    # min(1.0, (2 * 0.15) / max(1, 3 + 1)) = min(1.0, 0.3/4) = 0.075
    assert snap.io_pressure == pytest.approx(0.075)


# ---------------------------------------------------------------------------
# Edge case: zero jobs — no division by zero
# ---------------------------------------------------------------------------


def test_snapshot_zero_jobs_no_division_by_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    collector = make_collector()
    patch_runner(monkeypatch, collector, ["idle\nidle", "", ""])
    snap = collector.snapshot()
    assert snap.pending_jobs == 0
    assert snap.running_jobs == 0
    assert snap.cpu_pressure == pytest.approx(0.0)
    assert snap.mem_pressure == pytest.approx(0.0)
    assert snap.io_pressure == pytest.approx(0.0)


def test_snapshot_zero_jobs_cpu_pressure_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    collector = make_collector()
    patch_runner(monkeypatch, collector, ["idle", "", ""])
    snap = collector.snapshot()
    assert snap.cpu_pressure == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Edge case: only pending jobs
# ---------------------------------------------------------------------------


def test_snapshot_only_pending_jobs_cpu_pressure_is_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collector = make_collector()
    squeue = "PENDING|Resources\nPENDING|Resources\nPENDING|Priority"
    patch_runner(monkeypatch, collector, ["idle", squeue, ""])
    snap = collector.snapshot()
    assert snap.pending_jobs == 3
    assert snap.running_jobs == 0
    # pending/(pending+running) = 3/3 = 1.0
    assert snap.cpu_pressure == pytest.approx(1.0)


def test_snapshot_only_pending_collects_all_reasons(monkeypatch: pytest.MonkeyPatch) -> None:
    collector = make_collector()
    squeue = "PENDING|Resources\nPENDING|Resources\nPENDING|Priority"
    patch_runner(monkeypatch, collector, ["idle", squeue, ""])
    snap = collector.snapshot()
    assert snap.pending_reasons["Resources"] == 2
    assert snap.pending_reasons["Priority"] == 1


# ---------------------------------------------------------------------------
# Edge case: only running jobs
# ---------------------------------------------------------------------------


def test_snapshot_only_running_jobs_cpu_pressure_is_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collector = make_collector()
    squeue = "RUNNING|\nRUNNING|\nRUNNING|"
    patch_runner(monkeypatch, collector, ["alloc\nalloc\nalloc", squeue, ""])
    snap = collector.snapshot()
    assert snap.pending_jobs == 0
    assert snap.running_jobs == 3
    assert snap.cpu_pressure == pytest.approx(0.0)


def test_snapshot_only_running_jobs_mem_and_io_pressure_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collector = make_collector()
    squeue = "RUNNING|\nRUNNING|"
    patch_runner(monkeypatch, collector, ["alloc\nalloc", squeue, ""])
    snap = collector.snapshot()
    # pending=0 -> numerator=0 -> pressures=0
    assert snap.mem_pressure == pytest.approx(0.0)
    assert snap.io_pressure == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Pressure cap at 1.0
# ---------------------------------------------------------------------------


def test_snapshot_mem_pressure_capped_at_1_point_0(monkeypatch: pytest.MonkeyPatch) -> None:
    collector = make_collector()
    # 1000 pending, 0 running => (1000*0.2)/max(1,0+1) = 200 -> capped 1.0
    squeue = "\n".join(["PENDING|Resources"] * 1000)
    patch_runner(monkeypatch, collector, ["idle", squeue, ""])
    snap = collector.snapshot()
    assert snap.mem_pressure == pytest.approx(1.0)


def test_snapshot_io_pressure_capped_at_1_point_0(monkeypatch: pytest.MonkeyPatch) -> None:
    collector = make_collector()
    squeue = "\n".join(["PENDING|Resources"] * 1000)
    patch_runner(monkeypatch, collector, ["idle", squeue, ""])
    snap = collector.snapshot()
    assert snap.io_pressure == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# runner failure raises RuntimeError
# ---------------------------------------------------------------------------


def test_snapshot_raises_runtime_error_when_runner_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collector = make_collector()

    def fake_run(command: str, check: bool = True) -> tuple[bool, str]:
        return False, "permission denied"

    monkeypatch.setattr(collector.runner, "run", fake_run)
    with pytest.raises(RuntimeError, match="permission denied"):
        collector.snapshot()
