"""Shared fixtures for the slurm-ai test suite."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from controller.config import ControllerConfig
from controller.types import ClusterSnapshot


def make_snapshot(
    *,
    total_nodes: int = 4,
    idle_nodes: int = 2,
    alloc_nodes: int = 2,
    down_nodes: int = 0,
    pending_jobs: int = 5,
    running_jobs: int = 5,
    completed_jobs_5m: int = 3,
    cpu_pressure: float = 0.5,
    mem_pressure: float = 0.1,
    io_pressure: float = 0.075,
    pending_reasons: dict[str, int] | None = None,
) -> ClusterSnapshot:
    return ClusterSnapshot(
        timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
        total_nodes=total_nodes,
        idle_nodes=idle_nodes,
        alloc_nodes=alloc_nodes,
        down_nodes=down_nodes,
        pending_jobs=pending_jobs,
        running_jobs=running_jobs,
        completed_jobs_5m=completed_jobs_5m,
        cpu_pressure=cpu_pressure,
        mem_pressure=mem_pressure,
        io_pressure=io_pressure,
        pending_reasons=pending_reasons if pending_reasons is not None else {},
    )


@pytest.fixture
def default_cfg() -> ControllerConfig:
    """ControllerConfig with known defaults (no env side-effects)."""
    return ControllerConfig(
        interval_sec=15,
        cooldown_sec=60,
        dry_run=True,
        max_jobs_floor=2,
        max_jobs_ceil=128,
        pressure_high=0.85,
        pressure_low=0.45,
        compose_file="infra/docker/docker-compose.yml",
        slurm_service="slurm",
        slurm_exec_mode="local",
        slurm_ssh_host="",
        slurm_ssh_user="",
        slurm_ssh_key_file="",
    )
