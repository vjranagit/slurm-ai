"""Tests for policy/engine.py — saturation_score and fairshare_weight."""
from __future__ import annotations

import pytest

from controller.config import ControllerConfig
from controller.policy.engine import PolicyEngine
from controller.tests.conftest import make_snapshot


@pytest.fixture
def cfg() -> ControllerConfig:
    return ControllerConfig(
        interval_sec=15,
        cooldown_sec=60,
        dry_run=True,
        max_jobs_floor=2,
        max_jobs_ceil=128,
        pressure_high=0.85,
        pressure_low=0.45,
        compose_file="",
        slurm_service="slurm",
        slurm_exec_mode="local",
        slurm_ssh_host="",
        slurm_ssh_user="",
        slurm_ssh_key_file="",
    )


@pytest.fixture
def engine(cfg: ControllerConfig) -> PolicyEngine:
    return PolicyEngine(cfg)


# ---------------------------------------------------------------------------
# saturation_score = max(cpu, mem, io)
# ---------------------------------------------------------------------------


def test_saturation_score_returns_max_of_three_pressures(engine: PolicyEngine) -> None:
    snap = make_snapshot(cpu_pressure=0.3, mem_pressure=0.7, io_pressure=0.5)
    assert engine.saturation_score(snap) == pytest.approx(0.7)


def test_saturation_score_uses_cpu_when_highest(engine: PolicyEngine) -> None:
    snap = make_snapshot(cpu_pressure=0.9, mem_pressure=0.1, io_pressure=0.1)
    assert engine.saturation_score(snap) == pytest.approx(0.9)


def test_saturation_score_uses_io_when_highest(engine: PolicyEngine) -> None:
    snap = make_snapshot(cpu_pressure=0.2, mem_pressure=0.3, io_pressure=0.8)
    assert engine.saturation_score(snap) == pytest.approx(0.8)


def test_saturation_score_when_all_pressures_equal(engine: PolicyEngine) -> None:
    snap = make_snapshot(cpu_pressure=0.5, mem_pressure=0.5, io_pressure=0.5)
    assert engine.saturation_score(snap) == pytest.approx(0.5)


def test_saturation_score_at_zero(engine: PolicyEngine) -> None:
    snap = make_snapshot(cpu_pressure=0.0, mem_pressure=0.0, io_pressure=0.0)
    assert engine.saturation_score(snap) == pytest.approx(0.0)


def test_saturation_score_at_one(engine: PolicyEngine) -> None:
    snap = make_snapshot(cpu_pressure=1.0, mem_pressure=1.0, io_pressure=1.0)
    assert engine.saturation_score(snap) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# fairshare_weight: high >= pressure_high -> 5000
# ---------------------------------------------------------------------------


def test_fairshare_weight_returns_5000_when_saturation_above_pressure_high(
    engine: PolicyEngine, cfg: ControllerConfig
) -> None:
    assert engine.fairshare_weight(cfg.pressure_high + 0.01) == 5000


def test_fairshare_weight_returns_5000_at_exact_pressure_high_boundary(
    engine: PolicyEngine, cfg: ControllerConfig
) -> None:
    assert engine.fairshare_weight(cfg.pressure_high) == 5000


def test_fairshare_weight_returns_5000_at_saturation_1_point_0(
    engine: PolicyEngine,
) -> None:
    assert engine.fairshare_weight(1.0) == 5000


# ---------------------------------------------------------------------------
# fairshare_weight: low <= pressure_low -> 1000
# ---------------------------------------------------------------------------


def test_fairshare_weight_returns_1000_when_saturation_below_pressure_low(
    engine: PolicyEngine, cfg: ControllerConfig
) -> None:
    assert engine.fairshare_weight(cfg.pressure_low - 0.01) == 1000


def test_fairshare_weight_returns_1000_at_exact_pressure_low_boundary(
    engine: PolicyEngine, cfg: ControllerConfig
) -> None:
    assert engine.fairshare_weight(cfg.pressure_low) == 1000


def test_fairshare_weight_returns_1000_at_saturation_0_point_0(
    engine: PolicyEngine,
) -> None:
    assert engine.fairshare_weight(0.0) == 1000


# ---------------------------------------------------------------------------
# fairshare_weight: mid-range -> 2500
# ---------------------------------------------------------------------------


def test_fairshare_weight_returns_2500_in_mid_range(
    engine: PolicyEngine, cfg: ControllerConfig
) -> None:
    mid = (cfg.pressure_low + cfg.pressure_high) / 2
    assert engine.fairshare_weight(mid) == 2500


def test_fairshare_weight_returns_2500_just_above_pressure_low(
    engine: PolicyEngine, cfg: ControllerConfig
) -> None:
    assert engine.fairshare_weight(cfg.pressure_low + 0.001) == 2500


def test_fairshare_weight_returns_2500_just_below_pressure_high(
    engine: PolicyEngine, cfg: ControllerConfig
) -> None:
    assert engine.fairshare_weight(cfg.pressure_high - 0.001) == 2500
