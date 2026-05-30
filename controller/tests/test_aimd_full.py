"""Comprehensive tests for tuner/aimd.py — all branches and bounds."""
from __future__ import annotations

import pytest

from controller.config import ControllerConfig
from controller.tuner.aimd import AimdTuner


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
def tuner(cfg: ControllerConfig) -> AimdTuner:
    return AimdTuner(cfg)


# ---------------------------------------------------------------------------
# DECREASE branch: saturation >= pressure_high -> current // 2
# ---------------------------------------------------------------------------


def test_decrease_halves_current_when_saturation_above_pressure_high(
    tuner: AimdTuner, cfg: ControllerConfig
) -> None:
    result = tuner.next_max_jobs(16, cfg.pressure_high + 0.01)
    assert result == 8


def test_decrease_at_exact_pressure_high_boundary(
    tuner: AimdTuner, cfg: ControllerConfig
) -> None:
    result = tuner.next_max_jobs(20, cfg.pressure_high)
    assert result == 10


def test_decrease_at_saturation_1_point_0(tuner: AimdTuner) -> None:
    result = tuner.next_max_jobs(32, 1.0)
    assert result == 16


def test_decrease_clamps_at_max_jobs_floor_when_halving_would_go_below(
    tuner: AimdTuner, cfg: ControllerConfig
) -> None:
    # current=3, floor=2; 3//2=1 < 2, must clamp to floor
    result = tuner.next_max_jobs(3, cfg.pressure_high)
    assert result == cfg.max_jobs_floor


def test_decrease_when_current_equals_floor_stays_at_floor(
    tuner: AimdTuner, cfg: ControllerConfig
) -> None:
    result = tuner.next_max_jobs(cfg.max_jobs_floor, cfg.pressure_high)
    assert result == cfg.max_jobs_floor


def test_decrease_odd_current_truncates_integer_division(
    tuner: AimdTuner, cfg: ControllerConfig
) -> None:
    result = tuner.next_max_jobs(7, cfg.pressure_high)
    assert result == 3  # 7 // 2 = 3


# ---------------------------------------------------------------------------
# INCREASE branch: saturation <= pressure_low -> current + 1
# ---------------------------------------------------------------------------


def test_increase_increments_by_one_when_saturation_below_pressure_low(
    tuner: AimdTuner, cfg: ControllerConfig
) -> None:
    result = tuner.next_max_jobs(8, cfg.pressure_low - 0.01)
    assert result == 9


def test_increase_at_exact_pressure_low_boundary(
    tuner: AimdTuner, cfg: ControllerConfig
) -> None:
    result = tuner.next_max_jobs(8, cfg.pressure_low)
    assert result == 9


def test_increase_at_saturation_0_point_0(tuner: AimdTuner) -> None:
    result = tuner.next_max_jobs(8, 0.0)
    assert result == 9


def test_increase_clamps_at_max_jobs_ceil_when_already_at_ceiling(
    tuner: AimdTuner, cfg: ControllerConfig
) -> None:
    result = tuner.next_max_jobs(cfg.max_jobs_ceil, cfg.pressure_low)
    assert result == cfg.max_jobs_ceil


def test_increase_clamps_at_max_jobs_ceil_when_one_below(
    tuner: AimdTuner, cfg: ControllerConfig
) -> None:
    result = tuner.next_max_jobs(cfg.max_jobs_ceil - 1, cfg.pressure_low)
    assert result == cfg.max_jobs_ceil


# ---------------------------------------------------------------------------
# HOLD branch: pressure_low < saturation < pressure_high -> unchanged
# ---------------------------------------------------------------------------


def test_hold_returns_current_unchanged_in_mid_range(
    tuner: AimdTuner, cfg: ControllerConfig
) -> None:
    mid = (cfg.pressure_low + cfg.pressure_high) / 2
    result = tuner.next_max_jobs(10, mid)
    assert result == 10


def test_hold_just_above_pressure_low(tuner: AimdTuner, cfg: ControllerConfig) -> None:
    result = tuner.next_max_jobs(15, cfg.pressure_low + 0.001)
    assert result == 15


def test_hold_just_below_pressure_high(tuner: AimdTuner, cfg: ControllerConfig) -> None:
    result = tuner.next_max_jobs(15, cfg.pressure_high - 0.001)
    assert result == 15


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_next_max_jobs_always_returns_int(tuner: AimdTuner, cfg: ControllerConfig) -> None:
    for sat in [0.0, cfg.pressure_low, cfg.pressure_high, 1.0]:
        result = tuner.next_max_jobs(10, sat)
        assert isinstance(result, int)
