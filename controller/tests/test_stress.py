"""Stress / property tests for tuner + actuator — invariants over many iterations."""
from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

import pytest

from controller.actuator.slurm import SlurmActuator
from controller.config import ControllerConfig
from controller.tuner.aimd import AimdTuner


def make_cfg(
    *,
    dry_run: bool = True,
    cooldown_sec: int = 0,
    max_jobs_floor: int = 2,
    max_jobs_ceil: int = 128,
    pressure_high: float = 0.85,
    pressure_low: float = 0.45,
) -> ControllerConfig:
    return ControllerConfig(
        interval_sec=15,
        cooldown_sec=cooldown_sec,
        dry_run=dry_run,
        max_jobs_floor=max_jobs_floor,
        max_jobs_ceil=max_jobs_ceil,
        pressure_high=pressure_high,
        pressure_low=pressure_low,
        compose_file="",
        slurm_service="slurm",
        slurm_exec_mode="local",
        slurm_ssh_host="",
        slurm_ssh_user="",
        slurm_ssh_key_file="",
    )


ITERATIONS = 2000
SEED = 42


# ---------------------------------------------------------------------------
# Invariant: tuner never returns a value outside [floor, ceil]
# ---------------------------------------------------------------------------


def test_tuner_output_always_within_bounds_over_many_iterations() -> None:
    cfg = make_cfg()
    tuner = AimdTuner(cfg)
    rng = random.Random(SEED)
    current = 8
    for _ in range(ITERATIONS):
        sat = rng.random()
        nxt = tuner.next_max_jobs(current, sat)
        assert cfg.max_jobs_floor <= nxt <= cfg.max_jobs_ceil, (
            f"tuner returned {nxt} outside [{cfg.max_jobs_floor}, {cfg.max_jobs_ceil}]"
        )
        current = nxt


def test_tuner_output_is_always_int_type() -> None:
    cfg = make_cfg()
    tuner = AimdTuner(cfg)
    rng = random.Random(SEED)
    current = 8
    for _ in range(ITERATIONS):
        sat = rng.random()
        nxt = tuner.next_max_jobs(current, sat)
        assert isinstance(nxt, int), f"tuner returned non-int: {type(nxt)}"
        current = nxt


def test_tuner_never_raises_exception_over_many_iterations() -> None:
    cfg = make_cfg()
    tuner = AimdTuner(cfg)
    rng = random.Random(SEED)
    current = 8
    for _ in range(ITERATIONS):
        sat = rng.random()
        nxt = tuner.next_max_jobs(current, sat)
        current = nxt


# ---------------------------------------------------------------------------
# Invariant: actuator max_jobs stays within [floor, ceil] every iteration
# ---------------------------------------------------------------------------


def test_actuator_max_jobs_stays_within_bounds_over_many_iterations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = make_cfg(cooldown_sec=0)
    actuator = SlurmActuator(cfg)
    rng = random.Random(SEED)

    for _ in range(ITERATIONS):
        target = rng.randint(0, 200)
        weight = rng.choice([1000, 2500, 5000])
        # Expire cooldown for each iteration
        actuator.last_apply = datetime.now(timezone.utc) - timedelta(seconds=1)
        actuator.apply(target, weight)
        assert cfg.max_jobs_floor <= actuator.state.max_jobs <= cfg.max_jobs_ceil, (
            f"actuator max_jobs={actuator.state.max_jobs} outside "
            f"[{cfg.max_jobs_floor}, {cfg.max_jobs_ceil}]"
        )


# ---------------------------------------------------------------------------
# Invariant: cooldown is respected — second apply immediately after first is blocked
# ---------------------------------------------------------------------------


def test_cooldown_blocks_second_apply_immediately_after_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify: after an apply, the NEXT call within cooldown always returns 'cooldown: skip'."""
    cooldown_sec = 3600
    cfg = make_cfg(cooldown_sec=cooldown_sec)
    actuator = SlurmActuator(cfg)
    rng = random.Random(SEED)

    blocked_count = 0
    total_attempts = 500

    for _ in range(total_attempts):
        # Expire cooldown: set last_apply to far in the past
        actuator.last_apply = datetime.now(timezone.utc) - timedelta(seconds=cooldown_sec + 1)
        # First apply always succeeds (cooldown expired)
        target = rng.randint(2, 128)
        weight = rng.choice([1000, 2500, 5000])
        action1 = actuator.apply(target, weight)
        assert action1.command_log != ["cooldown: skip"], "First apply should not be blocked"

        # Second apply immediately — within cooldown window
        action2 = actuator.apply(target + 1, weight)
        if action2.command_log == ["cooldown: skip"]:
            blocked_count += 1

    # Every second immediate apply must be blocked
    assert blocked_count == total_attempts, (
        f"Expected {total_attempts} blocked applies, got {blocked_count}"
    )


# ---------------------------------------------------------------------------
# No exceptions over many iterations (combined tuner + actuator)
# ---------------------------------------------------------------------------


def test_no_exceptions_in_combined_tuner_actuator_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = make_cfg(cooldown_sec=0)
    tuner = AimdTuner(cfg)
    actuator = SlurmActuator(cfg)
    rng = random.Random(SEED)
    current = 8

    for _ in range(ITERATIONS):
        sat = rng.random()
        nxt = tuner.next_max_jobs(current, sat)
        weight = rng.choice([1000, 2500, 5000])
        actuator.last_apply = datetime.now(timezone.utc) - timedelta(seconds=1)
        actuator.apply(nxt, weight)
        current = nxt


# ---------------------------------------------------------------------------
# Convergence: sustained high saturation trends max_jobs DOWN to floor
# ---------------------------------------------------------------------------


def test_sustained_high_saturation_trends_max_jobs_to_floor() -> None:
    cfg = make_cfg(max_jobs_floor=2, max_jobs_ceil=128, pressure_high=0.85)
    tuner = AimdTuner(cfg)
    current = 128
    # Under sustained high saturation, repeated halving must reach floor
    for _ in range(100):  # 128 -> 64 -> 32 -> ... -> 2 takes ~6 steps, 100 is plenty
        current = tuner.next_max_jobs(current, 1.0)
    assert current == cfg.max_jobs_floor


def test_sustained_high_saturation_max_jobs_monotonically_non_increasing() -> None:
    cfg = make_cfg()
    tuner = AimdTuner(cfg)
    current = 64
    prev = current
    for _ in range(20):
        nxt = tuner.next_max_jobs(current, 1.0)
        assert nxt <= prev, f"max_jobs increased under high saturation: {prev} -> {nxt}"
        prev = nxt
        current = nxt


# ---------------------------------------------------------------------------
# Convergence: sustained low saturation trends max_jobs UP to ceil
# ---------------------------------------------------------------------------


def test_sustained_low_saturation_trends_max_jobs_to_ceil() -> None:
    cfg = make_cfg(max_jobs_floor=2, max_jobs_ceil=128, pressure_low=0.45)
    tuner = AimdTuner(cfg)
    current = 2
    # Under sustained low saturation, +1 per step needs 126 iterations to reach ceil
    for _ in range(130):
        current = tuner.next_max_jobs(current, 0.0)
    assert current == cfg.max_jobs_ceil


def test_sustained_low_saturation_max_jobs_monotonically_non_decreasing() -> None:
    cfg = make_cfg()
    tuner = AimdTuner(cfg)
    current = 10
    prev = current
    for _ in range(20):
        nxt = tuner.next_max_jobs(current, 0.0)
        assert nxt >= prev, f"max_jobs decreased under low saturation: {prev} -> {nxt}"
        prev = nxt
        current = nxt


# ---------------------------------------------------------------------------
# Custom floor/ceil bounds are respected
# ---------------------------------------------------------------------------


def test_custom_floor_and_ceil_bounds_respected() -> None:
    cfg = make_cfg(max_jobs_floor=5, max_jobs_ceil=50)
    tuner = AimdTuner(cfg)
    rng = random.Random(SEED)
    current = 25
    for _ in range(ITERATIONS):
        sat = rng.random()
        nxt = tuner.next_max_jobs(current, sat)
        assert 5 <= nxt <= 50
        current = nxt
