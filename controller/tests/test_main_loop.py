"""Tests for main.py — loop hardening: run_once extraction + try/except + error counter.

controller/main.py's `run()` wraps each control-loop iteration in try/except so a
transient failure (e.g. collector.snapshot() raising) never crashes the process.
`run_once()` is the extracted, directly-testable single-iteration helper that the
while-loop calls.
"""
from __future__ import annotations

import pytest

from controller import main as main_module
from controller.actuator.slurm import SlurmActuator
from controller.config import ControllerConfig
from controller.metrics import LOOP_ERRORS
from controller.policy.engine import PolicyEngine
from controller.tuner import build_tuner


class _StopLoop(Exception):
    """Sentinel used to break out of main.run()'s infinite while loop in tests."""


def _make_cfg(**kwargs: object) -> ControllerConfig:
    defaults: dict[str, object] = dict(
        interval_sec=0,
        cooldown_sec=0,
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
    defaults.update(kwargs)
    return ControllerConfig(**defaults)  # type: ignore[arg-type]


class _BoomCollector:
    def snapshot(self) -> None:
        raise RuntimeError("boom")


# ---------------------------------------------------------------------------
# run_once() itself does not swallow — that's the while-loop's job
# ---------------------------------------------------------------------------


def test_run_once_propagates_exception_from_collector() -> None:
    cfg = _make_cfg()
    policy = PolicyEngine(cfg)
    tuner = build_tuner(cfg)
    actuator = SlurmActuator(cfg)

    with pytest.raises(RuntimeError, match="boom"):
        main_module.run_once(cfg, _BoomCollector(), policy, tuner, actuator)


# ---------------------------------------------------------------------------
# run()'s while loop swallows exceptions, increments LOOP_ERRORS, and continues
# ---------------------------------------------------------------------------


def test_run_loop_swallows_exception_increments_counter_and_continues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _make_cfg()

    call_state = {"n": 0}

    def fake_run_once(cfg, collector, policy, tuner, actuator):  # noqa: ANN001
        call_state["n"] += 1
        if call_state["n"] == 1:
            raise RuntimeError("simulated collector failure")
        return None

    sleep_calls = {"n": 0}

    def fake_sleep(seconds: float) -> None:
        sleep_calls["n"] += 1
        if sleep_calls["n"] >= 2:
            raise _StopLoop()

    monkeypatch.setattr(main_module, "run_once", fake_run_once)
    monkeypatch.setattr(main_module.time, "sleep", fake_sleep)
    monkeypatch.setattr(main_module, "start_http_server", lambda port: None)

    before = LOOP_ERRORS._value.get()

    with pytest.raises(_StopLoop):
        main_module.run(cfg)

    after = LOOP_ERRORS._value.get()
    assert after == before + 1, "LOOP_ERRORS must increment exactly once for the one failure"
    assert call_state["n"] == 2, "loop must retry after the failure (not crash/stop)"
    assert sleep_calls["n"] == 2, "loop must sleep after both the failed and successful iteration"


def test_run_loop_never_raises_the_swallowed_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """The RuntimeError from run_once must never propagate out of run()'s while loop."""
    cfg = _make_cfg()

    def always_fails(cfg, collector, policy, tuner, actuator):  # noqa: ANN001
        raise RuntimeError("persistent failure")

    sleep_calls = {"n": 0}

    def fake_sleep(seconds: float) -> None:
        sleep_calls["n"] += 1
        if sleep_calls["n"] >= 3:
            raise _StopLoop()

    monkeypatch.setattr(main_module, "run_once", always_fails)
    monkeypatch.setattr(main_module.time, "sleep", fake_sleep)
    monkeypatch.setattr(main_module, "start_http_server", lambda port: None)

    # Only _StopLoop (our sentinel) should escape — never RuntimeError.
    with pytest.raises(_StopLoop):
        main_module.run(cfg)
    assert sleep_calls["n"] == 3


# ---------------------------------------------------------------------------
# --dry-run CLI arg uses fail-safe _parse_bool (never accidentally goes live)
# ---------------------------------------------------------------------------


def test_dry_run_cli_arg_garbage_stays_true() -> None:
    from controller.config import _parse_bool

    cfg = _make_cfg(dry_run=False)
    # Simulate the __main__ block's override logic
    cfg.dry_run = _parse_bool("garbage-value", True)
    assert cfg.dry_run is True


def test_dry_run_cli_arg_explicit_false_goes_live() -> None:
    from controller.config import _parse_bool

    cfg = _make_cfg(dry_run=True)
    cfg.dry_run = _parse_bool("false", True)
    assert cfg.dry_run is False


# ---------------------------------------------------------------------------
# build_config_from_args() — CLI overrides mutate cfg *after* __post_init__ has
# already run, which used to silently bypass validation entirely (e.g.
# `--interval -5` would sail through construction, then crash `time.sleep(-5)`
# deep inside run()'s while loop instead of failing fast at startup). This
# helper re-validates after applying overrides so bad CLI input fails fast
# with a clear error, exactly like a bad env var does at construction time.
# ---------------------------------------------------------------------------


class _Args:
    def __init__(self, interval: int | None = None, dry_run: str | None = None) -> None:
        self.interval = interval
        self.dry_run = dry_run


def test_build_config_from_args_no_overrides_uses_env_defaults() -> None:
    cfg = main_module.build_config_from_args(_Args())
    assert cfg.interval_sec == 15  # default, no env override in this test process
    assert cfg.dry_run is True


def test_build_config_from_args_valid_interval_override_applies() -> None:
    cfg = main_module.build_config_from_args(_Args(interval=30))
    assert cfg.interval_sec == 30


def test_build_config_from_args_negative_interval_override_raises() -> None:
    """The bug this closes: a mutated-after-construction field must still be validated."""
    with pytest.raises(ValueError, match="interval_sec"):
        main_module.build_config_from_args(_Args(interval=-5))


def test_build_config_from_args_dry_run_override_garbage_stays_safe_true() -> None:
    cfg = main_module.build_config_from_args(_Args(dry_run="garbage"))
    assert cfg.dry_run is True


def test_build_config_from_args_dry_run_override_explicit_false_goes_live() -> None:
    cfg = main_module.build_config_from_args(_Args(dry_run="false"))
    assert cfg.dry_run is False
