"""Tests for actuator/slurm.py — apply(), dry_run, cooldown, bounds, changed flag."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from controller.actuator.slurm import SlurmActuator
from controller.config import ControllerConfig


def make_cfg(
    *,
    dry_run: bool = True,
    cooldown_sec: int = 60,
    max_jobs_floor: int = 2,
    max_jobs_ceil: int = 128,
) -> ControllerConfig:
    return ControllerConfig(
        interval_sec=15,
        cooldown_sec=cooldown_sec,
        dry_run=dry_run,
        max_jobs_floor=max_jobs_floor,
        max_jobs_ceil=max_jobs_ceil,
        pressure_high=0.85,
        pressure_low=0.45,
        compose_file="infra/docker/docker-compose.yml",
        slurm_service="slurm",
        slurm_exec_mode="local",
        slurm_ssh_host="",
        slurm_ssh_user="",
        slurm_ssh_key_file="",
    )


# ---------------------------------------------------------------------------
# dry_run=True: no runner calls, DRY_RUN entries in log
# ---------------------------------------------------------------------------


def test_dry_run_makes_no_runner_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    actuator = SlurmActuator(make_cfg(dry_run=True))
    called: list[str] = []

    def fake_run(command: str, check: bool = True) -> tuple[bool, str]:
        called.append(command)
        return True, ""

    monkeypatch.setattr(actuator.runner, "run", fake_run)
    actuator.apply(10, 2500)
    assert called == []


def test_dry_run_command_log_contains_dry_run_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    actuator = SlurmActuator(make_cfg(dry_run=True))
    action = actuator.apply(10, 2500)
    assert all(entry.startswith("DRY_RUN") for entry in action.command_log)


def test_dry_run_command_log_is_non_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    actuator = SlurmActuator(make_cfg(dry_run=True))
    action = actuator.apply(10, 2500)
    assert len(action.command_log) > 0


# ---------------------------------------------------------------------------
# live mode: runner called with sacctmgr and scontrol commands
# ---------------------------------------------------------------------------


def test_live_mode_calls_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    actuator = SlurmActuator(make_cfg(dry_run=False))
    called: list[str] = []

    def fake_run(command: str, check: bool = True) -> tuple[bool, str]:
        called.append(command)
        return True, "ok"

    monkeypatch.setattr(actuator.runner, "run", fake_run)
    actuator.apply(20, 5000)
    assert len(called) >= 1


def test_live_mode_command_log_contains_ok_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    actuator = SlurmActuator(make_cfg(dry_run=False))

    def fake_run(command: str, check: bool = True) -> tuple[bool, str]:
        return True, "done"

    monkeypatch.setattr(actuator.runner, "run", fake_run)
    action = actuator.apply(20, 5000)
    assert all(entry.startswith("OK") for entry in action.command_log)


def test_live_mode_commands_include_sacctmgr(monkeypatch: pytest.MonkeyPatch) -> None:
    actuator = SlurmActuator(make_cfg(dry_run=False))
    called: list[str] = []

    def fake_run(command: str, check: bool = True) -> tuple[bool, str]:
        called.append(command)
        return True, ""

    monkeypatch.setattr(actuator.runner, "run", fake_run)
    actuator.apply(20, 5000)
    assert any("sacctmgr" in c for c in called)


def test_live_mode_commands_include_scontrol(monkeypatch: pytest.MonkeyPatch) -> None:
    actuator = SlurmActuator(make_cfg(dry_run=False))
    called: list[str] = []

    def fake_run(command: str, check: bool = True) -> tuple[bool, str]:
        called.append(command)
        return True, ""

    monkeypatch.setattr(actuator.runner, "run", fake_run)
    actuator.apply(20, 5000)
    assert any("scontrol" in c for c in called)


# ---------------------------------------------------------------------------
# bounds clamping
# ---------------------------------------------------------------------------


def test_apply_clamps_target_above_ceil_to_ceil(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = make_cfg(dry_run=True, max_jobs_ceil=128)
    actuator = SlurmActuator(cfg)
    action = actuator.apply(999, 2500)
    assert action.new_max_jobs == 128


def test_apply_clamps_target_below_floor_to_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = make_cfg(dry_run=True, max_jobs_floor=2)
    actuator = SlurmActuator(cfg)
    action = actuator.apply(0, 2500)
    assert action.new_max_jobs == 2


def test_apply_exact_ceil_value_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = make_cfg(dry_run=True, max_jobs_ceil=128)
    actuator = SlurmActuator(cfg)
    action = actuator.apply(128, 2500)
    assert action.new_max_jobs == 128


def test_apply_exact_floor_value_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = make_cfg(dry_run=True, max_jobs_floor=2)
    actuator = SlurmActuator(cfg)
    action = actuator.apply(2, 2500)
    assert action.new_max_jobs == 2


# ---------------------------------------------------------------------------
# COOLDOWN: second apply within cooldown returns changed=False with no runner calls
# ---------------------------------------------------------------------------


def test_cooldown_second_apply_returns_changed_false(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = make_cfg(dry_run=True, cooldown_sec=3600)
    actuator = SlurmActuator(cfg)
    # First apply — succeeds
    actuator.apply(10, 2500)
    # Simulate last_apply just happened (very recent)
    actuator.last_apply = datetime.now(timezone.utc)
    # Second apply within cooldown
    action = actuator.apply(20, 5000)
    assert action.changed is False


def test_cooldown_second_apply_makes_no_runner_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = make_cfg(dry_run=False, cooldown_sec=3600)
    actuator = SlurmActuator(cfg)
    called: list[str] = []

    def fake_run(command: str, check: bool = True) -> tuple[bool, str]:
        called.append(command)
        return True, ""

    monkeypatch.setattr(actuator.runner, "run", fake_run)
    # First apply
    actuator.apply(10, 2500)
    called.clear()
    # Simulate last_apply just now
    actuator.last_apply = datetime.now(timezone.utc)
    # Second apply within cooldown
    actuator.apply(20, 5000)
    assert called == []


def test_cooldown_apply_after_cooldown_expires_proceeds(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = make_cfg(dry_run=True, cooldown_sec=10)
    actuator = SlurmActuator(cfg)
    actuator.apply(10, 2500)
    # Simulate last_apply 11 seconds ago
    actuator.last_apply = datetime.now(timezone.utc) - timedelta(seconds=11)
    action = actuator.apply(20, 5000)
    # Should proceed — no cooldown block
    assert action.command_log != ["cooldown: skip"]


def test_cooldown_command_log_contains_skip_marker(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = make_cfg(dry_run=True, cooldown_sec=3600)
    actuator = SlurmActuator(cfg)
    actuator.apply(10, 2500)
    actuator.last_apply = datetime.now(timezone.utc)
    action = actuator.apply(20, 5000)
    assert action.command_log == ["cooldown: skip"]


# ---------------------------------------------------------------------------
# changed flag
# ---------------------------------------------------------------------------


def test_changed_is_true_when_max_jobs_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = make_cfg(dry_run=True)
    actuator = SlurmActuator(cfg)
    # Initial state: max_jobs = max(floor=2, 8) = 8
    action = actuator.apply(20, 1000)  # definitely different from 8
    assert action.changed is True


def test_changed_is_false_when_nothing_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = make_cfg(dry_run=True, cooldown_sec=0)
    actuator = SlurmActuator(cfg)
    # First apply: set state to known values
    actuator.apply(10, 2500)
    # Expire cooldown
    actuator.last_apply = datetime.now(timezone.utc) - timedelta(seconds=1)
    # Apply same values again
    action = actuator.apply(10, 2500)
    assert action.changed is False


def test_changed_is_true_when_only_priority_weight_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = make_cfg(dry_run=True, cooldown_sec=0)
    actuator = SlurmActuator(cfg)
    actuator.apply(10, 2500)
    actuator.last_apply = datetime.now(timezone.utc) - timedelta(seconds=1)
    action = actuator.apply(10, 5000)
    assert action.changed is True


# ---------------------------------------------------------------------------
# old/new fields in returned action
# ---------------------------------------------------------------------------


def test_apply_returns_correct_old_and_new_max_jobs(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = make_cfg(dry_run=True)
    actuator = SlurmActuator(cfg)
    initial = actuator.state.max_jobs
    action = actuator.apply(initial + 5, 2500)
    assert action.old_max_jobs == initial
    assert action.new_max_jobs == initial + 5


# ---------------------------------------------------------------------------
# Fix 1 regression: state must NOT advance when live command fails
# ---------------------------------------------------------------------------


def test_live_command_failure_does_not_advance_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """State stays at old values when SlurmCommandRunner.run returns (False, 'err')."""
    cfg = make_cfg(dry_run=False, max_jobs_floor=2, max_jobs_ceil=128)
    actuator = SlurmActuator(cfg)
    initial_max = actuator.state.max_jobs
    initial_weight = actuator.state.priority_weight_fs

    def failing_run(command: str, check: bool = True) -> tuple[bool, str]:
        return False, "err: sacctmgr permission denied"

    monkeypatch.setattr(actuator.runner, "run", failing_run)
    action = actuator.apply(64, 5000)

    # State must be frozen at pre-apply values
    assert actuator.state.max_jobs == initial_max
    assert actuator.state.priority_weight_fs == initial_weight
    # Returned action reflects no change
    assert action.changed is False
    assert action.new_max_jobs == initial_max
    assert action.new_priority_weight_fs == initial_weight


def test_live_command_failure_does_not_update_last_apply(monkeypatch: pytest.MonkeyPatch) -> None:
    """last_apply timestamp must not advance when the command fails (cooldown stays reset-able)."""
    cfg = make_cfg(dry_run=False, cooldown_sec=3600)
    actuator = SlurmActuator(cfg)
    original_last = actuator.last_apply

    def failing_run(command: str, check: bool = True) -> tuple[bool, str]:
        return False, "err"

    monkeypatch.setattr(actuator.runner, "run", failing_run)
    actuator.apply(10, 2500)

    assert actuator.last_apply == original_last


def test_dry_run_still_simulates_state_advance(monkeypatch: pytest.MonkeyPatch) -> None:
    """dry_run must simulate new state (unchanged behavior)."""
    cfg = make_cfg(dry_run=True, max_jobs_floor=2, max_jobs_ceil=128)
    actuator = SlurmActuator(cfg)
    action = actuator.apply(50, 5000)
    assert actuator.state.max_jobs == 50
    assert actuator.state.priority_weight_fs == 5000
    assert action.changed is True


# ---------------------------------------------------------------------------
# Fix 2 regression: initial max_jobs must not exceed max_jobs_ceil
# ---------------------------------------------------------------------------


def test_init_clamps_max_jobs_when_ceil_below_8() -> None:
    """When max_jobs_ceil < 8, initial max_jobs must be clamped to ceil."""
    cfg = make_cfg(dry_run=True, max_jobs_floor=1, max_jobs_ceil=4)
    actuator = SlurmActuator(cfg)
    assert actuator.state.max_jobs <= cfg.max_jobs_ceil
    assert actuator.state.max_jobs == 4


def test_init_max_jobs_equals_8_when_ceil_above_8() -> None:
    """Normal case: ceil >= 8 means initial max_jobs = 8."""
    cfg = make_cfg(dry_run=True, max_jobs_floor=2, max_jobs_ceil=128)
    actuator = SlurmActuator(cfg)
    assert actuator.state.max_jobs == 8


def test_init_max_jobs_equals_floor_when_floor_above_8() -> None:
    """When floor > 8, initial max_jobs = floor (clamped up)."""
    cfg = make_cfg(dry_run=True, max_jobs_floor=20, max_jobs_ceil=128)
    actuator = SlurmActuator(cfg)
    assert actuator.state.max_jobs == 20
