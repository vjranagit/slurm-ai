"""Tests for config.py — defaults and env overrides via subprocess."""
from __future__ import annotations

import subprocess
import sys

import pytest

PYTHON = sys.executable


def run_cfg_expr(expr: str, env: dict[str, str] | None = None) -> str:
    """Run expr in a subprocess that imports ControllerConfig and prints a field."""
    import os

    base_env = {k: v for k, v in os.environ.items()}
    # Strip any controller env vars that could bleed in from parent process
    for key in [
        "CONTROLLER_INTERVAL_SEC", "CONTROLLER_COOLDOWN_SEC", "CONTROLLER_DRY_RUN",
        "MAX_JOBS_FLOOR", "MAX_JOBS_CEIL", "PRESSURE_HIGH", "PRESSURE_LOW",
        "SLURM_COMPOSE_FILE", "SLURM_SERVICE", "SLURM_EXEC_MODE",
        "SLURM_SSH_HOST", "SLURM_SSH_USER", "SLURM_SSH_KEY_FILE",
    ]:
        base_env.pop(key, None)
    if env:
        base_env.update(env)

    code = f"from controller.config import ControllerConfig; cfg = ControllerConfig(); print({expr})"
    result = subprocess.run(
        [PYTHON, "-c", code],
        capture_output=True,
        text=True,
        env=base_env,
        timeout=10,
    )
    assert result.returncode == 0, f"subprocess failed: {result.stderr}"
    return result.stdout.strip()


# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------


def test_default_interval_sec_is_15() -> None:
    assert run_cfg_expr("cfg.interval_sec") == "15"


def test_default_cooldown_sec_is_60() -> None:
    assert run_cfg_expr("cfg.cooldown_sec") == "60"


def test_default_dry_run_is_true() -> None:
    assert run_cfg_expr("cfg.dry_run") == "True"


def test_default_max_jobs_floor_is_2() -> None:
    assert run_cfg_expr("cfg.max_jobs_floor") == "2"


def test_default_max_jobs_ceil_is_128() -> None:
    assert run_cfg_expr("cfg.max_jobs_ceil") == "128"


def test_default_pressure_high_is_0_85() -> None:
    val = float(run_cfg_expr("cfg.pressure_high"))
    assert val == pytest.approx(0.85)


def test_default_pressure_low_is_0_45() -> None:
    val = float(run_cfg_expr("cfg.pressure_low"))
    assert val == pytest.approx(0.45)


def test_default_slurm_exec_mode_is_docker() -> None:
    assert run_cfg_expr("cfg.slurm_exec_mode") == "docker"


def test_default_slurm_service_is_slurm() -> None:
    assert run_cfg_expr("cfg.slurm_service") == "slurm"


def test_default_slurm_ssh_host_is_empty() -> None:
    assert run_cfg_expr("cfg.slurm_ssh_host") == ""


def test_default_slurm_ssh_user_is_empty() -> None:
    assert run_cfg_expr("cfg.slurm_ssh_user") == ""


def test_default_slurm_ssh_key_file_is_empty() -> None:
    assert run_cfg_expr("cfg.slurm_ssh_key_file") == ""


# ---------------------------------------------------------------------------
# Env overrides
# ---------------------------------------------------------------------------


def test_env_override_interval_sec() -> None:
    assert run_cfg_expr("cfg.interval_sec", {"CONTROLLER_INTERVAL_SEC": "30"}) == "30"


def test_env_override_cooldown_sec() -> None:
    assert run_cfg_expr("cfg.cooldown_sec", {"CONTROLLER_COOLDOWN_SEC": "120"}) == "120"


def test_env_override_dry_run_false() -> None:
    assert run_cfg_expr("cfg.dry_run", {"CONTROLLER_DRY_RUN": "false"}) == "False"


def test_env_override_dry_run_true_uppercase() -> None:
    assert run_cfg_expr("cfg.dry_run", {"CONTROLLER_DRY_RUN": "TRUE"}) == "True"


def test_env_override_max_jobs_floor() -> None:
    assert run_cfg_expr("cfg.max_jobs_floor", {"MAX_JOBS_FLOOR": "4"}) == "4"


def test_env_override_max_jobs_ceil() -> None:
    assert run_cfg_expr("cfg.max_jobs_ceil", {"MAX_JOBS_CEIL": "256"}) == "256"


def test_env_override_pressure_high() -> None:
    val = float(run_cfg_expr("cfg.pressure_high", {"PRESSURE_HIGH": "0.9"}))
    assert val == pytest.approx(0.9)


def test_env_override_pressure_low() -> None:
    val = float(run_cfg_expr("cfg.pressure_low", {"PRESSURE_LOW": "0.3"}))
    assert val == pytest.approx(0.3)


def test_env_override_slurm_exec_mode() -> None:
    assert run_cfg_expr("cfg.slurm_exec_mode", {"SLURM_EXEC_MODE": "local"}) == "local"


def test_env_override_slurm_service() -> None:
    assert run_cfg_expr("cfg.slurm_service", {"SLURM_SERVICE": "slurmctld"}) == "slurmctld"


def test_env_override_slurm_ssh_host() -> None:
    assert (
        run_cfg_expr("cfg.slurm_ssh_host", {"SLURM_SSH_HOST": "cluster.example.com"})
        == "cluster.example.com"
    )


def test_env_override_slurm_ssh_user() -> None:
    assert run_cfg_expr("cfg.slurm_ssh_user", {"SLURM_SSH_USER": "admin"}) == "admin"


def test_env_override_slurm_ssh_key_file() -> None:
    assert (
        run_cfg_expr("cfg.slurm_ssh_key_file", {"SLURM_SSH_KEY_FILE": "/home/user/.ssh/id_rsa"})
        == "/home/user/.ssh/id_rsa"
    )


# ---------------------------------------------------------------------------
# _parse_bool — fail-safe dry_run parsing (unrecognized input -> default, never raises)
# ---------------------------------------------------------------------------


def test_parse_bool_true_values() -> None:
    from controller.config import _parse_bool

    for v in ["true", "TRUE", " true ", "1", "yes", "on", "ON", "Yes"]:
        assert _parse_bool(v, False) is True, f"expected True for {v!r}"


def test_parse_bool_false_values() -> None:
    from controller.config import _parse_bool

    for v in ["false", "FALSE", " false ", "0", "no", "off", "OFF", "No"]:
        assert _parse_bool(v, True) is False, f"expected False for {v!r}"


def test_parse_bool_garbage_returns_default_true() -> None:
    from controller.config import _parse_bool

    for v in ["ture", "", "true1", "\n", "maybe", "yesno"]:
        assert _parse_bool(v, True) is True, f"expected default True for garbage {v!r}"


def test_parse_bool_garbage_returns_default_false() -> None:
    from controller.config import _parse_bool

    for v in ["ture", "", "true1", "\n", "maybe"]:
        assert _parse_bool(v, False) is False, f"expected default False for garbage {v!r}"


def test_parse_bool_never_raises_on_arbitrary_input() -> None:
    from controller.config import _parse_bool

    weird_inputs = ["🎉", "None", "null", "  ", "\t\n", "TrUe1", "0x1"]
    for v in weird_inputs:
        # Must not raise regardless of default
        assert _parse_bool(v, True) in (True, False)
        assert _parse_bool(v, False) in (True, False)


# ---------------------------------------------------------------------------
# dry_run via CONTROLLER_DRY_RUN env — fail-open/fail-safe behavior end to end
# ---------------------------------------------------------------------------


def test_env_dry_run_true_variants_stay_dry_run() -> None:
    for v in ["true", "TRUE", " true ", "1", "yes"]:
        assert run_cfg_expr("cfg.dry_run", {"CONTROLLER_DRY_RUN": v}) == "True"


def test_env_dry_run_false_variants_go_live() -> None:
    for v in ["false", "0", "no"]:
        assert run_cfg_expr("cfg.dry_run", {"CONTROLLER_DRY_RUN": v}) == "False"


def test_env_dry_run_garbage_defaults_to_safe_true() -> None:
    """Garbage/unknown CONTROLLER_DRY_RUN must never accidentally go live."""
    for v in ["ture", "", "true1", "garbage"]:
        assert run_cfg_expr("cfg.dry_run", {"CONTROLLER_DRY_RUN": v}) == "True"


# ---------------------------------------------------------------------------
# exec_timeout_sec / ssh_strict_host_key defaults
# ---------------------------------------------------------------------------


def test_default_exec_timeout_sec_is_30() -> None:
    assert run_cfg_expr("cfg.exec_timeout_sec") == "30"


def test_env_override_exec_timeout_sec() -> None:
    assert run_cfg_expr("cfg.exec_timeout_sec", {"SLURM_EXEC_TIMEOUT_SEC": "5"}) == "5"


def test_default_ssh_strict_host_key_is_true() -> None:
    assert run_cfg_expr("cfg.ssh_strict_host_key") == "True"


def test_env_override_ssh_strict_host_key_false() -> None:
    assert (
        run_cfg_expr("cfg.ssh_strict_host_key", {"SLURM_SSH_STRICT_HOST_KEY": "false"}) == "False"
    )


# ---------------------------------------------------------------------------
# floor <= ceil validation (__post_init__)
# ---------------------------------------------------------------------------


def test_floor_greater_than_ceil_raises_value_error() -> None:
    from controller.config import ControllerConfig

    with pytest.raises(ValueError, match="max_jobs_floor"):
        ControllerConfig(max_jobs_floor=200, max_jobs_ceil=128)


def test_floor_equal_to_ceil_is_allowed() -> None:
    from controller.config import ControllerConfig

    cfg = ControllerConfig(max_jobs_floor=10, max_jobs_ceil=10)
    assert cfg.max_jobs_floor == cfg.max_jobs_ceil == 10


def test_normal_floor_ceil_does_not_raise() -> None:
    from controller.config import ControllerConfig

    cfg = ControllerConfig(max_jobs_floor=2, max_jobs_ceil=128)
    assert cfg.max_jobs_floor == 2
    assert cfg.max_jobs_ceil == 128
