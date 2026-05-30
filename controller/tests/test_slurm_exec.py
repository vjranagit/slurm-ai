"""Tests for slurm_exec.py — _build command strings and run() behavior."""
from __future__ import annotations

import shlex

import pytest

from controller.slurm_exec import SlurmCommandRunner, SlurmExecConfig


def make_runner(
    mode: str,
    compose_file: str = "infra/docker/docker-compose.yml",
    service: str = "slurm",
    ssh_host: str = "",
    ssh_user: str = "",
    ssh_key_file: str = "",
) -> SlurmCommandRunner:
    cfg = SlurmExecConfig(
        mode=mode,
        compose_file=compose_file,
        service=service,
        ssh_host=ssh_host,
        ssh_user=ssh_user,
        ssh_key_file=ssh_key_file,
    )
    return SlurmCommandRunner(cfg)


# ---------------------------------------------------------------------------
# _build — docker mode
# ---------------------------------------------------------------------------


def test_build_docker_mode_produces_docker_compose_exec_command() -> None:
    runner = make_runner("docker", compose_file="infra/docker/docker-compose.yml", service="slurm")
    cmd = runner._build("sinfo -h")
    assert cmd.startswith("docker compose -f")
    assert "exec -T" in cmd
    assert shlex.quote("infra/docker/docker-compose.yml") in cmd
    assert shlex.quote("slurm") in cmd
    assert shlex.quote("sinfo -h") in cmd


def test_build_docker_mode_wraps_command_in_bash_lc() -> None:
    runner = make_runner("docker")
    cmd = runner._build("echo test")
    assert "bash -lc" in cmd
    assert shlex.quote("echo test") in cmd


def test_build_docker_mode_quotes_compose_file_with_spaces() -> None:
    runner = make_runner("docker", compose_file="path with spaces/docker-compose.yml")
    cmd = runner._build("sinfo")
    assert shlex.quote("path with spaces/docker-compose.yml") in cmd


# ---------------------------------------------------------------------------
# _build — local mode
# ---------------------------------------------------------------------------


def test_build_local_mode_produces_bash_lc_command() -> None:
    runner = make_runner("local")
    cmd = runner._build("sinfo -h")
    assert cmd == f"bash -lc {shlex.quote('sinfo -h')}"


def test_build_local_mode_no_docker_or_ssh_prefix() -> None:
    runner = make_runner("local")
    cmd = runner._build("echo hi")
    assert "docker" not in cmd
    assert "ssh" not in cmd


# ---------------------------------------------------------------------------
# _build — ssh mode
# ---------------------------------------------------------------------------


def test_build_ssh_mode_without_user_produces_bare_host() -> None:
    runner = make_runner("ssh", ssh_host="myhost.example.com")
    cmd = runner._build("sinfo")
    assert cmd.startswith("ssh ")
    assert shlex.quote("myhost.example.com") in cmd
    assert "@" not in cmd


def test_build_ssh_mode_with_user_produces_user_at_host() -> None:
    runner = make_runner("ssh", ssh_host="myhost.example.com", ssh_user="alice")
    cmd = runner._build("sinfo")
    assert shlex.quote("alice@myhost.example.com") in cmd


def test_build_ssh_mode_with_key_file_includes_identity_flag() -> None:
    runner = make_runner(
        "ssh", ssh_host="myhost.example.com", ssh_key_file="/home/user/.ssh/id_rsa"
    )
    cmd = runner._build("sinfo")
    assert "-i" in cmd
    assert shlex.quote("/home/user/.ssh/id_rsa") in cmd


def test_build_ssh_mode_without_key_file_omits_identity_flag() -> None:
    runner = make_runner("ssh", ssh_host="myhost.example.com")
    cmd = runner._build("sinfo")
    assert "-i " not in cmd


def test_build_ssh_mode_includes_strict_host_checking_options() -> None:
    runner = make_runner("ssh", ssh_host="myhost.example.com")
    cmd = runner._build("sinfo")
    assert "StrictHostKeyChecking=accept-new" in cmd
    assert "BatchMode=yes" in cmd


def test_build_ssh_mode_wraps_command_in_bash_lc() -> None:
    runner = make_runner("ssh", ssh_host="myhost.example.com")
    cmd = runner._build("squeue -h")
    assert "bash -lc" in cmd
    assert shlex.quote("squeue -h") in cmd


def test_build_ssh_mode_without_ssh_host_raises_value_error() -> None:
    runner = make_runner("ssh", ssh_host="")
    with pytest.raises(ValueError, match="SLURM_SSH_HOST"):
        runner._build("sinfo")


# ---------------------------------------------------------------------------
# _build — unsupported mode
# ---------------------------------------------------------------------------


def test_build_unsupported_mode_raises_value_error() -> None:
    runner = make_runner("kubernetes")
    with pytest.raises(ValueError, match="unsupported SLURM_EXEC_MODE"):
        runner._build("sinfo")


def test_build_mode_is_case_insensitive() -> None:
    """Mode matching strips and lowercases the mode string."""
    runner = make_runner("LOCAL")
    cmd = runner._build("echo hi")
    assert cmd == f"bash -lc {shlex.quote('echo hi')}"


def test_build_mode_strips_whitespace() -> None:
    runner = make_runner("  local  ")
    cmd = runner._build("echo hi")
    assert cmd == f"bash -lc {shlex.quote('echo hi')}"


# ---------------------------------------------------------------------------
# run() — real local subprocess (echo/false; does NOT touch Slurm)
# ---------------------------------------------------------------------------


def test_run_local_success_returns_true_and_stdout() -> None:
    runner = make_runner("local")
    ok, out = runner.run("echo hello-slurm-ai")
    assert ok is True
    assert "hello-slurm-ai" in out


def test_run_local_failure_returns_false_and_error_message() -> None:
    runner = make_runner("local")
    # 'false' exits with code 1; run() catches CalledProcessError
    ok, out = runner.run("false", check=True)
    assert ok is False


def test_run_local_stdout_is_stripped() -> None:
    runner = make_runner("local")
    ok, out = runner.run("printf '  trimmed  '")
    assert ok is True
    assert out == "trimmed"


def test_run_local_captures_multiline_output() -> None:
    runner = make_runner("local")
    ok, out = runner.run("printf 'line1\nline2\nline3'")
    assert ok is True
    assert "line1" in out
    assert "line3" in out


# ---------------------------------------------------------------------------
# Fix 4 regression: check=False must still return ok=False on non-zero exit
# ---------------------------------------------------------------------------


def test_run_check_false_nonzero_exit_returns_false() -> None:
    """run(check=False) on a command that exits non-zero must return ok=False, not True."""
    runner = make_runner("local")
    ok, _out = runner.run("false", check=False)
    assert ok is False


def test_run_check_false_zero_exit_returns_true() -> None:
    """run(check=False) on a successful command still returns ok=True."""
    runner = make_runner("local")
    ok, out = runner.run("echo check-false-ok", check=False)
    assert ok is True
    assert "check-false-ok" in out


def test_run_check_true_nonzero_exit_returns_false() -> None:
    """Existing behaviour: run(check=True) on failure returns ok=False via CalledProcessError."""
    runner = make_runner("local")
    ok, _out = runner.run("false", check=True)
    assert ok is False
