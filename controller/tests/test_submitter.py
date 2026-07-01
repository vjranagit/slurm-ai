"""Tests for controller/workload/submitter.py — non-predictable job script paths."""
from __future__ import annotations

import glob
import os

import pytest

from controller.workload.submitter import submit_jobs

_TMP_GLOB = "/tmp/adaptive-job-*.sh"


@pytest.fixture(autouse=True)
def _cleanup_leftover_tmp_files():
    """Guard the shared /tmp namespace: sweep before and after every test in this
    module so a failing assertion (or a bug under test) never leaks files across
    test runs or leaves the sandbox dirty."""

    def _sweep() -> None:
        for path in glob.glob(_TMP_GLOB):
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass

    _sweep()
    yield
    _sweep()


def test_submit_jobs_uses_non_predictable_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Generated script path must not be the old fixed '/tmp/adaptive-job-{i}.sh' string —
    it should come from tempfile.mkstemp (unique, unpredictable suffix)."""
    seen_commands: list[str] = []

    class FakeRunner:
        def run(self, command: str, check: bool = True) -> tuple[bool, str]:
            seen_commands.append(command)
            return True, "Submitted batch job 1"

    monkeypatch.setattr(
        "controller.workload.submitter.SlurmCommandRunner", lambda cfg: FakeRunner()
    )

    before = set(glob.glob(_TMP_GLOB))

    submit_jobs(
        compose_file="infra/docker/docker-compose.yml",
        service="slurm",
        count=1,
        seconds=1,
        exec_mode="local",
    )

    after = set(glob.glob(_TMP_GLOB))

    assert len(seen_commands) == 2  # create_script + sbatch
    create_cmd, sbatch_cmd = seen_commands
    assert "/tmp/adaptive-job-0.sh" not in create_cmd
    assert "/tmp/adaptive-job-0.sh" not in sbatch_cmd
    assert "adaptive-job-" in create_cmd
    assert create_cmd.count("adaptive-job-") >= 1

    # The local placeholder file (created only to get an unpredictable name) must
    # be unlinked after the job is submitted — no leftover /tmp/adaptive-job-*.sh.
    assert after == before, f"leftover temp files after submit_jobs: {after - before}"


def test_submit_jobs_paths_are_unique_across_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two jobs in the same run must not collide on the same predictable path."""
    seen_commands: list[str] = []

    class FakeRunner:
        def run(self, command: str, check: bool = True) -> tuple[bool, str]:
            seen_commands.append(command)
            return True, "Submitted batch job 1"

    monkeypatch.setattr(
        "controller.workload.submitter.SlurmCommandRunner", lambda cfg: FakeRunner()
    )

    before = set(glob.glob(_TMP_GLOB))

    submit_jobs(
        compose_file="infra/docker/docker-compose.yml",
        service="slurm",
        count=2,
        seconds=1,
        exec_mode="local",
    )

    after = set(glob.glob(_TMP_GLOB))

    create_cmd_0 = seen_commands[0]
    create_cmd_1 = seen_commands[2]
    assert create_cmd_0 != create_cmd_1

    # No leftover placeholder files for either of the two submitted jobs.
    assert after == before, f"leftover temp files after submit_jobs: {after - before}"


def test_submit_jobs_leaves_no_leftover_tmp_files_across_multiple_jobs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit leak check: snapshot /tmp/adaptive-job-*.sh count before/after a
    5-job submission run — the count must be identical (no net leak), and no
    single leftover path should remain from this run."""
    seen_commands: list[str] = []
    created_paths: list[str] = []

    class FakeRunner:
        def run(self, command: str, check: bool = True) -> tuple[bool, str]:
            seen_commands.append(command)
            if command.startswith("cat >"):
                # extract the path token right after "cat > "
                path = command.split("cat > ", 1)[1].split(" <<", 1)[0]
                created_paths.append(path)
            return True, "Submitted batch job 1"

    monkeypatch.setattr(
        "controller.workload.submitter.SlurmCommandRunner", lambda cfg: FakeRunner()
    )

    before_count = len(glob.glob(_TMP_GLOB))

    submit_jobs(
        compose_file="infra/docker/docker-compose.yml",
        service="slurm",
        count=5,
        seconds=1,
        exec_mode="local",
    )

    after_count = len(glob.glob(_TMP_GLOB))

    assert len(created_paths) == 5
    assert after_count == before_count, (
        f"temp file count changed: before={before_count} after={after_count}"
    )
    for path in created_paths:
        assert not os.path.exists(path), f"leftover placeholder file: {path}"
