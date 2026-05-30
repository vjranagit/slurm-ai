"""Tests for simulator/replay.py — replay_csv averages and error handling."""
from __future__ import annotations

import os
import tempfile

import pytest

from controller.simulator.replay import ReplayResult, replay_csv


def write_csv(content: str) -> str:
    """Write CSV content to a temp file, return path. Caller must delete."""
    fd, path = tempfile.mkstemp(suffix=".csv")
    with os.fdopen(fd, "w") as f:
        f.write(content)
    return path


@pytest.fixture
def csv_path(tmp_path: pytest.TempPathFactory) -> str:  # type: ignore[type-arg]
    p = str(tmp_path / "trace.csv")
    return p


# ---------------------------------------------------------------------------
# Normal CSV with multiple rows
# ---------------------------------------------------------------------------


def test_replay_csv_computes_correct_avg_pending() -> None:
    content = "pending_jobs,running_jobs,saturation\n10,5,0.5\n20,10,0.8\n"
    path = write_csv(content)
    try:
        result = replay_csv(path)
        assert result.avg_pending == pytest.approx(15.0)
    finally:
        os.unlink(path)


def test_replay_csv_computes_correct_avg_running() -> None:
    content = "pending_jobs,running_jobs,saturation\n10,5,0.5\n20,10,0.8\n"
    path = write_csv(content)
    try:
        result = replay_csv(path)
        assert result.avg_running == pytest.approx(7.5)
    finally:
        os.unlink(path)


def test_replay_csv_computes_correct_avg_saturation() -> None:
    content = "pending_jobs,running_jobs,saturation\n10,5,0.5\n20,10,0.8\n"
    path = write_csv(content)
    try:
        result = replay_csv(path)
        assert result.avg_saturation == pytest.approx(0.65)
    finally:
        os.unlink(path)


def test_replay_csv_single_row_returns_exact_values() -> None:
    content = "pending_jobs,running_jobs,saturation\n42,7,0.75\n"
    path = write_csv(content)
    try:
        result = replay_csv(path)
        assert result.avg_pending == pytest.approx(42.0)
        assert result.avg_running == pytest.approx(7.0)
        assert result.avg_saturation == pytest.approx(0.75)
    finally:
        os.unlink(path)


def test_replay_csv_returns_replay_result_instance() -> None:
    content = "pending_jobs,running_jobs,saturation\n5,5,0.5\n"
    path = write_csv(content)
    try:
        result = replay_csv(path)
        assert isinstance(result, ReplayResult)
    finally:
        os.unlink(path)


def test_replay_csv_three_rows_averages_correctly() -> None:
    content = "pending_jobs,running_jobs,saturation\n0,0,0.0\n6,3,0.6\n12,6,0.9\n"
    path = write_csv(content)
    try:
        result = replay_csv(path)
        assert result.avg_pending == pytest.approx(6.0)
        assert result.avg_running == pytest.approx(3.0)
        assert result.avg_saturation == pytest.approx(0.5)
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Empty CSV raises ValueError
# ---------------------------------------------------------------------------


def test_replay_csv_empty_body_raises_value_error() -> None:
    # Only header, no data rows
    content = "pending_jobs,running_jobs,saturation\n"
    path = write_csv(content)
    try:
        with pytest.raises(ValueError, match="empty"):
            replay_csv(path)
    finally:
        os.unlink(path)


def test_replay_csv_completely_empty_file_raises_value_error() -> None:
    content = ""
    path = write_csv(content)
    try:
        with pytest.raises(ValueError, match="empty"):
            replay_csv(path)
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Missing columns default to zero
# ---------------------------------------------------------------------------


def test_replay_csv_missing_saturation_column_defaults_to_zero() -> None:
    content = "pending_jobs,running_jobs\n10,5\n"
    path = write_csv(content)
    try:
        result = replay_csv(path)
        assert result.avg_saturation == pytest.approx(0.0)
    finally:
        os.unlink(path)
