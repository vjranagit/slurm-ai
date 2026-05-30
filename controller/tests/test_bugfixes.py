"""Regression tests for bugs found during end-to-end testing.

1. Actuator must not run `scontrol reconfigure` — it reverted the non-slurmdbd
   partition fallback (MaxCPUsPerNode), which slurm.conf does not persist.
2. `/metrics` must expose the `adaptive_*` gauges (apps/api/main.py must import
   controller.metrics so they register in the default Prometheus registry).
"""

import pytest

from controller.actuator.slurm import SlurmActuator
from controller.config import ControllerConfig


def _live_cfg():
    cfg = ControllerConfig()
    cfg.dry_run = False
    cfg.cooldown_sec = 0
    cfg.max_jobs_floor = 2
    cfg.max_jobs_ceil = 16
    return cfg


def test_actuator_does_not_run_reconfigure(monkeypatch):
    calls = []

    def fake_run(self, command, check=True):
        calls.append(command)
        return True, "ok"

    monkeypatch.setattr("controller.slurm_exec.SlurmCommandRunner.run", fake_run)
    SlurmActuator(_live_cfg()).apply(4, 2500)
    joined = " ".join(calls)
    assert "reconfigure" not in joined
    assert ("MaxJobsPU=4" in joined) or ("MaxCPUsPerNode=4" in joined)


def test_metrics_endpoint_exposes_adaptive_gauges():
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    import apps.api.main as apimod

    body = TestClient(apimod.app).get("/metrics").text
    assert any(line.startswith("adaptive_") for line in body.splitlines())
