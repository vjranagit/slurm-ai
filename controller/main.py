from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import logging
import time

from prometheus_client import start_http_server

from controller.actuator.slurm import SlurmActuator
from controller.collectors.slurm import SlurmCollector
from controller.config import ControllerConfig, _parse_bool
from controller.metrics import (
    ACTIONS,
    LOOP_ERRORS,
    MAX_JOBS,
    PENDING,
    PRIORITY,
    RUNNING,
    SATURATION,
)
from controller.policy.engine import PolicyEngine
from controller.tuner import build_tuner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
LOG = logging.getLogger("adaptive-controller")


def run_once(cfg: ControllerConfig, collector, policy, tuner, actuator) -> None:
    """Execute a single control-loop iteration: collect -> decide -> apply -> report."""
    snap = collector.snapshot()
    sat = policy.saturation_score(snap)
    target_max = tuner.next_max_jobs(actuator.state.max_jobs, sat)
    target_weight = policy.fairshare_weight(sat)
    applied = actuator.apply(target_max, target_weight)

    SATURATION.set(sat)
    PENDING.set(snap.pending_jobs)
    RUNNING.set(snap.running_jobs)
    MAX_JOBS.set(applied.new_max_jobs)
    PRIORITY.set(applied.new_priority_weight_fs)
    ACTIONS.set(1 if applied.changed else 0)

    LOG.info(
        "loop=%s",
        json.dumps(
            {
                "snapshot": asdict(snap),
                "decision": {
                    "target_max_jobs": target_max,
                    "target_priority_weight_fs": target_weight,
                },
                "applied": asdict(applied),
            },
            default=str,
        ),
    )


def run(cfg: ControllerConfig) -> None:
    start_http_server(9108)
    collector = SlurmCollector(
        cfg.compose_file,
        cfg.slurm_service,
        exec_mode=cfg.slurm_exec_mode,
        ssh_host=cfg.slurm_ssh_host,
        ssh_user=cfg.slurm_ssh_user,
        ssh_key_file=cfg.slurm_ssh_key_file,
        exec_timeout_sec=cfg.exec_timeout_sec,
        ssh_strict_host_key=cfg.ssh_strict_host_key,
    )
    policy = PolicyEngine(cfg)
    tuner = build_tuner(cfg)
    actuator = SlurmActuator(cfg)

    LOG.info("controller started dry_run=%s interval=%s", cfg.dry_run, cfg.interval_sec)

    while True:
        try:
            run_once(cfg, collector, policy, tuner, actuator)
        except Exception:
            LOG.exception("control loop iteration failed; will retry after interval")
            LOOP_ERRORS.inc()
            time.sleep(cfg.interval_sec)
            continue
        time.sleep(cfg.interval_sec)


def build_config_from_args(args: argparse.Namespace) -> ControllerConfig:
    """Construct config from env, apply CLI overrides, then re-validate.

    `ControllerConfig.__post_init__` only runs at construction time, so mutating
    `interval_sec`/`dry_run` afterwards (as the CLI overrides below do) would
    silently bypass validation — e.g. `--interval -5` would reach the while-loop
    and crash `time.sleep(-5)` deep in the loop instead of failing fast here with
    a clear error. Call `cfg.validate()` again after every post-construction
    mutation.
    """
    cfg = ControllerConfig()
    if args.interval is not None:
        cfg.interval_sec = args.interval
    if args.dry_run is not None:
        cfg.dry_run = _parse_bool(args.dry_run, True)
    cfg.validate()
    return cfg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=None)
    parser.add_argument("--dry-run", type=str, default=None)
    args = parser.parse_args()

    run(build_config_from_args(args))
