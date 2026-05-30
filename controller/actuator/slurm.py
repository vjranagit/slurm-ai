from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from controller.config import ControllerConfig
from controller.slurm_exec import SlurmCommandRunner, SlurmExecConfig
from controller.types import AppliedAction

LOG = logging.getLogger("adaptive-controller.actuator")


@dataclass(slots=True)
class ControllerState:
    max_jobs: int
    priority_weight_fs: int


class SlurmActuator:
    def __init__(self, cfg: ControllerConfig) -> None:
        self.cfg = cfg
        init_max = max(cfg.max_jobs_floor, min(cfg.max_jobs_ceil, 8))
        self.state = ControllerState(max_jobs=init_max, priority_weight_fs=1000)
        self.last_apply = datetime.min.replace(tzinfo=timezone.utc)
        self.runner = SlurmCommandRunner(
            SlurmExecConfig(
                mode=cfg.slurm_exec_mode,
                compose_file=cfg.compose_file,
                service=cfg.slurm_service,
                ssh_host=cfg.slurm_ssh_host,
                ssh_user=cfg.slurm_ssh_user,
                ssh_key_file=cfg.slurm_ssh_key_file,
            )
        )

    def apply(self, target_max_jobs: int, target_priority_weight_fs: int) -> AppliedAction:
        now = datetime.now(timezone.utc)
        if now - self.last_apply < timedelta(seconds=self.cfg.cooldown_sec):
            return AppliedAction(
                changed=False,
                old_max_jobs=self.state.max_jobs,
                new_max_jobs=self.state.max_jobs,
                old_priority_weight_fs=self.state.priority_weight_fs,
                new_priority_weight_fs=self.state.priority_weight_fs,
                command_log=["cooldown: skip"],
            )

        old_max = self.state.max_jobs
        old_weight = self.state.priority_weight_fs
        new_max = max(self.cfg.max_jobs_floor, min(self.cfg.max_jobs_ceil, target_max_jobs))
        new_weight = max(1, min(10000, target_priority_weight_fs))

        commands = [
            (
                "if grep -q 'AccountingStorageType=accounting_storage/slurmdbd' /etc/slurm/slurm.conf; "
                f"then sacctmgr -i modify qos normal set MaxJobsPU={new_max}; "
                "else scontrol update PartitionName=debug MaxCPUsPerNode="
                f"{new_max}; fi"
            ),
        ]

        command_log: list[str] = []
        all_ok = True
        if not self.cfg.dry_run:
            for cmd in commands:
                ok, out = self.runner.run(cmd, check=True)
                command_log.append(f"{'OK' if ok else 'ERR'} {cmd} :: {out}")
                if not ok:
                    LOG.warning("actuator command failed cmd=%s err=%s", cmd, out)
                    all_ok = False
        else:
            command_log = [f"DRY_RUN {c}" for c in commands]

        if all_ok:
            self.state.max_jobs = new_max
            self.state.priority_weight_fs = new_weight
            self.last_apply = now
            return AppliedAction(
                changed=(old_max != new_max) or (old_weight != new_weight),
                old_max_jobs=old_max,
                new_max_jobs=new_max,
                old_priority_weight_fs=old_weight,
                new_priority_weight_fs=new_weight,
                command_log=command_log,
            )

        # Live command failed — do not advance state, do not update last_apply
        return AppliedAction(
            changed=False,
            old_max_jobs=old_max,
            new_max_jobs=old_max,
            old_priority_weight_fs=old_weight,
            new_priority_weight_fs=old_weight,
            command_log=command_log,
        )
