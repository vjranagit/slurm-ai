from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone

from controller.slurm_exec import SlurmCommandRunner, SlurmExecConfig
from controller.types import ClusterSnapshot


class SlurmCollector:
    def __init__(
        self,
        compose_file: str,
        slurm_service: str,
        exec_mode: str = "docker",
        ssh_host: str = "",
        ssh_user: str = "",
        ssh_key_file: str = "",
    ) -> None:
        self.runner = SlurmCommandRunner(
            SlurmExecConfig(
                mode=exec_mode,
                compose_file=compose_file,
                service=slurm_service,
                ssh_host=ssh_host,
                ssh_user=ssh_user,
                ssh_key_file=ssh_key_file,
            )
        )

    def _exec(self, command: str) -> str:
        ok, out = self.runner.run(command, check=True)
        if not ok:
            raise RuntimeError(out)
        return out

    @staticmethod
    def _pressure_ratio(pending: int, running: int) -> float:
        total = max(1, pending + running)
        return pending / total

    def snapshot(self) -> ClusterSnapshot:
        sinfo = self._exec("sinfo -h -o '%t' || true").splitlines()
        squeue_rows = self._exec("squeue -h -o '%T|%r' || true").splitlines()
        sacct_rows = self._exec(
            "sacct -S now-5minutes -X -n -o State --parsable2 2>/dev/null || true"
        ).splitlines()

        idle = sum(1 for x in sinfo if "idle" in x.lower())
        alloc = sum(1 for x in sinfo if "alloc" in x.lower())
        down = sum(1 for x in sinfo if "down" in x.lower())

        pending_jobs = 0
        running_jobs = 0
        reasons: dict[str, int] = {}
        for row in squeue_rows:
            if not row:
                continue
            state, reason = (row.split("|", 1) + [""])[:2]
            s = state.lower()
            if "pend" in s:
                pending_jobs += 1
                reasons[reason] = reasons.get(reason, 0) + 1
            elif "run" in s:
                running_jobs += 1

        completed_5m = sum(1 for r in sacct_rows if r.lower().startswith("completed"))

        cpu_pressure = self._pressure_ratio(pending_jobs, running_jobs)
        mem_pressure = min(1.0, (pending_jobs * 0.2) / max(1, running_jobs + 1))
        io_pressure = min(1.0, (pending_jobs * 0.15) / max(1, running_jobs + 1))

        return ClusterSnapshot(
            timestamp=datetime.now(timezone.utc),
            total_nodes=len(sinfo),
            idle_nodes=idle,
            alloc_nodes=alloc,
            down_nodes=down,
            pending_jobs=pending_jobs,
            running_jobs=running_jobs,
            completed_jobs_5m=completed_5m,
            cpu_pressure=cpu_pressure,
            mem_pressure=mem_pressure,
            io_pressure=io_pressure,
            pending_reasons=reasons,
        )

    def info_json(self) -> str:
        return json.dumps(asdict(self.snapshot()), default=str)
