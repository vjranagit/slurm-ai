from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass


@dataclass(slots=True)
class SlurmExecConfig:
    mode: str
    compose_file: str
    service: str
    ssh_host: str = ""
    ssh_user: str = ""
    ssh_key_file: str = ""


class SlurmCommandRunner:
    def __init__(self, cfg: SlurmExecConfig) -> None:
        self.cfg = cfg

    def _build(self, command: str) -> str:
        mode = self.cfg.mode.strip().lower()
        if mode == "docker":
            return (
                f"docker compose -f {shlex.quote(self.cfg.compose_file)} exec -T "
                f"{shlex.quote(self.cfg.service)} bash -lc {shlex.quote(command)}"
            )
        if mode == "local":
            return f"bash -lc {shlex.quote(command)}"
        if mode == "ssh":
            if not self.cfg.ssh_host:
                raise ValueError("SLURM_SSH_HOST is required when SLURM_EXEC_MODE=ssh")
            identity = (
                f"-i {shlex.quote(self.cfg.ssh_key_file)} " if self.cfg.ssh_key_file else ""
            )
            strict = "-o StrictHostKeyChecking=accept-new -o BatchMode=yes"
            user_host = (
                f"{self.cfg.ssh_user}@{self.cfg.ssh_host}"
                if self.cfg.ssh_user
                else self.cfg.ssh_host
            )
            return (
                f"ssh {identity}{strict} {shlex.quote(user_host)} "
                f"bash -lc {shlex.quote(command)}"
            )
        raise ValueError(f"unsupported SLURM_EXEC_MODE: {self.cfg.mode}")

    def run(self, command: str, check: bool = True) -> tuple[bool, str]:
        wrapped = self._build(command)
        try:
            proc = subprocess.run(wrapped, shell=True, check=check, capture_output=True, text=True)
            return True, proc.stdout.strip()
        except subprocess.CalledProcessError as exc:
            msg = (exc.stderr or exc.stdout or str(exc)).strip()
            return False, msg

