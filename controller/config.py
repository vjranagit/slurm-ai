import os
from dataclasses import dataclass


@dataclass(slots=True)
class ControllerConfig:
    interval_sec: int = int(os.getenv("CONTROLLER_INTERVAL_SEC", "15"))
    cooldown_sec: int = int(os.getenv("CONTROLLER_COOLDOWN_SEC", "60"))
    dry_run: bool = os.getenv("CONTROLLER_DRY_RUN", "true").lower() == "true"
    max_jobs_floor: int = int(os.getenv("MAX_JOBS_FLOOR", "2"))
    max_jobs_ceil: int = int(os.getenv("MAX_JOBS_CEIL", "128"))
    pressure_high: float = float(os.getenv("PRESSURE_HIGH", "0.85"))
    pressure_low: float = float(os.getenv("PRESSURE_LOW", "0.45"))
    compose_file: str = os.getenv("SLURM_COMPOSE_FILE", "infra/docker/docker-compose.yml")
    slurm_service: str = os.getenv("SLURM_SERVICE", "slurm")
    slurm_exec_mode: str = os.getenv("SLURM_EXEC_MODE", "docker")
    slurm_ssh_host: str = os.getenv("SLURM_SSH_HOST", "")
    slurm_ssh_user: str = os.getenv("SLURM_SSH_USER", "")
    slurm_ssh_key_file: str = os.getenv("SLURM_SSH_KEY_FILE", "")
