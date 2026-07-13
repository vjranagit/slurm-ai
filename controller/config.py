import os
from dataclasses import dataclass

_TRUE_VALUES = {"true", "1", "yes", "on"}
_FALSE_VALUES = {"false", "0", "no", "off"}


def _parse_bool(value: str, default: bool) -> bool:
    """Parse a boolean-ish string fail-safe: unrecognized input returns *default*.

    Never raises. Recognized true/false spellings are case- and whitespace-
    insensitive; anything else (garbage, empty string, typos) falls back to
    *default* rather than being silently misinterpreted.
    """
    v = value.strip().lower()
    if v in _TRUE_VALUES:
        return True
    if v in _FALSE_VALUES:
        return False
    return default


@dataclass(slots=True)
class ControllerConfig:
    interval_sec: int = int(os.getenv("CONTROLLER_INTERVAL_SEC", "15"))
    cooldown_sec: int = int(os.getenv("CONTROLLER_COOLDOWN_SEC", "60"))
    dry_run: bool = _parse_bool(os.getenv("CONTROLLER_DRY_RUN", "true"), True)
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
    exec_timeout_sec: int = int(os.getenv("SLURM_EXEC_TIMEOUT_SEC", "30"))
    ssh_strict_host_key: bool = _parse_bool(os.getenv("SLURM_SSH_STRICT_HOST_KEY", "true"), True)
    # RL tuner settings
    tuner_kind: str = os.getenv("TUNER_KIND", "aimd")
    rl_qtable_path: str = os.getenv("RL_QTABLE_PATH", "models/qtable.json")
    rl_alpha: float = float(os.getenv("RL_ALPHA", "0.1"))
    rl_gamma: float = float(os.getenv("RL_GAMMA", "0.9"))
    rl_epsilon: float = float(os.getenv("RL_EPSILON", "0.1"))
    rl_train_episodes: int = int(os.getenv("RL_TRAIN_EPISODES", "300"))

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Re-check invariants — call again after mutating fields post-construction.

        `__post_init__` only runs once, at construction time. `controller/main.py`'s
        CLI overrides (`--interval`, `--dry-run`) mutate the dataclass *after*
        construction, which would silently bypass `__post_init__`-only validation;
        callers that mutate fields after construction must call `validate()` again.

        Every check here guards a documented safety invariant (see CLAUDE.md
        "Safety invariants") against a garbage/hostile/typo'd env var or CLI flag
        producing a value that is accepted but behaves unsafely at runtime instead
        of failing fast with a clear error.
        """
        if self.max_jobs_floor > self.max_jobs_ceil:
            raise ValueError(
                f"max_jobs_floor ({self.max_jobs_floor}) must be <= "
                f"max_jobs_ceil ({self.max_jobs_ceil})"
            )
        if self.max_jobs_floor < 0:
            raise ValueError(f"max_jobs_floor ({self.max_jobs_floor}) must be >= 0")
        if self.interval_sec < 0:
            raise ValueError(
                f"interval_sec ({self.interval_sec}) must be >= 0 "
                "(negative would raise ValueError out of time.sleep at runtime)"
            )
        if self.cooldown_sec < 0:
            raise ValueError(
                f"cooldown_sec ({self.cooldown_sec}) must be >= 0 "
                "(negative silently disables the actuator cooldown safety invariant)"
            )
        if self.exec_timeout_sec <= 0:
            raise ValueError(
                f"exec_timeout_sec ({self.exec_timeout_sec}) must be > 0 "
                "(non-positive breaks/disables the subprocess timeout guard)"
            )
        if not (0.0 <= self.pressure_low <= self.pressure_high <= 1.0):
            raise ValueError(
                f"pressure thresholds must satisfy 0 <= pressure_low ({self.pressure_low}) "
                f"<= pressure_high ({self.pressure_high}) <= 1"
            )
        if not (0.0 < self.rl_alpha <= 1.0):
            raise ValueError(f"rl_alpha ({self.rl_alpha}) must be in (0, 1]")
        if not (0.0 <= self.rl_gamma <= 1.0):
            raise ValueError(f"rl_gamma ({self.rl_gamma}) must be in [0, 1]")
        if not (0.0 <= self.rl_epsilon <= 1.0):
            raise ValueError(f"rl_epsilon ({self.rl_epsilon}) must be in [0, 1]")
        if self.rl_train_episodes <= 0:
            raise ValueError(f"rl_train_episodes ({self.rl_train_episodes}) must be > 0")
