"""Tuner package — exports AimdTuner, RLTuner, and the build_tuner factory."""
from __future__ import annotations

from typing import TYPE_CHECKING, Union

from controller.tuner.aimd import AimdTuner
from controller.tuner.rl import RLTuner

if TYPE_CHECKING:
    from controller.config import ControllerConfig

Tuner = Union[AimdTuner, RLTuner]

__all__ = ["AimdTuner", "RLTuner", "Tuner", "build_tuner"]


def build_tuner(cfg: "ControllerConfig") -> Tuner:
    """Factory: return RLTuner if cfg.tuner_kind is 'rl', else AimdTuner.

    Args:
        cfg: Controller configuration.

    Returns:
        A tuner instance exposing ``next_max_jobs(current, saturation) -> int``.
    """
    if cfg.tuner_kind.lower() == "rl":
        return RLTuner(cfg)
    return AimdTuner(cfg)
