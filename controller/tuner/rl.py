"""Tabular Q-learning tuner — drop-in replacement for AimdTuner.

State space  : (pressure_bucket, maxjobs_bucket)
Action space : decrease (halve, clamp floor) | hold | increase (+1, clamp ceil)
Fallback     : delegates to AimdTuner when Q-table has no entry for a state,
               guaranteeing safe bounded behaviour even with an empty table.
Persistence  : Q-table stored as JSON at cfg.rl_qtable_path; missing file → empty table.
"""
from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

from controller.tuner.aimd import AimdTuner

if TYPE_CHECKING:
    from controller.config import ControllerConfig

# ---------------------------------------------------------------------------
# Discretisation helpers
# ---------------------------------------------------------------------------

# Pressure buckets: 5 equal-width bins over [0, 1]
_PRESSURE_BINS = 5

# Max-jobs buckets: 6 logarithmically-spaced buckets across [floor, ceil]
_MAXJOBS_BINS = 6

# Actions
_ACTION_DECREASE = 0
_ACTION_HOLD = 1
_ACTION_INCREASE = 2
_N_ACTIONS = 3


def _pressure_bucket(saturation: float) -> int:
    """Map saturation ∈ [0,1] → bucket ∈ [0, _PRESSURE_BINS-1]."""
    bucket = int(saturation * _PRESSURE_BINS)
    return min(bucket, _PRESSURE_BINS - 1)


def _maxjobs_bucket(current: int, floor: int, ceil: int) -> int:
    """Map current ∈ [floor, ceil] → bucket ∈ [0, _MAXJOBS_BINS-1]."""
    if ceil <= floor:
        return 0
    span = ceil - floor
    bucket = int((current - floor) / span * _MAXJOBS_BINS)
    return min(bucket, _MAXJOBS_BINS - 1)


def _apply_action(action: int, current: int, floor: int, ceil: int) -> int:
    """Apply action and clamp to [floor, ceil]."""
    if action == _ACTION_DECREASE:
        return max(floor, current // 2)
    if action == _ACTION_INCREASE:
        return min(ceil, current + 1)
    # hold
    return current


# ---------------------------------------------------------------------------
# Q-table serialisation
# ---------------------------------------------------------------------------

# The table is stored as  {pressure_bucket: {maxjobs_bucket: [q0,q1,q2]}}
# JSON keys are always strings; we convert on load.

QTable = dict[tuple[int, int], list[float]]


def _qtable_to_json(qtable: QTable) -> dict[str, dict[str, list[float]]]:
    out: dict[str, dict[str, list[float]]] = {}
    for (pb, mb), vals in qtable.items():
        out.setdefault(str(pb), {})[str(mb)] = vals
    return out


def _json_to_qtable(data: dict[str, dict[str, list[float]]]) -> QTable:
    qtable: QTable = {}
    for pb_str, inner in data.items():
        for mb_str, vals in inner.items():
            qtable[(int(pb_str), int(mb_str))] = list(vals)
    return qtable


def save_qtable(qtable: QTable, path: str) -> None:
    """Persist Q-table to *path* as JSON (creates parent dirs)."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(_qtable_to_json(qtable), fh, indent=2)


def load_qtable(path: str) -> QTable:
    """Load Q-table from *path*; return empty dict if file is missing."""
    try:
        with open(path) as fh:
            return _json_to_qtable(json.load(fh))
    except FileNotFoundError:
        return {}


# ---------------------------------------------------------------------------
# RLTuner
# ---------------------------------------------------------------------------


class RLTuner:
    """Tabular Q-learning tuner with AIMD fallback for unseen states.

    Args:
        cfg: Controller configuration. Used for bounds and RL hyperparameters.
        qtable: Pre-loaded Q-table; loaded from cfg.rl_qtable_path if None.
    """

    def __init__(self, cfg: "ControllerConfig", qtable: QTable | None = None) -> None:
        self.cfg = cfg
        self._fallback = AimdTuner(cfg)
        if qtable is not None:
            self._qtable: QTable = qtable
        else:
            self._qtable = load_qtable(cfg.rl_qtable_path)

    # ------------------------------------------------------------------
    # Public interface — identical to AimdTuner
    # ------------------------------------------------------------------

    def next_max_jobs(self, current: int, saturation: float) -> int:
        """Return next max_jobs value within [cfg.max_jobs_floor, cfg.max_jobs_ceil].

        Uses greedy Q-policy for known states; falls back to AIMD for unseen states.

        Args:
            current: Current max_jobs setting.
            saturation: Saturation/pressure score in [0, 1].

        Returns:
            New max_jobs as int, always within [floor, ceil].
        """
        floor = self.cfg.max_jobs_floor
        ceil = self.cfg.max_jobs_ceil

        # Clamp current to bounds (defensive)
        current = max(floor, min(ceil, current))

        pb = _pressure_bucket(saturation)
        mb = _maxjobs_bucket(current, floor, ceil)
        state = (pb, mb)

        if state in self._qtable:
            action = int(max(range(_N_ACTIONS), key=lambda a: self._qtable[state][a]))
            result = _apply_action(action, current, floor, ceil)
        else:
            # Fallback: delegate to AIMD so an untrained table is safe
            result = self._fallback.next_max_jobs(current, saturation)

        # Final clamp — must always be within bounds
        return max(floor, min(ceil, int(result)))

    # ------------------------------------------------------------------
    # Q-learning update (used during training; not called at inference)
    # ------------------------------------------------------------------

    def update(
        self,
        state: tuple[int, int],
        action: int,
        reward: float,
        next_state: tuple[int, int],
    ) -> None:
        """Single Q-learning update step (Bellman equation).

        Args:
            state: Current (pressure_bucket, maxjobs_bucket).
            action: Action taken (0=decrease, 1=hold, 2=increase).
            reward: Observed scalar reward.
            next_state: Resulting (pressure_bucket, maxjobs_bucket).
        """
        alpha = self.cfg.rl_alpha
        gamma = self.cfg.rl_gamma

        if state not in self._qtable:
            self._qtable[state] = [0.0] * _N_ACTIONS
        if next_state not in self._qtable:
            self._qtable[next_state] = [0.0] * _N_ACTIONS

        q_current = self._qtable[state][action]
        q_next_max = max(self._qtable[next_state])
        self._qtable[state][action] = q_current + alpha * (reward + gamma * q_next_max - q_current)

    @property
    def qtable(self) -> QTable:
        """Read-only view of the current Q-table."""
        return self._qtable
