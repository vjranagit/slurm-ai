"""Deterministic queue simulator and Q-learning trainer for RLTuner.

The simulator models a fixed-capacity cluster with Poisson-ish job arrivals
controlled by a seeded RNG (pure stdlib — no numpy).  It produces reward
signals that make an optimal bounded policy learnable:

  reward = utilisation_score - saturation_penalty - thrash_penalty

where
  utilisation_score = running / max_jobs         (reward high utilisation)
  saturation_penalty = sat * 0.5                 (penalise high pressure)
  thrash_penalty = |max_jobs_t - max_jobs_{t-1}| * 0.01  (penalise churn)

This combination means the optimal policy holds or gently increases when
utilisation is good, and backs off sharply only when genuinely saturated.
"""
from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

from controller.tuner.rl import (
    QTable,
    _N_ACTIONS,
    _apply_action,
    _maxjobs_bucket,
    _pressure_bucket,
    load_qtable,
    save_qtable,
)

if TYPE_CHECKING:
    from controller.config import ControllerConfig

# ---------------------------------------------------------------------------
# Queue simulator
# ---------------------------------------------------------------------------


class QueueSimulator:
    """Minimal deterministic queue model.

    Args:
        capacity: Maximum concurrent running jobs (node slots).
        arrival_rate: Mean jobs arriving per step (Poisson parameter).
        seed: RNG seed for full reproducibility.
    """

    def __init__(self, capacity: int = 32, arrival_rate: float = 3.0, seed: int = 42) -> None:
        self._capacity = capacity
        self._arrival_rate = arrival_rate
        self._rng = random.Random(seed)
        self._queue: int = 0  # pending jobs
        self._running: int = 0

    def reset(self) -> None:
        """Reset simulator state (RNG is NOT reset — call with a fresh instance for that)."""
        self._queue = 0
        self._running = 0

    def step(self, max_jobs: int) -> tuple[float, float]:
        """Advance the simulation one tick.

        Args:
            max_jobs: Current max concurrency limit.

        Returns:
            (saturation, utilisation) both ∈ [0, 1].
        """
        # Arrivals: Poisson approximated via sum of Bernoulli trials (stdlib)
        arrivals = _poisson(self._rng, self._arrival_rate)
        self._queue += arrivals

        # Completions: each running job finishes with p=0.4
        completions = sum(1 for _ in range(self._running) if self._rng.random() < 0.4)
        self._running = max(0, self._running - completions)

        # Admit up to max_jobs (also capped by physical capacity)
        effective_cap = min(max_jobs, self._capacity)
        admissible = max(0, effective_cap - self._running)
        admitted = min(self._queue, admissible)
        self._queue -= admitted
        self._running += admitted

        total = self._queue + self._running
        saturation = total / (self._capacity + 1e-9)
        saturation = min(1.0, saturation)
        utilisation = self._running / (effective_cap + 1e-9)
        utilisation = min(1.0, utilisation)
        return saturation, utilisation


def _poisson(rng: random.Random, lam: float) -> int:
    """Draw a Poisson sample via Knuth's algorithm (pure stdlib)."""
    if lam <= 0:
        return 0
    l_ = math.exp(-lam)
    k = 0
    p = 1.0
    while p > l_:
        p *= rng.random()
        k += 1
    return k - 1


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------


def _reward(saturation: float, utilisation: float, max_jobs: int, prev_max_jobs: int) -> float:
    """Compute step reward.

    Rewards high utilisation, penalises saturation and thrashing.
    """
    thrash_penalty = abs(max_jobs - prev_max_jobs) * 0.01
    return utilisation - saturation * 0.5 - thrash_penalty


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_rl(
    cfg: "ControllerConfig",
    episodes: int | None = None,
    seed: int = 42,
) -> tuple[QTable, list[float]]:
    """Train a Q-table using tabular Q-learning on the queue simulator.

    Args:
        cfg: Controller config (bounds + RL hyperparameters).
        episodes: Number of training episodes; defaults to cfg.rl_train_episodes.
        seed: Master seed for reproducibility.

    Returns:
        (qtable, reward_history) where reward_history[i] is the total
        undiscounted reward for episode i.
    """
    from controller.tuner.rl import RLTuner

    if episodes is None:
        episodes = cfg.rl_train_episodes

    floor = cfg.max_jobs_floor
    ceil = cfg.max_jobs_ceil
    epsilon = cfg.rl_epsilon
    steps_per_episode = 200

    master_rng = random.Random(seed)
    tuner = RLTuner(cfg, qtable={})
    reward_history: list[float] = []

    for ep in range(episodes):
        ep_seed = master_rng.randint(0, 2**31)
        sim = QueueSimulator(capacity=ceil, arrival_rate=3.0, seed=ep_seed)
        max_jobs = (floor + ceil) // 2
        prev_max_jobs = max_jobs
        total_reward = 0.0

        sat, util = sim.step(max_jobs)

        for _ in range(steps_per_episode):
            pb = _pressure_bucket(sat)
            mb = _maxjobs_bucket(max_jobs, floor, ceil)
            state = (pb, mb)

            # Epsilon-greedy action selection
            if master_rng.random() < epsilon or state not in tuner.qtable:
                action = master_rng.randint(0, _N_ACTIONS - 1)
            else:
                action = int(max(range(_N_ACTIONS), key=lambda a: tuner.qtable[state][a]))

            new_max_jobs = _apply_action(action, max_jobs, floor, ceil)
            sat, util = sim.step(new_max_jobs)

            r = _reward(sat, util, new_max_jobs, prev_max_jobs)
            total_reward += r

            next_pb = _pressure_bucket(sat)
            next_mb = _maxjobs_bucket(new_max_jobs, floor, ceil)
            next_state = (next_pb, next_mb)

            tuner.update(state, action, r, next_state)

            prev_max_jobs = max_jobs
            max_jobs = new_max_jobs

        reward_history.append(total_reward)

    return tuner.qtable, reward_history


# ---------------------------------------------------------------------------
# Persistence helpers (thin wrappers; canonical implementations live in rl.py)
# ---------------------------------------------------------------------------


def save_qtable_json(qtable: QTable, path: str) -> None:
    """Save Q-table to *path* (creates parent dirs)."""
    save_qtable(qtable, path)


def load_qtable_json(path: str) -> QTable:
    """Load Q-table from *path*; empty dict if missing."""
    return load_qtable(path)
