"""Training entrypoint for the RL tuner.

Usage::

    python -m controller.tuner.train

Trains a Q-table using the built-in queue simulator and saves it to the path
configured in RL_QTABLE_PATH (default: models/qtable.json).

Prints reward-improvement metrics so progress can be monitored.
"""
from __future__ import annotations

import os
import sys

from controller.config import ControllerConfig
from controller.tuner.rl_env import save_qtable_json, train_rl


def main() -> None:
    cfg = ControllerConfig()
    episodes = cfg.rl_train_episodes
    qtable_path = cfg.rl_qtable_path

    print(f"Training RL tuner: episodes={episodes} alpha={cfg.rl_alpha} "
          f"gamma={cfg.rl_gamma} epsilon={cfg.rl_epsilon}")
    print(f"Bounds: floor={cfg.max_jobs_floor} ceil={cfg.max_jobs_ceil}")
    print(f"Output: {qtable_path}")
    print()

    qtable, reward_history = train_rl(cfg, episodes=episodes, seed=42)

    n = len(reward_history)
    quarter = max(1, n // 4)
    first_q_mean = sum(reward_history[:quarter]) / quarter
    last_q_mean = sum(reward_history[n - quarter:]) / quarter
    improvement = last_q_mean - first_q_mean
    improvement_pct = (improvement / (abs(first_q_mean) + 1e-9)) * 100

    print(f"Episodes completed : {n}")
    print(f"First-quarter mean reward : {first_q_mean:.4f}")
    print(f"Last-quarter mean reward  : {last_q_mean:.4f}")
    print(f"Absolute improvement      : {improvement:.4f}")
    print(f"Relative improvement      : {improvement_pct:.1f}%")
    print(f"Q-table states populated  : {len(qtable)}")
    print()

    # Ensure output dir exists
    out_dir = os.path.dirname(qtable_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    save_qtable_json(qtable, qtable_path)
    print(f"Q-table saved to {qtable_path}")

    if improvement <= 0:
        print("WARNING: no reward improvement detected — check simulator / hyperparameters.",
              file=sys.stderr)


if __name__ == "__main__":
    main()
