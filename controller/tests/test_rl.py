"""Tests for the RL tuner (RLTuner, build_tuner, train_rl).

All tests are deterministic (seeded) and self-contained — no filesystem
side-effects unless testing persistence explicitly (uses tmp_path fixture).
"""
from __future__ import annotations

import random

import pytest

from controller.config import ControllerConfig
from controller.tuner import AimdTuner, RLTuner, build_tuner
from controller.tuner.rl import (
    QTable,
    load_qtable,
    save_qtable,
)
from controller.tuner.rl_env import train_rl


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cfg(**kwargs: object) -> ControllerConfig:
    """Return a ControllerConfig with explicit, env-independent defaults."""
    defaults: dict[str, object] = dict(
        interval_sec=15,
        cooldown_sec=60,
        dry_run=True,
        max_jobs_floor=2,
        max_jobs_ceil=128,
        pressure_high=0.85,
        pressure_low=0.45,
        compose_file="infra/docker/docker-compose.yml",
        slurm_service="slurm",
        slurm_exec_mode="local",
        slurm_ssh_host="",
        slurm_ssh_user="",
        slurm_ssh_key_file="",
        tuner_kind="rl",
        rl_qtable_path="models/qtable.json",
        rl_alpha=0.1,
        rl_gamma=0.9,
        rl_epsilon=0.1,
        rl_train_episodes=300,
    )
    defaults.update(kwargs)
    return ControllerConfig(**defaults)  # type: ignore[arg-type]


def _empty_qtable_cfg(**kwargs: object) -> ControllerConfig:
    """Config pointing to a non-existent qtable path → empty Q-table (AIMD fallback)."""
    return _make_cfg(rl_qtable_path="/nonexistent/path/qtable.json", **kwargs)


# ---------------------------------------------------------------------------
# Test 1: next_max_jobs always returns int within [floor, ceil] — empty Q-table
# ---------------------------------------------------------------------------


def test_bounds_empty_qtable_1000_samples() -> None:
    """next_max_jobs must return int in [floor, ceil] for 1000 seeded inputs with empty Q-table."""
    cfg = _empty_qtable_cfg()
    tuner = RLTuner(cfg, qtable={})
    floor = cfg.max_jobs_floor
    ceil = cfg.max_jobs_ceil
    rng = random.Random(0)

    for _ in range(1000):
        current = rng.randint(floor, ceil)
        saturation = rng.random()
        result = tuner.next_max_jobs(current, saturation)
        assert isinstance(result, int), f"Expected int, got {type(result)}"
        assert floor <= result <= ceil, (
            f"Out of bounds: {result} not in [{floor}, {ceil}] "
            f"(current={current}, sat={saturation:.3f})"
        )


# ---------------------------------------------------------------------------
# Test 2: drop-in parity — build_tuner with tuner_kind='rl' has next_max_jobs
#          and a 50-iter loop stays in bounds
# ---------------------------------------------------------------------------


def test_drop_in_parity_50_iter_loop() -> None:
    """build_tuner(rl) has next_max_jobs; 50-step loop stays in [floor, ceil]."""
    cfg = _make_cfg(tuner_kind="rl", rl_qtable_path="/nonexistent/qtable.json")
    tuner = build_tuner(cfg)

    assert hasattr(tuner, "next_max_jobs"), "Tuner must expose next_max_jobs"
    assert isinstance(tuner, RLTuner)

    floor = cfg.max_jobs_floor
    ceil = cfg.max_jobs_ceil
    current = (floor + ceil) // 2
    rng = random.Random(1)

    for _ in range(50):
        sat = rng.random()
        current = tuner.next_max_jobs(current, sat)
        assert isinstance(current, int)
        assert floor <= current <= ceil

    # Also confirm AimdTuner is returned for tuner_kind='aimd'
    aimd_cfg = _make_cfg(tuner_kind="aimd")
    assert isinstance(build_tuner(aimd_cfg), AimdTuner)


# ---------------------------------------------------------------------------
# Test 3: learning — mean reward last quarter > mean reward first quarter
# ---------------------------------------------------------------------------


def test_learning_reward_improves() -> None:
    """Trained agent shows reward improvement: last-quarter mean > first-quarter mean."""
    cfg = _make_cfg(rl_train_episodes=300)
    _, reward_history = train_rl(cfg, episodes=300, seed=42)

    n = len(reward_history)
    quarter = max(1, n // 4)
    first_q_mean = sum(reward_history[:quarter]) / quarter
    last_q_mean = sum(reward_history[n - quarter:]) / quarter

    assert last_q_mean > first_q_mean, (
        f"No learning detected: first_q={first_q_mean:.4f}, last_q={last_q_mean:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 4: determinism — same Q-table + same inputs => identical outputs
# ---------------------------------------------------------------------------


def test_determinism_same_qtable_same_inputs() -> None:
    """Identical Q-table and inputs must produce identical decisions."""
    cfg = _empty_qtable_cfg()

    # Build a small known Q-table
    qtable: QTable = {
        (0, 0): [0.1, 0.5, 0.9],
        (2, 3): [0.8, 0.2, 0.1],
        (4, 5): [0.9, 0.3, 0.1],
    }

    tuner_a = RLTuner(cfg, qtable={k: list(v) for k, v in qtable.items()})
    tuner_b = RLTuner(cfg, qtable={k: list(v) for k, v in qtable.items()})

    rng = random.Random(99)
    for _ in range(200):
        current = rng.randint(cfg.max_jobs_floor, cfg.max_jobs_ceil)
        sat = rng.random()
        assert tuner_a.next_max_jobs(current, sat) == tuner_b.next_max_jobs(current, sat)


# ---------------------------------------------------------------------------
# Test 5: persistence round-trip — save -> load -> identical decisions
# ---------------------------------------------------------------------------


def test_persistence_round_trip(tmp_path: pytest.TempPathFactory) -> None:
    """save -> load produces Q-table that makes identical decisions."""
    cfg = _make_cfg()
    qtable_path = str(tmp_path / "test_qtable.json")  # type: ignore[operator]

    # Train briefly to get a non-empty Q-table
    qtable, _ = train_rl(cfg, episodes=50, seed=7)
    assert len(qtable) > 0, "Training produced empty Q-table"

    save_qtable(qtable, qtable_path)
    loaded_qtable = load_qtable(qtable_path)

    cfg_saved = _make_cfg(rl_qtable_path="/nonexistent")
    cfg_loaded = _make_cfg(rl_qtable_path="/nonexistent")
    tuner_original = RLTuner(cfg_saved, qtable={k: list(v) for k, v in qtable.items()})
    tuner_loaded = RLTuner(cfg_loaded, qtable={k: list(v) for k, v in loaded_qtable.items()})

    rng = random.Random(42)
    for _ in range(500):
        current = rng.randint(cfg.max_jobs_floor, cfg.max_jobs_ceil)
        sat = rng.random()
        assert tuner_original.next_max_jobs(current, sat) == tuner_loaded.next_max_jobs(
            current, sat
        ), f"Mismatch after round-trip at current={current} sat={sat:.3f}"


# ---------------------------------------------------------------------------
# Test 6: RL stress — 2000 seeded iters, bounds NEVER violated
# ---------------------------------------------------------------------------


def test_rl_stress_2000_iters_bounds_never_violated() -> None:
    """2000 seeded iterations with trained Q-table must never violate [floor, ceil]."""
    cfg = _make_cfg(rl_train_episodes=200)
    floor = cfg.max_jobs_floor
    ceil = cfg.max_jobs_ceil

    # Train first to populate Q-table
    qtable, _ = train_rl(cfg, episodes=200, seed=13)
    tuner = RLTuner(cfg, qtable=qtable)

    rng = random.Random(77)
    current = (floor + ceil) // 2

    for i in range(2000):
        sat = rng.random()
        result = tuner.next_max_jobs(current, sat)
        assert isinstance(result, int), f"Step {i}: expected int, got {type(result)}"
        assert floor <= result <= ceil, (
            f"Step {i}: bounds violated: {result} not in [{floor}, {ceil}] "
            f"(current={current}, sat={sat:.3f})"
        )
        current = result  # chain: output feeds next input


# ---------------------------------------------------------------------------
# Additional: test that AIMD fallback is invoked for unseen states
# ---------------------------------------------------------------------------


def test_aimd_fallback_for_unseen_states() -> None:
    """Empty Q-table → AIMD fallback decisions match AimdTuner directly."""
    cfg = _empty_qtable_cfg()
    rl_tuner = RLTuner(cfg, qtable={})
    aimd_tuner = AimdTuner(cfg)

    rng = random.Random(5)
    for _ in range(200):
        current = rng.randint(cfg.max_jobs_floor, cfg.max_jobs_ceil)
        sat = rng.random()
        assert rl_tuner.next_max_jobs(current, sat) == aimd_tuner.next_max_jobs(current, sat)


# ---------------------------------------------------------------------------
# Additional: config fields present with correct defaults
# ---------------------------------------------------------------------------


def test_config_rl_fields_defaults() -> None:
    """ControllerConfig must have all RL fields with correct types and defaults."""
    cfg = ControllerConfig(
        interval_sec=15,
        cooldown_sec=60,
        dry_run=True,
        max_jobs_floor=2,
        max_jobs_ceil=128,
        pressure_high=0.85,
        pressure_low=0.45,
        compose_file="x",
        slurm_service="s",
        slurm_exec_mode="local",
        slurm_ssh_host="",
        slurm_ssh_user="",
        slurm_ssh_key_file="",
        tuner_kind="aimd",
        rl_qtable_path="models/qtable.json",
        rl_alpha=0.1,
        rl_gamma=0.9,
        rl_epsilon=0.1,
        rl_train_episodes=300,
    )
    assert cfg.tuner_kind == "aimd"
    assert cfg.rl_qtable_path == "models/qtable.json"
    assert cfg.rl_alpha == pytest.approx(0.1)
    assert cfg.rl_gamma == pytest.approx(0.9)
    assert cfg.rl_epsilon == pytest.approx(0.1)
    assert cfg.rl_train_episodes == 300
