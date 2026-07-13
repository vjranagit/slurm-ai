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


def test_load_qtable_corrupt_json_returns_empty_no_raise(tmp_path: pytest.TempPathFactory) -> None:
    """A truncated/corrupt JSON file must not raise — load_qtable falls back to {}."""
    path = str(tmp_path / "corrupt.json")  # type: ignore[operator]
    with open(path, "w") as fh:
        fh.write('{"0": {"0": [0.1, 0.2, 0.3]')  # truncated JSON, missing closing braces

    result = load_qtable(path)
    assert result == {}


def test_load_qtable_non_json_garbage_returns_empty(tmp_path: pytest.TempPathFactory) -> None:
    path = str(tmp_path / "garbage.json")  # type: ignore[operator]
    with open(path, "w") as fh:
        fh.write("this is not json at all {{{")

    result = load_qtable(path)
    assert result == {}


def test_load_qtable_corrupt_file_falls_back_to_aimd_behavior() -> None:
    """A corrupt qtable path used by RLTuner must behave exactly like AIMD (no crash)."""
    cfg = _make_cfg(rl_qtable_path="__corrupt_will_be_created__")
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fh:
        fh.write("{not valid json")
        corrupt_path = fh.name

    cfg = _make_cfg(rl_qtable_path=corrupt_path)
    rl_tuner = RLTuner(cfg)
    aimd_tuner = AimdTuner(cfg)

    rng = random.Random(11)
    for _ in range(100):
        current = rng.randint(cfg.max_jobs_floor, cfg.max_jobs_ceil)
        sat = rng.random()
        assert rl_tuner.next_max_jobs(current, sat) == aimd_tuner.next_max_jobs(current, sat)


def test_json_to_qtable_skips_wrong_length_action_list() -> None:
    """An entry whose action-value list length != _N_ACTIONS is ignored, not crashed on."""
    from controller.tuner.rl import _json_to_qtable

    data = {
        "0": {"0": [0.1, 0.2, 0.3]},  # valid: length 3
        "1": {"1": [0.1, 0.2]},  # malformed: length 2 -> must be skipped
        "2": {"2": [0.1, 0.2, 0.3, 0.4]},  # malformed: length 4 -> must be skipped
    }
    qtable = _json_to_qtable(data)  # type: ignore[arg-type]
    assert (0, 0) in qtable
    assert (1, 1) not in qtable
    assert (2, 2) not in qtable
    assert len(qtable) == 1


def test_json_to_qtable_skips_nan_inf_action_values(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """NaN/inf/-inf Q-values must be dropped, not accepted as valid floats.

    Python's json module parses the bare literals NaN/Infinity/-Infinity by
    default, so a Q-value of NaN would previously pass the
    isinstance(v, (int, float)) check and silently corrupt the greedy argmax
    in next_max_jobs (NaN comparisons are always False, biasing selection
    toward the first/DECREASE action). Such entries must now be skipped,
    exactly like a wrong-length entry, falling back to AIMD for that state.
    """
    from controller.tuner.rl import _json_to_qtable

    # Raw JSON string with a bare NaN literal (valid per Python's json parser).
    path = str(tmp_path / "nan_qtable.json")  # type: ignore[operator]
    with open(path, "w") as fh:
        fh.write('{"0": {"0": [NaN, 1.0, 2.0]}}')

    result = load_qtable(path)
    assert result == {}, "entry with NaN Q-value must be dropped"

    # Also exercise inf / -inf directly via _json_to_qtable with float().
    data = {
        "0": {"0": [float("nan"), 1.0, 2.0]},
        "1": {"1": [float("inf"), 0.5, 0.2]},
        "2": {"2": [float("-inf"), 0.5, 0.2]},
        "3": {"3": [0.1, 0.5, 0.9]},  # valid finite entry, must be kept
    }
    qtable = _json_to_qtable(data)  # type: ignore[arg-type]
    assert (0, 0) not in qtable
    assert (1, 1) not in qtable
    assert (2, 2) not in qtable
    assert (3, 3) in qtable
    assert qtable[(3, 3)] == [0.1, 0.5, 0.9]
    assert len(qtable) == 1

    # RLTuner must still construct and stay in bounds when a NaN entry is
    # present alongside otherwise-valid data.
    cfg = _make_cfg(rl_qtable_path=path)
    tuner = RLTuner(cfg)
    rng = random.Random(17)
    for _ in range(200):
        current = rng.randint(cfg.max_jobs_floor, cfg.max_jobs_ceil)
        sat = rng.random()
        out = tuner.next_max_jobs(current, sat)
        assert cfg.max_jobs_floor <= out <= cfg.max_jobs_ceil


def test_json_to_qtable_normal_finite_table_loads_fully_no_regression() -> None:
    """A normal Q-table with only finite values must load every entry unchanged."""
    from controller.tuner.rl import _json_to_qtable

    data = {
        "0": {"0": [0.1, 0.5, 0.9], "1": [1.0, -1.0, 0.0]},
        "1": {"2": [3.5, 2.5, -2.5]},
        "4": {"5": [0.0, 0.0, 0.0]},
    }
    qtable = _json_to_qtable(data)  # type: ignore[arg-type]
    assert len(qtable) == 4
    assert qtable[(0, 0)] == [0.1, 0.5, 0.9]
    assert qtable[(0, 1)] == [1.0, -1.0, 0.0]
    assert qtable[(1, 2)] == [3.5, 2.5, -2.5]
    assert qtable[(4, 5)] == [0.0, 0.0, 0.0]


def test_malformed_qtable_entry_does_not_crash_next_max_jobs(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """A qtable with a malformed entry must not raise IndexError on next_max_jobs."""
    import json as _json

    path = str(tmp_path / "malformed.json")  # type: ignore[operator]
    with open(path, "w") as fh:
        _json.dump({"0": {"0": [0.1, 0.2]}, "1": {"1": [0.5, 0.1, 0.9]}}, fh)

    cfg = _make_cfg(rl_qtable_path=path)
    tuner = RLTuner(cfg)

    rng = random.Random(3)
    for _ in range(200):
        current = rng.randint(cfg.max_jobs_floor, cfg.max_jobs_ceil)
        sat = rng.random()
        result = tuner.next_max_jobs(current, sat)
        assert cfg.max_jobs_floor <= result <= cfg.max_jobs_ceil


def test_json_to_qtable_inner_value_not_dict_returns_empty() -> None:
    """{'0': 5} — inner value is an int, not a dict of maxjobs-buckets.

    Previously crashed with AttributeError: 'int' object has no attribute 'items'.
    Must now be skipped, yielding an empty table (no raise).
    """
    from controller.tuner.rl import _json_to_qtable

    qtable = _json_to_qtable({"0": 5})  # type: ignore[arg-type]
    assert qtable == {}


def test_json_to_qtable_action_value_not_list_returns_empty() -> None:
    """{'0': {'0': 5}} — action value is an int, not a list of Q-values.

    Previously crashed with TypeError: object of type 'int' has no len().
    Must now be skipped, yielding an empty table (no raise).
    """
    from controller.tuner.rl import _json_to_qtable

    qtable = _json_to_qtable({"0": {"0": 5}})  # type: ignore[arg-type]
    assert qtable == {}


def test_load_qtable_inner_value_not_dict_falls_back_to_empty(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """load_qtable on {'0': 5} on disk must not raise; returns {} (AIMD fallback)."""
    import json as _json

    path = str(tmp_path / "wrong_shape_1.json")  # type: ignore[operator]
    with open(path, "w") as fh:
        _json.dump({"0": 5}, fh)

    result = load_qtable(path)
    assert result == {}

    cfg = _make_cfg(rl_qtable_path=path)
    tuner = RLTuner(cfg)  # must construct without raising
    assert tuner.next_max_jobs(10, 0.5) is not None


def test_load_qtable_action_value_not_list_falls_back_to_empty(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """load_qtable on {'0': {'0': 5}} on disk must not raise; returns {} (AIMD fallback)."""
    import json as _json

    path = str(tmp_path / "wrong_shape_2.json")  # type: ignore[operator]
    with open(path, "w") as fh:
        _json.dump({"0": {"0": 5}}, fh)

    result = load_qtable(path)
    assert result == {}

    cfg = _make_cfg(rl_qtable_path=path)
    tuner = RLTuner(cfg)  # must construct without raising
    assert tuner.next_max_jobs(10, 0.5) is not None


def test_load_qtable_truncated_json_via_load_qtable_returns_empty(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Truncated JSON loaded through load_qtable() must not raise; returns {}."""
    path = str(tmp_path / "truncated.json")  # type: ignore[operator]
    with open(path, "w") as fh:
        fh.write('{"0": {"0": [0.1, 0.2, 0.3]}')  # missing closing brace

    result = load_qtable(path)
    assert result == {}

    cfg = _make_cfg(rl_qtable_path=path)
    tuner = RLTuner(cfg)  # must construct without raising
    assert tuner.next_max_jobs(10, 0.5) is not None


def test_load_qtable_mixed_valid_and_wrong_length_entry_keeps_valid(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """A table with one valid entry and one wrong-length entry must not raise;
    the valid entry is preserved, the malformed one is dropped, and RLTuner
    still constructs and next_max_jobs still works end-to-end."""
    import json as _json

    path = str(tmp_path / "mixed.json")  # type: ignore[operator]
    with open(path, "w") as fh:
        _json.dump(
            {
                "0": {"0": [0.1, 0.5, 0.9]},  # valid: length 3
                "1": {"1": [0.1, 0.2]},  # malformed: length 2
            },
            fh,
        )

    result = load_qtable(path)
    assert (0, 0) in result
    assert result[(0, 0)] == [0.1, 0.5, 0.9]
    assert (1, 1) not in result
    assert len(result) == 1

    cfg = _make_cfg(rl_qtable_path=path)
    tuner = RLTuner(cfg)  # must construct without raising

    rng = random.Random(21)
    for _ in range(200):
        current = rng.randint(cfg.max_jobs_floor, cfg.max_jobs_ceil)
        sat = rng.random()
        out = tuner.next_max_jobs(current, sat)
        assert cfg.max_jobs_floor <= out <= cfg.max_jobs_ceil


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
