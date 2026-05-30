from controller.config import ControllerConfig
from controller.tuner.aimd import AimdTuner


def test_aimd_decrease_on_high_pressure() -> None:
    cfg = ControllerConfig()
    tuner = AimdTuner(cfg)
    assert tuner.next_max_jobs(16, cfg.pressure_high + 0.01) == 8


def test_aimd_increase_on_low_pressure() -> None:
    cfg = ControllerConfig()
    tuner = AimdTuner(cfg)
    assert tuner.next_max_jobs(8, cfg.pressure_low - 0.01) == 9


# ---------------------------------------------------------------------------
# Fix 3 regression: hold path must clamp to [floor, ceil]
# ---------------------------------------------------------------------------


def test_aimd_hold_clamps_above_ceil() -> None:
    """Hold path must clamp current down to max_jobs_ceil."""
    cfg = ControllerConfig(max_jobs_floor=2, max_jobs_ceil=10)
    tuner = AimdTuner(cfg)
    mid_pressure = (cfg.pressure_low + cfg.pressure_high) / 2
    result = tuner.next_max_jobs(999, mid_pressure)
    assert result == 10


def test_aimd_hold_clamps_below_floor() -> None:
    """Hold path must clamp current up to max_jobs_floor."""
    cfg = ControllerConfig(max_jobs_floor=5, max_jobs_ceil=128)
    tuner = AimdTuner(cfg)
    mid_pressure = (cfg.pressure_low + cfg.pressure_high) / 2
    result = tuner.next_max_jobs(0, mid_pressure)
    assert result == 5


def test_aimd_hold_returns_current_when_in_bounds() -> None:
    """Hold path must return current unchanged when already in [floor, ceil]."""
    cfg = ControllerConfig(max_jobs_floor=2, max_jobs_ceil=128)
    tuner = AimdTuner(cfg)
    mid_pressure = (cfg.pressure_low + cfg.pressure_high) / 2
    result = tuner.next_max_jobs(20, mid_pressure)
    assert result == 20
