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
