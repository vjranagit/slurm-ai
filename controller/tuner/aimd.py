from controller.config import ControllerConfig


class AimdTuner:
    def __init__(self, cfg: ControllerConfig) -> None:
        self.cfg = cfg

    def next_max_jobs(self, current: int, saturation: float) -> int:
        if saturation >= self.cfg.pressure_high:
            next_value = max(self.cfg.max_jobs_floor, current // 2)
        elif saturation <= self.cfg.pressure_low:
            next_value = min(self.cfg.max_jobs_ceil, current + 1)
        else:
            next_value = current
        return next_value
