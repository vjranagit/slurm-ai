from controller.config import ControllerConfig
from controller.types import ClusterSnapshot


class PolicyEngine:
    def __init__(self, cfg: ControllerConfig) -> None:
        self.cfg = cfg

    def saturation_score(self, snap: ClusterSnapshot) -> float:
        return max(snap.cpu_pressure, snap.mem_pressure, snap.io_pressure)

    def fairshare_weight(self, saturation: float) -> int:
        if saturation >= self.cfg.pressure_high:
            return 5000
        if saturation <= self.cfg.pressure_low:
            return 1000
        return 2500
