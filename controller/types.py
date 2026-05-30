from dataclasses import dataclass, field
from datetime import datetime


@dataclass(slots=True)
class ClusterSnapshot:
    timestamp: datetime
    total_nodes: int
    idle_nodes: int
    alloc_nodes: int
    down_nodes: int
    pending_jobs: int
    running_jobs: int
    completed_jobs_5m: int
    cpu_pressure: float
    mem_pressure: float
    io_pressure: float
    pending_reasons: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class PolicyDecision:
    target_max_jobs: int
    priority_weight_fs: int
    reason: str


@dataclass(slots=True)
class AppliedAction:
    changed: bool
    old_max_jobs: int
    new_max_jobs: int
    old_priority_weight_fs: int
    new_priority_weight_fs: int
    command_log: list[str]
