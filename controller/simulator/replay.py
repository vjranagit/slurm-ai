from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass


@dataclass(slots=True)
class ReplayResult:
    avg_pending: float
    avg_running: float
    avg_saturation: float


def replay_csv(path: str) -> ReplayResult:
    pending_sum = 0.0
    running_sum = 0.0
    sat_sum = 0.0
    n = 0

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pending = float(row.get("pending_jobs", 0))
            running = float(row.get("running_jobs", 0))
            saturation = float(row.get("saturation", 0))
            pending_sum += pending
            running_sum += running
            sat_sum += saturation
            n += 1

    if n == 0:
        raise ValueError("trace file is empty")

    return ReplayResult(
        avg_pending=pending_sum / n,
        avg_running=running_sum / n,
        avg_saturation=sat_sum / n,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay workload traces")
    parser.add_argument("trace_csv", help="CSV with pending_jobs,running_jobs,saturation columns")
    args = parser.parse_args()
    res = replay_csv(args.trace_csv)
    print(
        f"avg_pending={res.avg_pending:.3f} avg_running={res.avg_running:.3f} "
        f"avg_saturation={res.avg_saturation:.3f}"
    )
