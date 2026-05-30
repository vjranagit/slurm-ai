# End-to-end test results

Environment: local Docker Slurm sandbox (`infra/docker/docker-compose.yml`, service `slurm`,
slurmctld UP, partition `debug`, 1 node / 8 CPUs, node state `unk` — slurmd not registered, so
submitted jobs stay PENDING). Note: `slurm.example.com` was probed but has **no Slurm binaries
installed**, so the Docker sandbox is the live Slurm test target, not the SSH host.

## Unit + stress suite (pre-RL)
- `160 passed` via `.venv/bin/python -m pytest -q`; `ruff check .` clean.
- Covers: config (env override via subprocess), slurm_exec (docker/local/ssh command build +
  real local run), collector (canned sinfo/squeue/sacct parsing + pressure math + edge cases),
  policy, AIMD tuner (all branches + bound clamps + boundaries), actuator
  (dry-run/live/cooldown/clamp/changed-flag), simulator replay, API (TestClient), and stress
  invariants (2000 iters: bounds never violated; sustained pressure converges to floor/ceil).

## Live E2E against real slurmctld (sandbox)
- Submitted jobs → all PENDING; collector read pending>0, **saturation = 1.0** from the real
  slurmctld (total_nodes=1).
- Dry-run control loop (cooldown=0): AIMD **decrease path 8 → 4 → 2 → 2** (clamps at floor=2) on
  real snapshot data. Verified.
- LIVE actuation (dry_run=false): the actuator's commands executed and returned OK. Isolated
  probe of the underlying commands:
  - `scontrol update PartitionName=debug MaxCPUsPerNode=4` → partition shows `MaxCPUsPerNode=4`.
  - `scontrol reconfigure` (the actuator runs this next) → partition **reverts to `UNLIMITED`**,
    because the sandbox `slurm.conf` does not persist `MaxCPUsPerNode`.

## Bugs found during E2E / testing (see docs/IMPLEMENTATION.md "Known issues")
1. **`/metrics` serves no `adaptive_*` metrics** — `apps/api/main.py` never imports
   `controller/metrics.py`, so those gauges are never registered in the default Prometheus
   registry for the API process.
2. **Actuator fallback path is non-persistent** — when slurmdbd accounting is absent, the
   actuator uses `scontrol update PartitionName=... MaxCPUsPerNode=N` then `scontrol
   reconfigure`; the reconfigure reverts the just-applied value (not written to slurm.conf). The
   primary path (`sacctmgr modify qos ... MaxJobsPU`) persists in the accounting DB and is not
   reverted by reconfigure, so this affects only no-slurmdbd deployments.
3. **`ControllerConfig` reads env at import time** (dataclass field defaults), so env overrides
   must be set before first import; runtime `os.environ` changes have no effect.

## Safety notes
- Live scheduler mutation was exercised only against the disposable Docker sandbox.
- `slurm.example.com` (no Slurm) and any shared cluster are never mutated by these tests.
