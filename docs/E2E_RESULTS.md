# End-to-end test results

Environment: local Docker Slurm sandbox (`infra/docker/docker-compose.yml`, service `slurm`,
slurmctld UP, partition `debug`, 1 node / 8 CPUs, node state `unk` — slurmd not registered, so
submitted jobs stay PENDING). An external SSH host was probed but had no Slurm binaries
installed, so the Docker sandbox is the live Slurm test target.

## Unit + stress suite (historical baseline, pre-RL)

- `160 passed` via `.venv/bin/python -m pytest -q`; `ruff check .` clean.
- Covers: config (env override via subprocess), slurm_exec (docker/local/ssh command build +
  real local run), collector (canned sinfo/squeue/sacct parsing + pressure math + edge cases),
  policy, AIMD tuner (all branches + bound clamps + boundaries), actuator
  (dry-run/live/cooldown/clamp/changed-flag), simulator replay, API (TestClient), and stress
  invariants (2000 iters: bounds never violated; sustained pressure converges to floor/ceil).

**Current test count: 182 passed** (as of the hardening pass that added RL tuner, regression
tests for bugs 1–4, and stress tests). See the Testing section below.

## Live E2E against real slurmctld (sandbox)

- Submitted jobs → all PENDING; collector read pending>0, **saturation = 1.0** from the real
  slurmctld (total_nodes=1).
- Dry-run control loop (cooldown=0): AIMD **decrease path 8 → 4 → 2 → 2** (clamps at floor=2)
  on real snapshot data. Verified.
- LIVE actuation (dry_run=false): the actuator's commands executed and returned OK. Isolated
  probe of the underlying commands confirmed the fixed behavior:
  - `scontrol update PartitionName=debug MaxCPUsPerNode=4` → partition shows `MaxCPUsPerNode=4`.
  - The `scontrol reconfigure` step was removed (bug 2 fix); the runtime change now persists
    (`UNLIMITED → 4` remains) instead of reverting to `UNLIMITED`.

## Bugs found during E2E and fixed

All three issues below were identified during end-to-end testing and are now fixed in the
codebase. Regression tests are in `controller/tests/` (see `test_bugfixes.py`,
`test_actuator.py`, `test_aimd.py`, `test_slurm_exec.py`).

1. **`/metrics` exposed no `adaptive_*` gauges** — `apps/api/main.py` never imported
   `controller/metrics.py`, so the gauges were never registered in the API process's Prometheus
   registry. This was fixed by adding `from controller import metrics as _metrics` to
   `apps/api/main.py`; the import side-effect registers all `adaptive_*` gauges. Regression test
   in `controller/tests/test_bugfixes.py`.

2. **Actuator `scontrol reconfigure` reverted the fallback change** — on clusters without
   slurmdbd accounting, the actuator set `scontrol update PartitionName=... MaxCPUsPerNode=N`
   and then ran `scontrol reconfigure`; the reconfigure re-read `slurm.conf` and reverted the
   just-applied value (confirmed live in the sandbox: `UNLIMITED → 4 → UNLIMITED`). This was
   fixed by removing the `scontrol reconfigure` step entirely. The `sacctmgr` QoS path persists
   in the accounting DB and does not need reconfigure; the `scontrol update` runtime change now
   persists for the session (`UNLIMITED → 4` confirmed stable). Regression test in
   `controller/tests/test_bugfixes.py`.

3. **`ControllerConfig` reads env at import time** (dataclass field defaults) — documented
   behavior, not a bug. Env overrides must be set before first import. Config tests use a
   subprocess to validate overrides correctly.

4. **`SlurmCommandRunner.run()` returned hardcoded `True` when `check=False`** — the `run()`
   method called `subprocess.run(..., check=False)` but always returned `True` for `ok`, ignoring
   the actual returncode. This was fixed by returning `proc.returncode == 0` in all non-exception
   paths. Regression tests in `controller/tests/test_slurm_exec.py`.

5. **Actuator state advanced on failed live command** — when a live Slurm command returned
   `ok=False`, the actuator still updated `state.max_jobs`, `state.priority_weight_fs`, and
   `last_apply`, causing the next interval to start from a phantom applied state. This was fixed
   so that state and `last_apply` are only advanced when `all_ok=True` (or dry_run). On failure
   the actuator returns `changed=False` and leaves state frozen. Regression tests in
   `controller/tests/test_actuator.py`.

6. **Actuator initial `max_jobs` could violate `max_jobs_ceil`** — the init expression
   `max(floor, 8)` could produce a value above `max_jobs_ceil` when ceil was configured below 8.
   This was fixed to `max(floor, min(ceil, 8))`. Regression tests in
   `controller/tests/test_actuator.py`.

7. **AIMD hold path did not clamp to `[floor, ceil]`** — the hold branch returned `current`
   unchanged even when `current` was outside `[max_jobs_floor, max_jobs_ceil]`. This was fixed
   by clamping `current` in the hold path. Regression tests in
   `controller/tests/test_aimd.py`.

## Safety notes

- Live scheduler mutation was exercised only against the disposable Docker sandbox.
- The external SSH host (no Slurm) and any shared cluster are never mutated by these tests.
