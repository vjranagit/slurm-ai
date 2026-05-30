# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`adaptive-wlm-controller` — a local-first MVP that runs a quota-aware adaptive control loop over a Slurm cluster. Each interval it reads scheduler pressure, computes a saturation score, picks a target job-concurrency limit via AIMD, and (outside dry-run) applies it to Slurm. Slurm-only for now; LSF is a planned follow-on. Python 3.11+.

## Commands

```bash
pip install -e ".[dev]"      # editable install with dev tools

ruff check .                 # lint (line-length 100)
pytest -q                    # tests (CI runs exactly: ruff check . && pytest -q)
pytest controller/tests/test_aimd.py::test_aimd_increase_on_low_pressure -q   # single test

# Control loop. dry-run is the default; pass the string "true"/"false".
python -m controller.main --interval 15 --dry-run true

# API / dashboard (separate process from the loop)
uvicorn apps.api.main:app --host 0.0.0.0 --port 8080

# Offline policy eval — no Slurm needed; CSV cols: pending_jobs,running_jobs,saturation
python -m controller.simulator.replay <trace.csv>

# Local Slurm sandbox + Prometheus + Grafana (docker compose)
./scripts/up.sh                  # build, start, wait for `scontrol ping`
./scripts/down.sh                # down -v (removes volumes)
./scripts/submit-test-jobs.sh 20 # sbatch N sleep-jobs into the sandbox
./scripts/e2e-check.sh           # up → submit → one dry-run pass → API health hint
```

`pyproject.toml` sets `pythonpath = ["."]`, so tests and modules import as `controller...` / `apps...` from the repo root.

## Architecture

One linear pipeline, run once per interval in `controller/main.py:run`:

```
collect → saturation_score → AIMD next_max_jobs → fairshare_weight → actuator.apply → set metrics → sleep
```

Components are plain classes constructed from a single `ControllerConfig`:

- **`controller/collectors/slurm.py`** — `SlurmCollector.snapshot()` shells `sinfo -h -o '%t'`, `squeue -h -o '%T|%r'`, and (best-effort) `sacct`, then builds a `ClusterSnapshot`. `cpu_pressure = pending/(pending+running)`; `mem_pressure`/`io_pressure` are heuristic functions of the same counts (not real mem/IO telemetry).
- **`controller/policy/engine.py`** — pure threshold logic, no state. `saturation_score = max(cpu, mem, io)`. `fairshare_weight` returns `5000` (≥`pressure_high`) / `1000` (≤`pressure_low`) / `2500` otherwise.
- **`controller/tuner/aimd.py`** — `AimdTuner.next_max_jobs(current, saturation)`: multiplicative-decrease `current // 2` when `saturation ≥ pressure_high`, additive-increase `current + 1` when `≤ pressure_low`, hold otherwise; clamped to `[max_jobs_floor, max_jobs_ceil]`. Stateless — the baseline `current` comes from the actuator's state.
- **`controller/actuator/slurm.py`** — `SlurmActuator.apply()` owns the **cooldown** (skips and returns `changed=False` if within `cooldown_sec` of the last apply) and re-clamps bounds. It also holds the carried-over `ControllerState` (`max_jobs` init `max(floor, 8)`, `priority_weight_fs` init `1000`). In live mode it runs: if `slurm.conf` has slurmdbd accounting → `sacctmgr -i modify qos normal set MaxJobsPU=<n>`, else `scontrol update PartitionName=debug MaxCPUsPerNode=<n>`; then `scontrol reconfigure`. In dry-run it only logs `DRY_RUN <cmd>`.

Shared infrastructure:

- **`controller/slurm_exec.py`** — `SlurmCommandRunner.run()` is the single choke point for every Slurm CLI call. `mode` (`SLURM_EXEC_MODE`) selects the wrapper: `docker` → `docker compose exec -T <service> bash -lc`, `local` → `bash -lc` on the host, `ssh` → `ssh ... user@host bash -lc`. Returns `(ok: bool, output: str)`; never raises on command failure. Any new scheduler interaction must go through here, not raw `subprocess`.
- **`controller/config.py`** — `ControllerConfig` dataclass. **Defaults are read from environment variables at class-definition (import) time**, so env vars (or `.env` exported into the shell) must be set *before* importing the package, not after. `controller/main.py` additionally lets `--interval` / `--dry-run` override the constructed config.
- **`controller/types.py`** — data contracts: `ClusterSnapshot`, `PolicyDecision` (declared but the loop currently passes raw ints, not this type), `AppliedAction`.
- **`controller/metrics.py`** — module-level `prometheus_client` `Gauge` singletons on the default registry: `adaptive_saturation_score`, `adaptive_pending_jobs`, `adaptive_running_jobs`, `adaptive_target_max_jobs`, `adaptive_priority_weight_fs`, `adaptive_action_changed`. The loop process exposes them via `start_http_server(9108)`.

Entry points:

- **`apps/api/main.py`** — FastAPI app, **separate process** from the control loop. It builds only a `SlurmCollector` (not the policy/tuner/actuator). Endpoints: `GET /`, `/healthz`, `/cluster/snapshot` (live `ClusterSnapshot`), `/metrics`, `/favicon.ico`, and static `/ui` (the dashboard). Because it's a different process, `/metrics` reflects the API process's registry, not the running loop's `:9108`.
- **`apps/dashboard/`** — static `index.html`, served at `/ui`.
- **`infra/`** — `docker/` (single-container Slurm sandbox + `docker-compose.yml`), `prometheus/`, `grafana/` provisioning; all driven by `scripts/`.

## Safety invariants (don't weaken without intent)

- `dry_run` defaults to `true`; only the explicit `--dry-run false` / `CONTROLLER_DRY_RUN=false` path mutates Slurm.
- Bounds `[max_jobs_floor, max_jobs_ceil]` are enforced in **both** the tuner and the actuator.
- The cooldown lives in the actuator (`SlurmActuator.apply`) — keep it there.
- Only mutation is `sacctmgr modify qos` / `scontrol update` + `reconfigure`; no destructive scheduler ops.

## Gotchas

- The README/`docs/` describe some aspirational behavior (e.g. `/status`, `/docs`, fairshare); trust the source — the API surface and tuning knobs above are what's actually implemented.
- The single-container Docker Slurm sandbox usually can't launch job steps (no full cgroup/systemd), so submitted jobs stay `PD`. Fine for testing collector/policy/tuner/actuator; for real `RUNNING → COMPLETED` point the controller at a real cluster via `SLURM_EXEC_MODE=ssh`.
