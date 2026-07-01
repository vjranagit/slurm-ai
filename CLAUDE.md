# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`adaptive-wlm-controller` — a local-first MVP that runs a quota-aware adaptive control loop over a Slurm cluster. Each interval it reads scheduler pressure, computes a saturation score, picks a target job-concurrency limit via a pluggable tuner (AIMD or RL), and (outside dry-run) applies it to Slurm. Slurm-only for now; LSF is a planned follow-on. Python 3.11+ (the dev venv runs 3.10 fine).

## Commands

```bash
pip install -e ".[dev]"      # editable install with dev tools

ruff check .                 # lint (line-length 100)
pytest -q                    # 182 tests (CI runs exactly: ruff check . && pytest -q)
pytest controller/tests/test_actuator.py -q          # single file
pytest controller/tests/test_rl.py::test_bounds -q   # single test (example)

# Control loop. dry-run is the default; pass the string "true"/"false".
python -m controller.main --interval 15 --dry-run true

# RL tuner instead of AIMD: train a Q-table, then select it via env
python -m controller.tuner.train                     # writes models/qtable.json (seed=42)
TUNER_KIND=rl python -m controller.main --dry-run true

# API / dashboard (separate process from the loop)
uvicorn apps.api.main:app --host 0.0.0.0 --port 8080

# Offline policy eval — no Slurm needed; CSV cols: pending_jobs,running_jobs,saturation
python -m controller.simulator.replay <trace.csv>

# Local Slurm sandbox + Prometheus + Grafana (docker compose)
./scripts/up.sh                  # build, start, wait for `scontrol ping`
./scripts/down.sh                # down -v (removes volumes)
./scripts/submit-test-jobs.sh 20 # sbatch N sleep-jobs into the sandbox
./scripts/e2e-check.sh           # up → submit → squeue/sacct checks
```

`pyproject.toml` sets `pythonpath = ["."]`, so tests and modules import as `controller...` / `apps...` from the repo root.

## Architecture

One linear pipeline, run once per interval in `controller/main.py:run`:

```
collect → saturation_score → tuner.next_max_jobs → fairshare_weight → actuator.apply → set metrics → sleep
```

Components are plain classes constructed from a single `ControllerConfig`:

- **`controller/collectors/slurm.py`** — `SlurmCollector.snapshot()` shells `sinfo -h -o '%t'`, `squeue -h -o '%T|%r'`, and (best-effort) `sacct`, then builds a `ClusterSnapshot`. `cpu_pressure = pending/(pending+running)`; `mem_pressure`/`io_pressure` are heuristic functions of the same counts (not real mem/IO telemetry).
- **`controller/policy/engine.py`** — pure threshold logic, no state. `saturation_score = max(cpu, mem, io)`. `fairshare_weight` returns `5000` (≥`pressure_high`) / `1000` (≤`pressure_low`) / `2500` otherwise.
- **`controller/tuner/`** — pluggable tuner with one contract: `next_max_jobs(current, saturation) -> int`, always within `[max_jobs_floor, max_jobs_ceil]`.
  - `aimd.py` (`AimdTuner`, default): multiplicative-decrease `current // 2` when `saturation ≥ pressure_high`, additive-increase `current + 1` when `≤ pressure_low`, hold otherwise; clamped. Stateless — baseline `current` comes from the actuator's state.
  - `rl.py` (`RLTuner`): tabular Q-learning over a discretized `(pressure, max-jobs)` state with the same decrease/hold/increase primitives; loads a Q-table from `rl_qtable_path` (JSON) and falls back to AIMD for unseen states. `rl_env.py` has the seeded `QueueSimulator` + `train_rl()`; train via `python -m controller.tuner.train`.
  - `__init__.py` `build_tuner(cfg)` returns `RLTuner` if `TUNER_KIND=rl` else `AimdTuner`; `controller/main.py` calls it, so switching needs no loop change.
- **`controller/actuator/slurm.py`** — `SlurmActuator.apply()` owns the **cooldown** (skips and returns `changed=False` if within `cooldown_sec` of the last apply) and re-clamps bounds. Holds the carried-over `ControllerState` (`max_jobs` init `max(floor, 8)`, `priority_weight_fs` init `1000`). In live mode it runs one conditional command: if `slurm.conf` has slurmdbd accounting → `sacctmgr -i modify qos normal set MaxJobsPU=<n>`, else `scontrol update PartitionName=debug MaxCPUsPerNode=<n>`. It deliberately does **not** run `scontrol reconfigure` — that re-read `slurm.conf` and reverted the non-slurmdbd partition change. In dry-run it only logs `DRY_RUN <cmd>`.

Shared infrastructure:

- **`controller/slurm_exec.py`** — `SlurmCommandRunner.run()` is the single choke point for every Slurm CLI call. `mode` (`SLURM_EXEC_MODE`) selects the wrapper: `docker` → `docker compose exec -T <service> bash -lc`, `local` → `bash -lc` on the host, `ssh` → `ssh ... user@host bash -lc`. Returns `(ok: bool, output: str)`; never raises on command failure. Any new scheduler interaction must go through here, not raw `subprocess`.
- **`controller/config.py`** — `ControllerConfig` dataclass holds all tunables (thresholds, AIMD bounds, cooldown, exec mode, plus RL settings: `tuner_kind`, `rl_qtable_path`, `rl_alpha`, `rl_gamma`, `rl_epsilon`, `rl_train_episodes`). Field defaults call `os.getenv(...)` at **import time** (see `.env.example`); there is no `load_config()`, so env vars must be set before the package is first imported. `controller/main.py` additionally lets `--interval` / `--dry-run` override the constructed config.
- **`controller/types.py`** — data contracts: `ClusterSnapshot`, `PolicyDecision` (declared but the loop currently passes raw ints, not this type), `AppliedAction`.
- **`controller/metrics.py`** — module-level `prometheus_client` `Gauge` singletons on the default registry: `adaptive_saturation_score`, `adaptive_pending_jobs`, `adaptive_running_jobs`, `adaptive_target_max_jobs`, `adaptive_priority_weight_fs`, `adaptive_action_changed`. The loop process exposes them via `start_http_server(9108)`.
- **`controller/workload/submitter.py`** — synthetic job-submission helper backing `scripts/submit-test-jobs.sh` (`python -m controller.workload.submitter`); builds N sleep-job scripts and `sbatch`s them through `SlurmCommandRunner`, using the same `docker`/`local`/`ssh` exec plumbing as the collector and actuator.

Entry points:

- **`apps/api/main.py`** — FastAPI app, **separate process** from the control loop. Builds only a `SlurmCollector` (not policy/tuner/actuator), and imports `controller.metrics` so the `adaptive_*` gauges register in this process. Endpoints: `GET /`, `/healthz`, `/cluster/snapshot` (live `ClusterSnapshot`), `/metrics`, `/favicon.ico`, and static `/ui` (the dashboard). It is a different process from the loop, so the loop's live values are on its own `:9108`; the API's `/metrics` shows the gauges (zeroed unless that process updates them).
- **`apps/dashboard/`** — static `index.html`, served at `/ui`.
- **`infra/`** — `docker/` (single-container Slurm sandbox + `docker-compose.yml`), `prometheus/`, `grafana/` provisioning; all driven by `scripts/`.

## Testing

`pytest -q` runs 182 tests covering config (env override via subprocess), `slurm_exec` (docker/local/ssh command build + real local run), collector parsing + pressure math + edge cases, policy, AIMD + RL tuners, actuator (dry-run/live/cooldown/bounds/changed-flag), simulator replay, the API (TestClient), 2000-iteration stress invariants, and regression tests for the two bugs below. The Slurm boundary (`SlurmCommandRunner.run`) is monkeypatched — no test touches a real cluster. End-to-end runs against the Docker sandbox are recorded in `docs/E2E_RESULTS.md`.

## Safety invariants (don't weaken without intent)

- `dry_run` defaults to `true`; only the explicit `--dry-run false` / `CONTROLLER_DRY_RUN=false` path mutates Slurm.
- Bounds `[max_jobs_floor, max_jobs_ceil]` are enforced in **both** the tuner (AIMD and RL) and the actuator.
- The cooldown lives in the actuator (`SlurmActuator.apply`) — keep it there.
- Only mutation is `sacctmgr modify qos` / `scontrol update`; no `scontrol reconfigure`, no destructive scheduler ops.

## Gotchas

- `ControllerConfig` reads env at import time (see above) — set env before importing.
- The single-container Docker Slurm sandbox usually can't launch job steps (no full cgroup/systemd), so submitted jobs stay `PD`. Fine for testing collector/policy/tuner/actuator; for real `RUNNING → COMPLETED` point the controller at a real cluster via `SLURM_EXEC_MODE=ssh`.
- The no-slurmdbd actuator fallback (`scontrol update ... MaxCPUsPerNode`) applies at runtime but is not persisted to `slurm.conf`; the primary `sacctmgr` QoS path persists in the accounting DB. See `docs/E2E_RESULTS.md`.
