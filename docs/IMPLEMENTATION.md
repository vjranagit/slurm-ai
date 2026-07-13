# Implementation Notes

## Control loop

The controller runs a bounded loop:

1. Collect snapshot from Slurm (`sinfo`, `squeue`, optional `sacct`)
2. Build saturation score from queue pressure
3. Compute policy outputs (fairshare proxy weight + target max jobs)
4. Apply the configured tuner (AIMD or RL — see "Tuner backends") to the target max jobs
5. Actuate changes with cooldown and bounds
6. Emit metrics + structured logs

The tuner is pluggable via `controller.tuner.build_tuner(cfg)`, selected by `TUNER_KIND`
(`aimd` default, `rl`). Both tuners share the same `next_max_jobs(current, saturation) -> int`
contract and the same decrease/hold/increase primitives, so they are bounded by construction.

## Why AIMD first

AIMD is deterministic, easy to reason about, and safe under noisy signals. It offers a strong baseline before introducing RL. This follows production lessons from SelfTune and adjacent schedulers: start with stable control primitives and auditability.

## Slurm tuning knob

This MVP models `target_max_jobs` as the primary actuator variable and applies updates via Slurm management commands. In dry-run mode, it logs planned actions and never mutates scheduler state.

## Tuner backends (AIMD / RL)

Two interchangeable tuners implement `next_max_jobs(current, saturation) -> int`:

- **AIMD** (`controller/tuner/aimd.py`, default): multiplicative-decrease (`current // 2`) when
  `saturation >= pressure_high`, additive-increase (`current + 1`) when `<= pressure_low`, hold
  otherwise; clamped to `[max_jobs_floor, max_jobs_ceil]`.
- **RL** (`controller/tuner/rl.py`): tabular Q-learning over a discretized
  `(pressure-bucket, max-jobs-bucket)` state with actions `{decrease, hold, increase}` (the same
  bounded primitives). The Q-table is loaded from `rl_qtable_path` (JSON). For unseen/untrained
  states it falls back to AIMD, so it is safe and bounded even with an empty table.

Selection: `TUNER_KIND=aimd|rl` (env) → `build_tuner(cfg)` in `controller/tuner/__init__.py`.
`controller/main.py` uses `build_tuner(cfg)`, so no loop changes are needed to switch.

Training (`controller/tuner/rl_env.py` + `controller/tuner/train.py`): a deterministic, seeded
queue simulator (`QueueSimulator`, stdlib only) generates load; `train_rl(cfg, episodes, seed)`
runs epsilon-greedy Q-learning and returns `(qtable, reward_history)`. Reward rewards utilization
and penalizes saturation/overcommit/thrash, so an optimal bounded policy is learnable.

```bash
python -m controller.tuner.train                          # writes models/qtable.json
TUNER_KIND=rl python -m controller.main --dry-run true     # run loop with the RL tuner
```

Reproducible training result (seed=42, 300 episodes): first-quarter mean reward ≈ 120 → last-quarter
≈ 164 (≈ +36%), confirming the agent learns over random/early behavior. RL stays in scope as a
tabular method — no neural nets / external RL dependencies.

## Issues found and fixed during end-to-end testing

1. **`/metrics` exposed no `adaptive_*` gauges** — `apps/api/main.py` never imported
   `controller/metrics.py`, so the gauges were never registered in the API process's Prometheus
   registry. Fixed by importing the module; regression test in `controller/tests/test_bugfixes.py`.
2. **Actuator `scontrol reconfigure` reverted the fallback change** — on clusters without slurmdbd
   accounting the actuator sets `scontrol update PartitionName=... MaxCPUsPerNode=N`, but the
   following `scontrol reconfigure` re-read `slurm.conf` and reverted it (confirmed live in the
   sandbox: `UNLIMITED → 4 → UNLIMITED`). Fixed by removing the `scontrol reconfigure` step; the
   `sacctmgr` QoS path and the `scontrol update` runtime change both apply without it (verified
   live: `UNLIMITED → 4` now persists). Regression test in `controller/tests/test_bugfixes.py`.
3. **`ControllerConfig` reads env at import time** (dataclass field defaults) — documented; env
   overrides must be set before first import. Config tests use a subprocess to validate overrides.

## Issues found and fixed during the pre-release hardening pass

These were surfaced by an independent audit before the public v1 release and fixed with regression
tests (suite grew to 182):

4. **Actuator advanced internal state even when the live Slurm command failed** — in live mode
   `SlurmActuator.apply()` updated `state.max_jobs` / `state.priority_weight_fs` (and `last_apply`)
   regardless of command success, so a failed `sacctmgr`/`scontrol` call left the controller's
   state diverging from the real cluster. Fixed so state and the cooldown clock only advance when
   the command succeeded (`ok=True`) or in dry-run (which still simulates the new state); a failed
   live command now logs a warning and returns `changed=False`. Tests in `test_actuator.py`.
5. **Initial `max_jobs` could exceed the ceiling when `max_jobs_ceil < 8`** — the actuator seeded
   `max_jobs = max(floor, 8)`, which violated the upper bound for small ceilings on the first
   iteration. Fixed by clamping the initial value to `[floor, ceil]`. Tests in `test_actuator.py`.
6. **AIMD "hold" path returned `current` unclamped** — an out-of-bounds `current` could propagate
   through the hold branch. Fixed by clamping the hold return to `[floor, ceil]` like the other
   branches. Tests in `test_aimd.py`.
7. **`SlurmCommandRunner.run(check=False)` reported false success** — with `check=False` a non-zero
   exit skipped the exception path and returned `(True, stdout)`. Fixed so `ok` reflects the real
   `returncode == 0`. Tests in `test_slurm_exec.py`.

See `docs/E2E_RESULTS.md` for the full live run against the Docker Slurm sandbox.

## API security posture (post-release hardening)

The FastAPI app (`apps/api/main.py`) has three env-configurable, fail-safe security controls.
Each keeps its prior open-by-default behavior when unset, so nothing breaks on upgrade, and each
parses its env var defensively (garbage/blank -> the safe default, never a crash or a silent
downgrade):

- **Optional bearer-token auth** — `CONTROLLER_API_TOKEN`. When set, `/cluster/snapshot` and
  `/metrics` require `Authorization: Bearer <token>` (constant-time compare via `secrets`);
  `/`, `/healthz`, `/ui` stay open. Unset = those two endpoints stay open.
- **Per-client rate limiting** — `CONTROLLER_API_RATE_LIMIT_PER_MIN` (default 120). Fixed-window
  per client IP across all routes, `429` + `Retry-After` past the limit; `<= 0` disables.
  In-memory / per-process (documented; not shared across workers).
- **CORS allowlist** — `CONTROLLER_API_ALLOWED_ORIGINS` (comma-separated; empty/unset = same-origin
  only, no middleware attached). Only listed origins are permitted; a disallowed preflight gets
  `400`. `allow_credentials` is hardcoded `False` so the CORS-spec-forbidden "wildcard origin +
  credentials" pairing is unreachable even with a `*` allowlist — safe because auth uses the
  `Authorization` header, not cookies. The allowlist is read once at app construction (origins are
  deployment config, not per-request state), which is why `configure_cors()` is a small seam the
  tests drive against a fresh app with the env set.

Error handling: `/cluster/snapshot` maps a collector failure (the collector raises on Slurm CLI
errors by design — fail-loud) to a structured `503 Service Unavailable` with a generic detail
string instead of an unhandled 500. The underlying exception text (command lines, hostnames,
stderr) is logged server-side only and never echoed to the client, so scheduler internals cannot
leak through the unauthenticated-by-default endpoint. `/healthz` stays `200` during a scheduler
outage: scheduler down ≠ API down.

Dependency hygiene has two layers: CI runs `pip-audit` (fails the build on known-vulnerable
installed versions — this is what caught the starlette PYSEC-2026-248/249 CVEs), and
`.github/dependabot.yml` opens weekly update PRs for both the `pip` and `github-actions`
ecosystems. `controller/tests/test_dependencies.py` pins the starlette CVE floor (`>=1.3.1`),
the pip-audit CI step, and the Dependabot config in place as regression guards.

All of the above is covered twice over: in-process TestClient units (`controller/tests/test_api.py`)
and true end-to-end tests (`controller/tests/test_e2e_api.py`) that boot `uvicorn` in a subprocess
with stub `sinfo`/`squeue`/`sacct` binaries (`SLURM_EXEC_MODE=local`) and drive real HTTP against
the import-time env wiring — CORS allowlist, bearer auth, rate limiting, the full
collector-to-JSON snapshot pipeline, and the outage-503 path.

## Next step for stronger production fidelity

- Add SlurmDBD-backed QOS accounting limits (`MaxJobsPU`, `GrpTRES`) and preemption policy updates
- Add job_submit.lua hooks for admission + quota
- Add per-account policy targets and tenant-aware snapshots
- Evaluate the RL tuner against AIMD via trace replay before enabling it in production
