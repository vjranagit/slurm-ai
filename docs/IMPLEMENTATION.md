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

See `docs/E2E_RESULTS.md` for the full live run against the Docker Slurm sandbox.

## Next step for stronger production fidelity

- Add SlurmDBD-backed QOS accounting limits (`MaxJobsPU`, `GrpTRES`) and preemption policy updates
- Add job_submit.lua hooks for admission + quota
- Add per-account policy targets and tenant-aware snapshots
- Evaluate the RL tuner against AIMD via trace replay before enabling it in production
