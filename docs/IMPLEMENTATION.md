# Implementation Notes

## Control loop

The controller runs a bounded loop:

1. Collect snapshot from Slurm (`sinfo`, `squeue`, optional `sacct`)
2. Build saturation score from queue pressure
3. Compute policy outputs (fairshare proxy weight + target max jobs)
4. Apply AIMD update to target max jobs
5. Actuate changes with cooldown and bounds
6. Emit metrics + structured logs

## Why AIMD first

AIMD is deterministic, easy to reason about, and safe under noisy signals. It offers a strong baseline before introducing RL. This follows production lessons from SelfTune and adjacent schedulers: start with stable control primitives and auditability.

## Slurm tuning knob

This MVP models `target_max_jobs` as the primary actuator variable and applies updates via Slurm management commands. In dry-run mode, it logs planned actions and never mutates scheduler state.

## Next step for stronger production fidelity

- Add SlurmDBD-backed QOS accounting limits (`MaxJobsPU`, `GrpTRES`) and preemption policy updates
- Add job_submit.lua hooks for admission + quota
- Add per-account policy targets and tenant-aware snapshots
