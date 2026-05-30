# Research Notes

## Sources used in this MVP design

- USENIX NSDI 2023 SelfTune paper and artifact direction
- Slurm configuration and runtime update docs (`slurm.conf`, `scontrol`, resource limits)
- Operational pattern: local sandbox first, then remote-machine rollout

## Chosen architecture decisions

- Local containerized Slurm for reproducible testing
- Safe control loop defaults to dry-run
- Observability-first with Prometheus + Grafana
- Explicit decomposition: collector/policy/tuner/actuator

## RL policy (implemented)

The RL tuner is tabular Q-learning, chosen to stay deterministic, dependency-free, and auditable
(consistent with the SelfTune philosophy: measure, adjust, observe, repeat):

- **State**: discretized `(pressure-bucket, max-jobs-bucket)`.
- **Actions**: `{decrease, hold, increase}` — the same bounded primitives as AIMD, so every action
  keeps `max_jobs` within `[floor, ceil]`.
- **Reward**: rewards utilization, penalizes saturation, overcommit, and action thrash — making an
  optimal bounded cap learnable.
- **Environment**: a seeded stdlib queue simulator (`QueueSimulator`) generating fluctuating load.
- **Fallback**: unseen states defer to AIMD, so the RL tuner is safe even before/without training.

Result (seed=42, 300 episodes): mean reward improves ≈ 36% from the first to the last quarter of
training. Evaluation against AIMD via trace replay (`controller/simulator/replay.py`) remains the
intended next comparison.

## Out of scope (MVP)

- Multi-cluster federation
- Deep RL with neural networks (the implemented RL policy is tabular Q-learning)
- LSF support (planned follow-on)
