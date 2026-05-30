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
