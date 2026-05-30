# adaptive-wlm-controller

`adaptive-wlm-controller` is a local-first, open-source MVP for adaptive workload management on Slurm.
It implements a quota-aware control loop inspired by SelfTune-style iterative tuning:

- collects scheduler + system pressure signals
- computes policy decisions from queue pressure and saturation
- applies safe Slurm tuning actions (AIMD-based, bounded)
- emits Prometheus metrics and structured logs
- provides a control-plane API and a lightweight dashboard
- includes a Slurm sandbox and workload replay tools

## MVP scope

This repository focuses on Slurm first. IBM LSF support is a planned follow-on branch.

## Architecture

- `controller/collectors`: gather pressure and utilization (`sinfo`, `squeue`, `sacct` optional)
- `controller/policy`: policy engine for quotas, fairshare, saturation rules
- `controller/tuner`: AIMD controller over actionable scheduler limits
- `controller/actuator`: safe application of updates (dry-run + bounded apply)
- `controller/simulator`: trace replay and policy evaluation utilities
- `controller/workload`: synthetic job submission and replay scripts
- `apps/api`: FastAPI control plane and status endpoints
- `apps/dashboard`: minimal web UI for utilization and policy signals
- `infra/docker/slurm`: single-container Slurm sandbox for local testing
- `infra/prometheus` + `infra/grafana`: observability stack

## Quickstart

### 1) Start infra

```bash
cp .env.example .env
./scripts/up.sh
```

For an existing Slurm host (for example `slurm.example.com`) instead of local Docker:

```bash
cp .env.example .env
sed -i 's/^SLURM_EXEC_MODE=.*/SLURM_EXEC_MODE=ssh/' .env
echo 'SLURM_SSH_HOST=slurm.example.com' >> .env
echo 'SLURM_SSH_USER=<your-user>' >> .env
# optional:
# echo 'SLURM_SSH_KEY_FILE=/path/to/id_rsa' >> .env
```

### 2) Start controller + API

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
uvicorn apps.api.main:app --host 0.0.0.0 --port 8080
```

### 3) Generate test jobs

```bash
./scripts/submit-test-jobs.sh 20
```

### 4) Run adaptive loop

```bash
python -m controller.main --interval 15 --dry-run false
```

### 5) Open observability

- API docs: http://localhost:8080/docs
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)
- Dashboard: `apps/dashboard/index.html` (served via API static endpoint)

## Safety model

- hard bounds on all tuned values
- cooldowns between actions
- dry-run mode by default
- no direct destructive scheduler operations

## Verification checklist

- Slurm sandbox starts and `scontrol ping` returns `UP`
- test jobs are accepted by `sbatch` and visible in `squeue`
- controller emits `adaptive_*` Prometheus metrics
- actuator logs bounded changes and respects cooldown

## Container runtime note

The included single-container Slurm sandbox is optimized for control-loop and API validation.
In Docker-only environments without full cgroup/systemd integration, `slurmd` may not launch job steps,
so submitted jobs can remain `PD` (pending). This does not affect collector/policy/tuner/actuator testing.

For end-to-end job execution (`RUNNING` -> `COMPLETED`), run Slurm against a host/VM setup with working
cgroup support (or point the controller at an existing Slurm cluster such as `slurm.example.com`).

## Publishing notes

Before making the repo public:

- review `.env` for sensitive values (this repo ships only placeholders)
- ensure no internal hostnames, tokens, or private traces are committed
- run `git grep -nE '(token|secret|password|key=)'` and inspect matches
