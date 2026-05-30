#!/usr/bin/env bash
set -euo pipefail

COUNT="${1:-10}"
SECONDS="${2:-5}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python3 -m controller.workload.submitter \
  --compose-file "${SLURM_COMPOSE_FILE:-$ROOT/infra/docker/docker-compose.yml}" \
  --service "${SLURM_SERVICE:-slurm}" \
  --exec-mode "${SLURM_EXEC_MODE:-docker}" \
  --ssh-host "${SLURM_SSH_HOST:-}" \
  --ssh-user "${SLURM_SSH_USER:-}" \
  --ssh-key-file "${SLURM_SSH_KEY_FILE:-}" \
  --count "$COUNT" \
  --seconds "$SECONDS"
