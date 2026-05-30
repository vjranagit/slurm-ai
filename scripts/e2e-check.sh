#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="${SLURM_COMPOSE_FILE:-$ROOT/infra/docker/docker-compose.yml}"

./scripts/up.sh
./scripts/submit-test-jobs.sh 5 2
sleep 4

docker compose -f "$COMPOSE_FILE" exec -T slurm bash -lc "squeue || true"
docker compose -f "$COMPOSE_FILE" exec -T slurm bash -lc "sacct -S now-10minutes -X -n -o JobID,State || true"

echo "e2e-check completed"
