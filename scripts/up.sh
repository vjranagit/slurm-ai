#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="${SLURM_COMPOSE_FILE:-$ROOT/infra/docker/docker-compose.yml}"

if [ -f "$ROOT/.env" ]; then
  set -a
  source "$ROOT/.env"
  set +a
fi

docker compose -f "$COMPOSE_FILE" up -d --build

echo "Waiting for Slurm bootstrap..."
sleep 6
docker compose -f "$COMPOSE_FILE" exec -T slurm bash -lc "sinfo -R || true; sinfo"
