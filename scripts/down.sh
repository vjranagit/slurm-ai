#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="${SLURM_COMPOSE_FILE:-$ROOT/infra/docker/docker-compose.yml}"

docker compose -f "$COMPOSE_FILE" down -v
