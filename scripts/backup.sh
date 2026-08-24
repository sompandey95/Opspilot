#!/usr/bin/env bash
# Dumps Postgres and snapshots the ChromaDB volume into ./backups/<timestamp>/.
# Run from the repo root, on the host running `docker compose`. Intended to be
# called from cron (see DEPLOY.md for a sample crontab line).
set -euo pipefail

cd "$(dirname "$0")/.."

TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="backups/${TIMESTAMP}"
mkdir -p "${OUT_DIR}"

echo "Backing up Postgres..."
docker compose exec -T postgres pg_dump -U opspilot -d opspilot \
    | gzip > "${OUT_DIR}/postgres.sql.gz"

echo "Backing up ChromaDB volume..."
docker compose exec -T chromadb tar czf - -C /chroma/chroma . \
    > "${OUT_DIR}/chromadata.tar.gz"

echo "Backup written to ${OUT_DIR}/"
echo "Keeping the last 14 backups; deleting older ones..."
ls -1dt backups/*/ 2>/dev/null | tail -n +15 | xargs -r rm -rf
