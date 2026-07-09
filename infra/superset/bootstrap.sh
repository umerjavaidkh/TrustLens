#!/usr/bin/env bash
# One-shot Superset init: install ClickHouse driver, migrate, create admin, run.
set -e

pip install --no-cache-dir clickhouse-connect clickhouse-sqlalchemy >/dev/null 2>&1 || true

superset db upgrade
superset fab create-admin \
    --username "${SUPERSET_ADMIN_USER:-admin}" \
    --firstname Admin --lastname User \
    --email admin@trustlens.local \
    --password "${SUPERSET_ADMIN_PASSWORD:-admin}" || true
superset init

exec gunicorn --bind 0.0.0.0:8088 --workers 2 --timeout 120 "superset.app:create_app()"
