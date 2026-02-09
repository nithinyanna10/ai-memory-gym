#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="${PWD}:${PYTHONPATH}"
exec uvicorn app.api.main:app --host 0.0.0.0 --port 8000
