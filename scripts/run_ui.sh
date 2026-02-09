#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="${PWD}:${PYTHONPATH}"
exec streamlit run app/ui/streamlit_app.py --server.port 8501
