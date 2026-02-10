#!/usr/bin/env bash
# Run Streamlit UI using local Ollama with gpt-oss:120b-cloud (ollama.com).
# Ensure Ollama is running (e.g. you can run: ollama run gpt-oss:120b-cloud).
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="${PWD}:${PYTHONPATH}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://localhost:11434/v1}"
export OPENAI_MODEL="${OPENAI_MODEL:-gpt-oss:120b-cloud}"
# No API key needed when using local Ollama (it proxies cloud model)
exec streamlit run app/ui/streamlit_app.py --server.port 8501
