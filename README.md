# AI Memory Gym — Cognitive Lab

A cognitive lab that benchmarks long-term memory strategies for LLM agents over simulated days and visualizes memory decay and cost tradeoffs.

## Requirements

- **Python 3.11+**
- No GPU required; CPU-only. Works without an LLM API key (MockLLM fallback).

## Setup

```bash
cd ai-memory-gym
pip install -r requirements.txt
```

## Environment variables (optional)

- **`OPENAI_API_KEY`** — If set, the benchmark uses an OpenAI-compatible API instead of MockLLM.
- **`OPENAI_BASE_URL`** — Optional; override API base URL.
- **`OPENAI_MODEL`** — Optional; model name (default: gpt-4o-mini).

Without `OPENAI_API_KEY`, the app uses **MockLLM** with ground-truth hints so the demo is still meaningful.

## Run locally

```bash
make install
make test
make run_ui    # Streamlit on http://localhost:8501
make run_api   # FastAPI on http://localhost:8000
```

### Streamlit UI

```bash
PYTHONPATH=. streamlit run app/ui/streamlit_app.py --server.port 8501
```

### FastAPI

```bash
PYTHONPATH=. uvicorn app.api.main:app --host 0.0.0.0 --port 8000
```

- **GET /health** — Health check.
- **POST /run_benchmark** — Run full benchmark; returns accuracy, citation metrics, forgetting_curve.
- **POST /run_step** — Simulate one step; returns answer, citations, retrieved memories.

## Project structure

```
ai-memory-gym/
  app/ui/streamlit_app.py   # Dashboard
  app/api/main.py           # FastAPI
  memory/                   # Memory systems
  agent/                    # LLM + runner
  sim/                      # Scenarios + generators
  bench/                    # Benchmark runner + metrics
  data/runs/                # Output JSON/CSV
  tests/
  scripts/
```

## Scenarios

- **personal_assistant** — Preferences (day 1) queried later; reminders and distractors.
- **research** — Facts across documents; contradictions and distractors.
- **ops** — Procedures and incident timeline; multi-step plan recall.

## Memory policies

- **no_memory**, **full_log**, **rolling_summary**, **vector_rag**, **hybrid_brain**, **salience_only**, **rehearsal**
