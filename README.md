# AI Memory Gym — Cognitive Lab

A cognitive lab that benchmarks long-term memory strategies for LLM agents over simulated days and visualizes memory decay and cost tradeoffs.

## Requirements

- **Python 3.11+**
- No GPU required; CPU-only. Works without an LLM API key (MockLLM fallback).

## Setup

**1. Create and activate a virtual environment (recommended):**

```bash
cd ai-memory-gym
make venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
```

**2. Or install into your current environment:**

```bash
cd ai-memory-gym
pip install -r requirements.txt
```

## How to run

Activate the venv first if you used `make venv`:

```bash
source .venv/bin/activate
```

### 1. Mock LLM (no API key)

Runs out of the box; the agent uses ground-truth hints so benchmarks still make sense.

```bash
make install    # if you didn't use make venv
make test
make run_ui
```

Then open **http://localhost:8501**, leave **“Use Mock LLM (no API key)”** checked, and click **Run benchmark**.

### 2. Ollama (local or Ollama Cloud)

The app uses an **OpenAI-compatible** API, so you can point it at **Ollama** (local) or **Ollama Cloud**:

- **Local Ollama** (run `ollama serve` and pull a model first, e.g. `ollama pull llama3.2`):

```bash
export OPENAI_BASE_URL=http://localhost:11434/v1
export OPENAI_MODEL=llama3.2
# No API key needed for local Ollama
make run_ui
```

- **Ollama Cloud** (use your cloud base URL and key if required):

```bash
export OPENAI_BASE_URL=https://api.ollama.com/v1
export OPENAI_API_KEY=your_ollama_cloud_key
export OPENAI_MODEL=llama3.2
make run_ui
```

- **Local Ollama with cloud model (e.g. gpt-oss:120b-cloud)**  
  When you run `ollama run gpt-oss:120b-cloud` locally, Ollama connects to ollama.com for the model. Use the same model via the local Ollama API:

```bash
export OPENAI_BASE_URL=http://localhost:11434/v1
export OPENAI_MODEL=gpt-oss:120b-cloud
make run_ui
```

Or use the helper script (same defaults):

```bash
./scripts/run_ui_ollama.sh
```

In the Streamlit UI, **uncheck “Use Mock LLM (no API key)”** so the benchmark uses your LLM.

### 3. OpenAI (or other OpenAI-compatible APIs)

```bash
export OPENAI_API_KEY=sk-...
export OPENAI_MODEL=gpt-4o-mini
make run_ui
```

Uncheck **“Use Mock LLM”** in the UI.

## Environment variables

| Variable | Purpose |
|----------|---------|
| `OPENAI_BASE_URL` | API base URL (Ollama: `http://localhost:11434/v1`; Ollama Cloud: your cloud URL). |
| `OPENAI_API_KEY` | API key (optional for local Ollama; required for Ollama Cloud / OpenAI). |
| `OPENAI_MODEL` | Model name (e.g. `llama3.2`, `mistral`, `gpt-4o-mini`). |

If **OPENAI_BASE_URL** is set (e.g. for Ollama), the app uses that endpoint even without **OPENAI_API_KEY**.

## Run commands

```bash
make install      # pip install -r requirements.txt
make test        # pytest
make run_ui      # Streamlit → http://localhost:8501
make run_api     # FastAPI → http://localhost:8000
make run_suite   # Batch suite (policies × scenarios × seeds); outputs CSV + manifest
```

### Streamlit UI

```bash
cd ai-memory-gym
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

## AI Memory Gym V2 — Full-scale lab

- **Suite runner:** Batch runs over policies × scenarios × seeds; results cached by config hash (reruns instant).
- **Stress modes:** `distraction_flood`, `contradiction_injection`, `distribution_shift`, `memory_corruption`.
- **LLM modes:** **mock** (ground-truth hints), **rule** (deterministic errors: forget, interference, contradiction), **real** (OpenAI-compatible).
- **V2 metrics:** M-Score, retention half-life, interference rate, grounding/contradiction rate, cost per correct, PII leakage, right-to-be-forgotten.
- **UI tabs:** Overview, Forgetting Curve, Memory Timeline, Retrieval Explainability, Ablations, Export, **Leaderboard**, **Stress Lab**, **Frontier** (cost vs quality), **Retention**, **Interference Map**, **Trace Viewer**, **Report** (PDF export).

### Run suite (headless)

```bash
make run_suite
```

Or with custom config:

```bash
PYTHONPATH=. python -c "
from bench.suite_runner import run_suite
from bench.schemas import SuiteRunConfig
r = run_suite(SuiteRunConfig(
    policies=['full_log','vector_rag','hybrid_brain'],
    scenarios=['personal_assistant','research','ops','sales_crm'],
    seeds=[42,43,44],
    number_of_days=7,
    stress_modes=[None, 'distraction_flood'],
    llm_mode='rule',
), use_cache=True)
print('Aggregated:', r.aggregated_csv_path)
"
```

### Interpreting leaderboard and half-life

- **M-Score** = Accuracy − α(cost) − β(PII leakage) − γ(contradictions). Higher is better.
- **Retention half-life** = first day where recall drops below 50%. Higher means memory lasts longer.
- **Leaderboard** ranks policies by M-Score (filter by scenario/stress in the UI).

## Scenarios

- **personal_assistant** — Preferences (day 1) queried later; reminders and distractors.
- **research** — Facts across documents; contradictions and distractors.
- **ops** — Procedures and incident timeline; multi-step plan recall.
- **sales_crm** — Customer requirements across calls; recall constraints later.
- **legal_contract** — Clauses, versions, redlines.
- **meeting_memory** — Action items across meetings; “what did we decide?”.
- **multi_agent_handoff** — Agent A stores; Agent B retrieves.
- **adversarial_injection** — False fact injected later; test overwrite resistance.

## Memory policies

- **no_memory**, **full_log**, **rolling_summary**, **vector_rag**, **hybrid_brain**, **salience_only**, **rehearsal**
