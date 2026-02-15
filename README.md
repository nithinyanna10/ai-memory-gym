# AI Memory Gym V2 — Cognitive Lab

A production-grade cognitive lab that benchmarks long-term memory strategies for LLM agents over simulated days. Supports single runs and suite runs (policies × scenarios × seeds × stress modes), config-hash caching, and rich artifacts (manifest, traces, summary CSV/parquet, PDF reports).

## Quickstart

```bash
cd ai-memory-gym
make install
make test
make run_ui
```

Open **http://localhost:8501**. Use the **Experiment Builder** sidebar to pick Scenario, Policy, and (optionally) Stress mode, then click **Run single**. No API key required (Mock or Rule LLM).

- **Run suite:** Click **Run suite** for a batch over 2 policies × 4 scenarios × 2 seeds; results are cached so reruns with the same config are instant.
- **Views:** Dashboard (metric cards, Pareto, forgetting curve, errors), Leaderboard, Stress Lab, Traces, Memory MRI, Reports (PDF), Legacy tabs.

## Requirements

- **Python 3.11+**
- No GPU required; runs without an LLM API key (Mock/Rule LLM).

## How caching works

- Every run is identified by a **stable config hash** (scenario, policy, seed, days, stress mode, stress kwargs, llm_mode, wm_size, top_k). Same config + seed ⇒ same hash.
- **Suite runner:** Before running a (policy, scenario, seed, stress) combination, the runner looks up `data/runs/cache/<hash>.json`. If it exists, it loads the existing run from `data/runs/run_<run_id>.json` and reuses it; it also ensures **per-run artifacts** under `data/runs/runs/<run_id>/` (manifest, traces, metrics, summary CSV/parquet, run log). So reruns with the same config produce no new computation and instant results.
- **Single run:** Saving a result writes both the flat `run_<run_id>.json` and the directory `data/runs/runs/<run_id>/` with the same artifacts.

## How to interpret metrics

| Metric | Meaning |
|--------|--------|
| **Accuracy** | Fraction of evaluation questions answered correctly. |
| **M-Score** | Composite: `accuracy − 0.002×token_est − 2×pii_leakage_rate − 1×contradiction_rate`. Higher is better; balances quality, cost, and safety. |
| **Retention half-life** | First day where accuracy (by day) drops below 50%. Higher = memory lasts longer. |
| **Interference rate** | Rate of retrievals that pull a wrong-but-similar memory. |
| **Contradiction rate** | Rate of answers that follow an injected contradiction instead of the original fact. |
| **PII leakage rate** | Fraction of answers that expose PII/secret-tagged content. |
| **Cost per correct** | Token estimate divided by number of correct answers. |
| **Citation precision/recall** | How well cited memories match gold fact IDs. |

Leaderboard ranks **policy × stress** combinations by M-Score; filter by scenario pack and stress mode in the UI.

## Demo script for judges

1. **Install and test**
   ```bash
   make install && make test
   ```

2. **Single run (no API key)**
   - `make run_ui` → open http://localhost:8501
   - Sidebar: Scenario = **personal_assistant**, Policy = **full_log**, Seed = **42**, LLM = **mock**
   - Click **Run single**
   - Open **Dashboard**: see Accuracy, Pareto (current run highlighted), Forgetting curve, Error taxonomy

3. **Traces**
   - View = **Traces** → select run → move step slider
   - Inspect Prompt, Retrieved memories (scores/reasons), Memory updates, Answer, Gold, Correctness; highest-influence memory is highlighted

4. **Suite and leaderboard**
   - Click **Run suite** (2 policies × 4 scenarios × 2 seeds)
   - View = **Leaderboard** → filter by scenario/stress, see M-Score ranking and distribution (box plot)

5. **Stress Lab**
   - Sidebar: Stress mode = **distraction_flood**, k_noise = 5
   - Run single again
   - View = **Stress Lab**: before/after comparison and scenario×policy heatmap

6. **Report**
   - View = **Reports** → **Generate PDF report** → Download

7. **Headless suite**
   ```bash
   make run_suite
   ```
   Outputs: `data/runs/suite_<id>_manifest.json`, `data/runs/suite_<id>_aggregated.csv`, and per-run dirs under `data/runs/runs/<run_id>/` (manifest.json, traces.jsonl, metrics.json, summary.csv, summary.parquet, run.log).

## Run commands

```bash
make install     # pip install -r requirements.txt
make lint        # ruff check .
make test        # pytest
make run_ui      # Streamlit → http://localhost:8501
make run_api     # FastAPI → http://localhost:8000
make run_suite   # Example suite (full_log, vector_rag × personal_assistant, ops × seeds 42,43)
```

## Environment variables

| Variable | Purpose |
|----------|---------|
| `OPENAI_BASE_URL` | API base URL (e.g. Ollama: `http://localhost:11434/v1`). |
| `OPENAI_API_KEY` | API key (optional for local Ollama; required for OpenAI/Ollama Cloud). |
| `OPENAI_MODEL` | Model name (e.g. `llama3.2`, `gpt-4o-mini`). |

## Project structure

```
ai-memory-gym/
  app/ui/streamlit_app.py   # Streamlit UI (Dashboard, Leaderboard, Stress Lab, Traces, Memory MRI, Reports, Legacy)
  app/ui/styles.css         # Theme / cards / section headers
  app/api/main.py           # FastAPI
  bench/                    # Experiment config, runner, suite, artifacts, metrics, logging
  memory/                   # Memory systems
  agent/                    # LLM (mock, rule, real) + runner
  sim/                      # Scenarios + stress modes
  data/runs/                # run_<id>.json, runs/<run_id>/*, cache/, suite_*_*
  tests/
  .github/workflows/ci.yml   # install, ruff, pytest
```

## Scenarios (10+ packs)

- **personal_assistant** / **personal_prefs** — Preferences and constraints; distractors.
- **research** — Documents and revisions; contradictions.
- **ops** — Procedures and incident timeline.
- **sales_crm** — Customer requirements; constraints.
- **legal_contract** — Clauses, versions, redlines.
- **meeting_memory** — Action items; “what did we decide?”.
- **multi_agent_handoff** — Agent A stores; Agent B retrieves.
- **adversarial_injection** — False fact later; overwrite resistance.
- **research_long** — Long-horizon research; query late.
- **ops_noisy_slack** — Noisy Slack + runbook; extract signal.
- **safety_pii** — PII/secret tags; leakage and right-to-be-forgotten.
- **tool_use_procedure** — Tool-call procedures; recall which tool.

## Memory policies

- **no_memory**, **full_log**, **rolling_summary**, **vector_rag**, **hybrid_brain**, **salience_only**, **rehearsal**

## Stress modes

- **distraction_flood** — Add k_noise irrelevant steps; optional similarity_to_target for confusable noise.
- **memory_corruption** — In-run drop/mutate episodic items (p_drop, p_mutate, mutate_strength).
- **contradiction_injection** — Inject contradicting step (p_contradict, targeted=True).
- **distribution_shift** — From style_switch_day onward, apply shift_style (slack/formal/noisy).

All stress modes are deterministic given seed.
