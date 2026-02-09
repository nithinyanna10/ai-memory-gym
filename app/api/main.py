"""FastAPI: /run_benchmark, /run_step, /health."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from bench.schemas import BenchmarkConfig
from bench.runner import run_benchmark, save_result
from agent.runner import AgentRunner, StepInput
from agent.llm import get_llm


app = FastAPI(title="AI Memory Gym API", version="0.1.0")


class RunBenchmarkParams(BaseModel):
    scenario_type: str = "personal_assistant"
    policy: str = "full_log"
    seed: Optional[int] = 42
    number_of_days: int = 7
    wm_size: int = 10
    top_k: int = 5
    decay_lambda: float = 0.1
    salience_threshold: float = 0.3
    rehearsal_frequency: int = 3
    use_mock_llm: bool = True


class RunStepParams(BaseModel):
    day: int = 1
    turn: int = 1
    event_text: str = "User said hello."
    question: Optional[str] = None
    policy: str = "full_log"
    wm_size: int = 10
    top_k: int = 5


@app.get("/health")
def health():
    return {"status": "ok", "message": "AI Memory Gym API"}


@app.post("/run_benchmark")
def api_run_benchmark(params: RunBenchmarkParams):
    config = BenchmarkConfig(
        scenario_type=params.scenario_type,
        policy=params.policy,
        seed=params.seed,
        number_of_days=params.number_of_days,
        wm_size=params.wm_size,
        top_k=params.top_k,
        decay_lambda=params.decay_lambda,
        salience_threshold=params.salience_threshold,
        rehearsal_frequency=params.rehearsal_frequency,
        use_mock_llm=params.use_mock_llm,
    )
    try:
        result = run_benchmark(config)
        out_dir = str(ROOT / "data" / "runs")
        save_result(result, out_dir)
        return {
            "run_id": result.run_id,
            "accuracy": result.accuracy,
            "citation_precision": result.citation_precision,
            "citation_recall": result.citation_recall,
            "hallucination_rate": result.hallucination_rate,
            "memory_items_stored": result.memory_items_stored,
            "token_estimate": result.token_estimate,
            "retrieval_latency_avg_s": result.retrieval_latency_avg_s,
            "forgetting_curve": result.forgetting_curve,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/run_step")
def api_run_step(params: RunStepParams):
    step = StepInput(
        day=params.day,
        turn=params.turn,
        event_text=params.event_text,
        question=params.question,
    )
    runner = AgentRunner(policy_name=params.policy, wm_size=params.wm_size, top_k=params.top_k)
    llm = get_llm(use_mock=True)
    runner.set_llm(llm)
    runner.reset_state()
    out = runner.run_step(step)
    return {
        "answer": out.answer,
        "citations": out.citations,
        "retrieved": [{"id": x[0], "reason": x[1], "score": x[2]} for x in out.retrieved],
        "latency_retrieve_s": out.latency_retrieve_s,
        "latency_llm_s": out.latency_llm_s,
    }
