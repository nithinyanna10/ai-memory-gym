"""Benchmark runner: run scenario with policy, collect metrics, output JSON/DataFrame."""

import json
import os
import uuid
from pathlib import Path

from bench.schemas import BenchmarkConfig, BenchmarkResult, RunRecord
from bench.metrics import answer_correct, citation_precision, citation_recall, compute_metrics
from agent.runner import AgentRunner, StepInput
from agent.llm import get_llm
from sim.generators import generate_scenario_steps


def run_benchmark(config: BenchmarkConfig) -> BenchmarkResult:
    if getattr(config, "stress_mode", None) or getattr(config, "llm_mode", "mock") != "mock":
        from bench.suite_runner import run_single_with_stress
        return run_single_with_stress(config)
    scenario = generate_scenario_steps(config.scenario_type, config.number_of_days, config.seed)
    runner = AgentRunner(
        policy_name=config.policy,
        wm_size=config.wm_size,
        top_k=config.top_k,
        decay_lambda=config.decay_lambda,
        salience_threshold=config.salience_threshold,
        rehearsal_frequency=config.rehearsal_frequency,
    )
    hints = [gt.gold_answer for gt in scenario.ground_truth]
    llm_mode = getattr(config, "llm_mode", "mock" if config.use_mock_llm else "real")
    llm = get_llm(use_mock=config.use_mock_llm, ground_truth_hints=hints, llm_mode=llm_mode, seed=config.seed)
    runner.set_llm(llm)
    runner.reset_state()

    records: list[RunRecord] = []
    total_tokens = 0
    for step in scenario.steps:
        step_in = StepInput(
            day=step.day,
            turn=step.turn,
            event_text=step.event_text,
            question=step.question,
            gold_answer=step.gold_answer,
            gold_fact_ids=step.gold_fact_ids or [],
        )
        out = runner.run_step(step_in)
        total_tokens += len(step.event_text.split()) * 2 + len(out.answer.split()) * 2
        gold_ids = step.gold_fact_ids or []
        records.append(RunRecord(
            day=step.day, turn=step.turn, question=step.question, gold_answer=step.gold_answer,
            answer=out.answer, citations=out.citations, gold_fact_ids=gold_ids, retrieved=out.retrieved,
            correct=answer_correct(step.gold_answer, out.answer),
            citation_precision=citation_precision(out.citations, gold_ids),
            citation_recall=citation_recall(out.citations, gold_ids),
            latency_retrieve_s=out.latency_retrieve_s, latency_llm_s=out.latency_llm_s,
        ))

    metrics = compute_metrics(records)
    mem_count = 0
    if runner.state:
        mem_count = (
            len(runner.state.working.list_items())
            + len(runner.state.episodic.list_items())
            + len(runner.state.semantic.list_items())
            + len(runner.state.procedural.list_items())
        )

    result = BenchmarkResult(
        config=config,
        accuracy=metrics["accuracy"],
        citation_precision=metrics["citation_precision"],
        citation_recall=metrics["citation_recall"],
        hallucination_rate=metrics["hallucination_rate"],
        memory_items_stored=mem_count,
        token_estimate=total_tokens,
        retrieval_latency_avg_s=metrics["retrieval_latency_avg_s"],
        forgetting_curve=metrics["forgetting_curve"],
        run_records=records,
        run_id=str(uuid.uuid4())[:8],
    )
    try:
        from bench.metrics_v2 import compute_metrics_v2
        result.metrics_v2 = compute_metrics_v2(result)
    except Exception:
        pass
    return result


def save_result(result: BenchmarkResult, out_dir: str = "data/runs") -> str:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(out_dir, f"run_{result.run_id}.json")
    data = {
        "run_id": result.run_id,
        "config": {
            "scenario_type": result.config.scenario_type,
            "policy": result.config.policy,
            "seed": result.config.seed,
            "number_of_days": result.config.number_of_days,
            "wm_size": result.config.wm_size,
            "top_k": result.config.top_k,
            "decay_lambda": result.config.decay_lambda,
            "salience_threshold": result.config.salience_threshold,
            "rehearsal_frequency": result.config.rehearsal_frequency,
            "use_mock_llm": result.config.use_mock_llm,
        },
        "accuracy": result.accuracy,
        "citation_precision": result.citation_precision,
        "citation_recall": result.citation_recall,
        "hallucination_rate": result.hallucination_rate,
        "memory_items_stored": result.memory_items_stored,
        "token_estimate": result.token_estimate,
        "retrieval_latency_avg_s": result.retrieval_latency_avg_s,
        "forgetting_curve": result.forgetting_curve,
        "run_records": [
            {"day": r.day, "turn": r.turn, "question": r.question, "gold_answer": r.gold_answer,
             "answer": r.answer, "citations": r.citations, "gold_fact_ids": r.gold_fact_ids,
             "retrieved": r.retrieved, "correct": r.correct, "citation_precision": r.citation_precision, "citation_recall": r.citation_recall,
             "latency_retrieve_s": r.latency_retrieve_s, "latency_llm_s": r.latency_llm_s}
            for r in result.run_records
        ],
        "metrics_v2": result.metrics_v2,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


def load_result(path: str) -> BenchmarkResult:
    """Load a BenchmarkResult from JSON file."""
    import json
    with open(path) as f:
        data = json.load(f)
    config = BenchmarkConfig(
        scenario_type=data["config"]["scenario_type"],
        policy=data["config"]["policy"],
        seed=data["config"].get("seed"),
        number_of_days=data["config"].get("number_of_days", 7),
        wm_size=data["config"].get("wm_size", 10),
        top_k=data["config"].get("top_k", 5),
        decay_lambda=data["config"].get("decay_lambda", 0.1),
        salience_threshold=data["config"].get("salience_threshold", 0.3),
        rehearsal_frequency=data["config"].get("rehearsal_frequency", 3),
        use_mock_llm=data["config"].get("use_mock_llm", True),
    )
    records = []
    for r in data.get("run_records", []):
        raw_ret = r.get("retrieved", [])
        if raw_ret and isinstance(raw_ret[0], (list, tuple)):
            retrieved = [(x[0], x[1], float(x[2]) if len(x) > 2 else 0.0) for x in raw_ret]
        else:
            retrieved = []
        records.append(RunRecord(
            day=r["day"], turn=r["turn"], question=r.get("question"), gold_answer=r.get("gold_answer"),
            answer=r.get("answer", ""), citations=r.get("citations", []), gold_fact_ids=r.get("gold_fact_ids", []),
            retrieved=retrieved,
            correct=r.get("correct", False), citation_precision=r.get("citation_precision", 0), citation_recall=r.get("citation_recall", 0),
            latency_retrieve_s=r.get("latency_retrieve_s", 0), latency_llm_s=r.get("latency_llm_s", 0),
        ))
    return BenchmarkResult(
        config=config,
        accuracy=data["accuracy"],
        citation_precision=data["citation_precision"],
        citation_recall=data["citation_recall"],
        hallucination_rate=data["hallucination_rate"],
        memory_items_stored=data["memory_items_stored"],
        token_estimate=data["token_estimate"],
        retrieval_latency_avg_s=data["retrieval_latency_avg_s"],
        forgetting_curve=[(x[0], x[1]) for x in data["forgetting_curve"]],
        run_records=records,
        run_id=data.get("run_id"),
        metrics_v2=data.get("metrics_v2"),
    )


def result_to_dataframe(result: BenchmarkResult):
    import pandas as pd
    rows = []
    for r in result.run_records:
        rows.append({
            "run_id": result.run_id,
            "day": r.day,
            "turn": r.turn,
            "question": r.question,
            "gold_answer": r.gold_answer,
            "answer": r.answer,
            "correct": r.correct,
            "citation_precision": r.citation_precision,
            "citation_recall": r.citation_recall,
            "latency_retrieve_s": r.latency_retrieve_s,
        })
    return pd.DataFrame(rows)
