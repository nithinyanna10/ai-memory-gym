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
from bench.serialization import benchmark_result_to_dict, dict_to_benchmark_result
from bench.logging_utils import run_log


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
        records.append(
            RunRecord(
                day=step.day,
                turn=step.turn,
                question=step.question,
                gold_answer=step.gold_answer,
                answer=out.answer,
                citations=out.citations,
                gold_fact_ids=gold_ids,
                retrieved=out.retrieved,
                correct=answer_correct(step.gold_answer, out.answer),
                citation_precision=citation_precision(out.citations, gold_ids),
                citation_recall=citation_recall(out.citations, gold_ids),
                latency_retrieve_s=out.latency_retrieve_s,
                latency_llm_s=out.latency_llm_s,
                prompt_text=getattr(out, "prompt_text", None),
                memory_updates=getattr(out, "memory_updates", None),
            )
        )

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
    except Exception as exc:
        run_log("metrics_v2_compute_failed", level="warning", error=str(exc), run_id=result.run_id)
    return result


def save_result(result: BenchmarkResult, out_dir: str = "data/runs", write_artifacts: bool = True) -> str:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(out_dir, f"run_{result.run_id}.json")
    run_dir = os.path.join(out_dir, "runs", result.run_id or "unknown")
    if write_artifacts:
        try:
            from bench.artifacts import write_run_artifacts
            write_run_artifacts(result, run_dir, cached=False)
        except Exception as exc:
            run_log("write_run_artifacts_failed", level="warning", run_id=result.run_id, error=str(exc), run_dir=run_dir)
    data = benchmark_result_to_dict(result)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return path


def load_result(path: str) -> BenchmarkResult:
    """Load a BenchmarkResult from JSON file."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return dict_to_benchmark_result(data)


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
