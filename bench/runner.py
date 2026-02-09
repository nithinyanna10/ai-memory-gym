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
    llm = get_llm(use_mock=config.use_mock_llm, ground_truth_hints=hints)
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

    return BenchmarkResult(
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
             "correct": r.correct, "citation_precision": r.citation_precision, "citation_recall": r.citation_recall,
             "latency_retrieve_s": r.latency_retrieve_s, "latency_llm_s": r.latency_llm_s}
            for r in result.run_records
        ],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


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
