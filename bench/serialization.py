"""Shared serialization and validation helpers for benchmark results."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from bench.schemas import BenchmarkConfig, BenchmarkResult, RunRecord


def sanitize_stress_kwargs(stress_kwargs: Any) -> dict[str, Any]:
    """Normalize stress kwargs into a plain dict."""
    if isinstance(stress_kwargs, dict):
        return dict(stress_kwargs)
    return {}


def config_to_dict(config: BenchmarkConfig) -> dict[str, Any]:
    """Serialize BenchmarkConfig to a stable JSON-safe dict."""
    return {
        "scenario_type": config.scenario_type,
        "policy": config.policy,
        "seed": config.seed,
        "number_of_days": config.number_of_days,
        "wm_size": config.wm_size,
        "top_k": config.top_k,
        "decay_lambda": config.decay_lambda,
        "salience_threshold": config.salience_threshold,
        "rehearsal_frequency": config.rehearsal_frequency,
        "use_mock_llm": config.use_mock_llm,
        "llm_mode": config.llm_mode,
        "stress_mode": config.stress_mode,
        "stress_kwargs": sanitize_stress_kwargs(config.stress_kwargs),
    }


def run_record_to_dict(record: RunRecord) -> dict[str, Any]:
    """Serialize one run step record."""
    return {
        "day": record.day,
        "turn": record.turn,
        "question": record.question,
        "gold_answer": record.gold_answer,
        "answer": record.answer,
        "citations": list(record.citations),
        "gold_fact_ids": list(record.gold_fact_ids),
        "retrieved": list(record.retrieved),
        "correct": record.correct,
        "citation_precision": record.citation_precision,
        "citation_recall": record.citation_recall,
        "latency_retrieve_s": record.latency_retrieve_s,
        "latency_llm_s": record.latency_llm_s,
        "prompt_text": record.prompt_text,
        "memory_updates": record.memory_updates,
    }


def benchmark_result_to_dict(result: BenchmarkResult) -> dict[str, Any]:
    """Serialize a full benchmark result payload."""
    return {
        "run_id": result.run_id,
        "config": config_to_dict(result.config),
        "accuracy": result.accuracy,
        "citation_precision": result.citation_precision,
        "citation_recall": result.citation_recall,
        "hallucination_rate": result.hallucination_rate,
        "memory_items_stored": result.memory_items_stored,
        "token_estimate": result.token_estimate,
        "retrieval_latency_avg_s": result.retrieval_latency_avg_s,
        "forgetting_curve": list(result.forgetting_curve),
        "run_records": [run_record_to_dict(r) for r in result.run_records],
        "metrics_v2": result.metrics_v2,
    }


def dict_to_config(data: dict[str, Any]) -> BenchmarkConfig:
    """Load BenchmarkConfig from JSON payload with safe defaults."""
    return BenchmarkConfig(
        scenario_type=data.get("scenario_type", "personal_assistant"),
        policy=data.get("policy", "full_log"),
        seed=data.get("seed"),
        number_of_days=int(data.get("number_of_days", 7)),
        wm_size=int(data.get("wm_size", 10)),
        top_k=int(data.get("top_k", 5)),
        decay_lambda=float(data.get("decay_lambda", 0.1)),
        salience_threshold=float(data.get("salience_threshold", 0.3)),
        rehearsal_frequency=int(data.get("rehearsal_frequency", 3)),
        use_mock_llm=bool(data.get("use_mock_llm", True)),
        llm_mode=str(data.get("llm_mode", "mock")),
        stress_mode=data.get("stress_mode"),
        stress_kwargs=sanitize_stress_kwargs(data.get("stress_kwargs", {})),
    )


def normalize_retrieved(raw_retrieved: Any) -> list[tuple[str, str, float]]:
    """Normalize retrieved triplets from historical JSON variants."""
    if not isinstance(raw_retrieved, list):
        return []
    normalized: list[tuple[str, str, float]] = []
    for item in raw_retrieved:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            memory_id = str(item[0])
            reason = str(item[1])
            score = float(item[2]) if len(item) > 2 else 0.0
            normalized.append((memory_id, reason, score))
            continue
        if isinstance(item, dict):
            memory_id = str(item.get("id", ""))
            reason = str(item.get("reason", ""))
            score = float(item.get("score", 0.0))
            if memory_id:
                normalized.append((memory_id, reason, score))
    return normalized


def dict_to_run_record(data: dict[str, Any]) -> RunRecord:
    """Deserialize one run record safely."""
    return RunRecord(
        day=int(data.get("day", 0)),
        turn=int(data.get("turn", 0)),
        question=data.get("question"),
        gold_answer=data.get("gold_answer"),
        answer=str(data.get("answer", "")),
        citations=list(data.get("citations", [])),
        gold_fact_ids=list(data.get("gold_fact_ids", [])),
        retrieved=normalize_retrieved(data.get("retrieved", [])),
        correct=bool(data.get("correct", False)),
        citation_precision=float(data.get("citation_precision", 0.0)),
        citation_recall=float(data.get("citation_recall", 0.0)),
        latency_retrieve_s=float(data.get("latency_retrieve_s", 0.0)),
        latency_llm_s=float(data.get("latency_llm_s", 0.0)),
        prompt_text=data.get("prompt_text"),
        memory_updates=data.get("memory_updates"),
    )


def dict_to_benchmark_result(data: dict[str, Any]) -> BenchmarkResult:
    """Deserialize benchmark result from persisted JSON."""
    config = dict_to_config(data.get("config", {}))
    records = [dict_to_run_record(r) for r in data.get("run_records", [])]
    forgetting_curve = []
    for item in data.get("forgetting_curve", []):
        if isinstance(item, (list, tuple)) and len(item) == 2:
            forgetting_curve.append((int(item[0]), float(item[1])))

    return BenchmarkResult(
        config=config,
        accuracy=float(data.get("accuracy", 0.0)),
        citation_precision=float(data.get("citation_precision", 0.0)),
        citation_recall=float(data.get("citation_recall", 0.0)),
        hallucination_rate=float(data.get("hallucination_rate", 0.0)),
        memory_items_stored=int(data.get("memory_items_stored", 0)),
        token_estimate=int(data.get("token_estimate", 0)),
        retrieval_latency_avg_s=float(data.get("retrieval_latency_avg_s", 0.0)),
        forgetting_curve=forgetting_curve,
        run_records=records,
        run_id=data.get("run_id"),
        metrics_v2=data.get("metrics_v2"),
    )
