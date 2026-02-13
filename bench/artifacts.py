"""Write per-run artifacts: manifest.json, traces.jsonl, metrics.json, summary.csv/parquet."""

import json
import os
from pathlib import Path
from typing import Optional

from bench.schemas import BenchmarkResult, RunRecord
from bench.experiment_config import RunManifest, stable_config_hash


def write_run_artifacts(result: BenchmarkResult, run_dir: str, cached: bool = False) -> dict:
    """Write manifest.json, traces.jsonl, metrics.json, summary.csv, summary.parquet under run_dir. Returns artifact paths."""
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    run_id = result.run_id or "unknown"
    config = result.config

    config_dict = {
        "scenario_type": config.scenario_type,
        "policy": config.policy,
        "seed": config.seed,
        "number_of_days": config.number_of_days,
        "wm_size": config.wm_size,
        "top_k": config.top_k,
        "llm_mode": getattr(config, "llm_mode", "mock"),
        "stress_mode": getattr(config, "stress_mode", None),
        "stress_kwargs": getattr(config, "stress_kwargs", {}),
    }
    config_hash = stable_config_hash(config_dict)

    # manifest.json
    manifest = RunManifest(
        run_id=run_id,
        config_hash=config_hash,
        config=config_dict,
        scenario_type=config.scenario_type,
        policy=config.policy,
        seed=config.seed,
        stress_mode=getattr(config, "stress_mode", None),
        accuracy=result.accuracy,
        m_score=(result.metrics_v2 or {}).get("m_score"),
        token_estimate=result.token_estimate,
        memory_items_stored=result.memory_items_stored,
        cached=cached,
        artifacts={},
    )
    manifest_path = os.path.join(run_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        f.write(manifest.model_dump_json(indent=2))
    manifest.artifacts["manifest"] = manifest_path

    # traces.jsonl (one line per step)
    traces_path = os.path.join(run_dir, "traces.jsonl")
    with open(traces_path, "w") as f:
        for idx, r in enumerate(result.run_records):
            trace = {
                "step_index": idx,
                "day": r.day,
                "turn": r.turn,
                "question": r.question,
                "gold_answer": r.gold_answer,
                "answer": r.answer,
                "correct": r.correct,
                "retrieved": [{"id": x[0], "reason": x[1], "score": x[2]} for x in r.retrieved],
                "citations": r.citations,
                "latency_retrieve_s": r.latency_retrieve_s,
                "latency_llm_s": r.latency_llm_s,
            }
            if getattr(r, "prompt_text", None):
                trace["prompt_text"] = r.prompt_text[:2000]
            if getattr(r, "memory_updates", None):
                trace["memory_updates"] = r.memory_updates
            f.write(json.dumps(trace, default=str) + "\n")
    manifest.artifacts["traces"] = traces_path

    # metrics.json
    metrics_path = os.path.join(run_dir, "metrics.json")
    metrics = {
        "accuracy": result.accuracy,
        "citation_precision": result.citation_precision,
        "citation_recall": result.citation_recall,
        "hallucination_rate": result.hallucination_rate,
        "token_estimate": result.token_estimate,
        "memory_items_stored": result.memory_items_stored,
        "retrieval_latency_avg_s": result.retrieval_latency_avg_s,
        "forgetting_curve": result.forgetting_curve,
    }
    if result.metrics_v2:
        metrics["v2"] = result.metrics_v2
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    manifest.artifacts["metrics"] = metrics_path

    # summary.csv and summary.parquet
    import pandas as pd
    rows = []
    for r in result.run_records:
        rows.append({
            "run_id": run_id,
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
    df = pd.DataFrame(rows)
    summary_csv = os.path.join(run_dir, "summary.csv")
    df.to_csv(summary_csv, index=False)
    manifest.artifacts["summary_csv"] = summary_csv
    try:
        summary_parquet = os.path.join(run_dir, "summary.parquet")
        df.to_parquet(summary_parquet, index=False)
        manifest.artifacts["summary_parquet"] = summary_parquet
    except Exception:
        pass
    run_log_path = os.path.join(run_dir, "run.log")
    if os.path.exists(run_log_path):
        manifest.artifacts["run_log"] = run_log_path

    # Update manifest with artifact paths and write again
    with open(manifest_path, "w") as f:
        f.write(manifest.model_dump_json(indent=2))

    return dict(manifest.artifacts)
