"""Suite runner: policies x scenarios x seeds x stress_modes with caching."""

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Optional

from bench.schemas import (
    BenchmarkConfig,
    BenchmarkResult,
    SuiteRunConfig,
    SuiteRunResult,
    RunRecord,
)
from bench.runner import run_benchmark, save_result
from bench.metrics_v2 import compute_metrics_v2
from bench.experiment_config import stable_config_hash
from bench.artifacts import write_run_artifacts
from sim.generators import generate_scenario_steps
from sim.stress import distraction_flood, contradiction_injection, distribution_shift, memory_corruption
from agent.runner import AgentRunner, StepInput
from agent.llm import get_llm
from bench.metrics import answer_correct, citation_precision, citation_recall
from bench.logging_utils import set_run_log_path, clear_run_log_path, run_log


def _config_hash(config: BenchmarkConfig) -> str:
    d = {
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
    return stable_config_hash(d)


def _apply_stress(scenario, config: BenchmarkConfig):
    from sim.base import Scenario
    s = scenario
    kw = config.stress_kwargs
    if config.stress_mode == "distraction_flood":
        k = kw.get("k_noise", 5)
        sim = kw.get("similarity_to_target", 0.0)
        s = distraction_flood(s, k_noise=k, seed=config.seed, similarity_to_target=sim)
    elif config.stress_mode == "contradiction_injection":
        p = kw.get("p_contradict", 0.5)
        targeted = kw.get("targeted", True)
        s = contradiction_injection(s, p_contradict=p, seed=config.seed, targeted=targeted)
    elif config.stress_mode == "distribution_shift":
        day = kw.get("style_switch_day", 2)
        shift_style = tuple(kw.get("shift_style", ("slack", "formal", "noisy")))
        s = distribution_shift(s, style_switch_day=day, shift_style=shift_style, seed=config.seed)
    return s


def run_single_with_stress(config: BenchmarkConfig, run_id: Optional[str] = None) -> BenchmarkResult:
    """Run one benchmark, applying stress to scenario; optional memory corruption during run."""
    if run_id is None:
        run_id = str(uuid.uuid4())[:8]
    run_log("run_start", run_id=run_id, scenario_type=config.scenario_type, policy=config.policy, seed=config.seed, stress_mode=getattr(config, "stress_mode", None))
    scenario = generate_scenario_steps(config.scenario_type, config.number_of_days, config.seed)
    scenario = _apply_stress(scenario, config)

    use_mock = config.llm_mode in ("mock", "rule")
    llm_mode = config.llm_mode
    hints = [gt.gold_answer for gt in scenario.ground_truth]
    llm = get_llm(use_mock=use_mock, ground_truth_hints=hints, llm_mode=llm_mode, seed=config.seed)

    runner = AgentRunner(
        policy_name=config.policy,
        wm_size=config.wm_size,
        top_k=config.top_k,
        decay_lambda=config.decay_lambda,
        salience_threshold=config.salience_threshold,
        rehearsal_frequency=config.rehearsal_frequency,
    )
    runner.set_llm(llm)
    runner.reset_state()

    p_drop = config.stress_kwargs.get("p_drop", 0.0)
    p_mutate = config.stress_kwargs.get("p_mutate", 0.0)
    do_corruption = config.stress_mode == "memory_corruption" and (p_drop > 0 or p_mutate > 0)

    records: list[RunRecord] = []
    total_tokens = 0
    for i, step in enumerate(scenario.steps):
        step_in = StepInput(
            day=step.day, turn=step.turn, event_text=step.event_text,
            question=step.question, gold_answer=step.gold_answer, gold_fact_ids=step.gold_fact_ids or [],
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
            prompt_text=getattr(out, "prompt_text", None),
            memory_updates=getattr(out, "memory_updates", None),
        ))
        if do_corruption and runner.state:
            mutate_strength = config.stress_kwargs.get("mutate_strength", 1.0)
            memory_corruption(runner.state, p_drop=p_drop, p_mutate=p_mutate, seed=(config.seed or 0) + i, mutate_strength=mutate_strength)
        run_log("step", run_id=run_id, step_index=i, day=step.day, turn=step.turn, correct=answer_correct(step.gold_answer, out.answer))

    from bench.metrics import compute_metrics
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
        run_id=run_id,
    )
    result.metrics_v2 = compute_metrics_v2(result)
    run_log("run_end", run_id=run_id, accuracy=result.accuracy, token_estimate=total_tokens, memory_items_stored=mem_count, m_score=result.metrics_v2.get("m_score") if result.metrics_v2 else None)
    return result


def run_suite(suite_config: SuiteRunConfig, out_dir: str = "data/runs", use_cache: bool = True) -> SuiteRunResult:
    """Run full suite; cache by config hash; output manifest + aggregated CSV/parquet."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    results: list[BenchmarkResult] = []
    for policy in suite_config.policies:
        for scenario_type in suite_config.scenarios:
            for seed in suite_config.seeds:
                for stress_mode in suite_config.stress_modes:
                    config = BenchmarkConfig(
                        scenario_type=scenario_type,
                        policy=policy,
                        seed=seed,
                        number_of_days=suite_config.number_of_days,
                        wm_size=suite_config.wm_size,
                        top_k=suite_config.top_k,
                        use_mock_llm=(suite_config.llm_mode != "real"),
                        llm_mode=suite_config.llm_mode,
                        stress_mode=stress_mode,
                        stress_kwargs=suite_config.stress_kwargs,
                    )
                    ch = _config_hash(config)
                    cache_path = Path(out_dir) / "cache" / f"{ch}.json"
                    run_dir = Path(out_dir) / "runs"
                    cached_run = False
                    if use_cache and cache_path.exists():
                        try:
                            with open(cache_path) as f:
                                cdata = json.load(f)
                            run_id = cdata.get("run_id")
                            run_file = Path(out_dir) / f"run_{run_id}.json"
                            if run_file.exists():
                                from bench.runner import load_result
                                result = load_result(str(run_file))
                                if not result.metrics_v2 and cdata.get("metrics_v2"):
                                    result.metrics_v2 = cdata["metrics_v2"]
                                run_dir_this = run_dir / run_id
                                if not (run_dir_this / "manifest.json").exists():
                                    write_run_artifacts(result, str(run_dir_this), cached=True)
                                results.append(result)
                                cached_run = True
                                continue
                        except Exception:
                            pass
                    if not cached_run:
                        run_id_new = str(uuid.uuid4())[:8]
                        run_dir_this = run_dir / run_id_new
                        run_dir_this.mkdir(parents=True, exist_ok=True)
                        set_run_log_path(str(run_dir_this / "run.log"))
                        try:
                            result = run_single_with_stress(config, run_id=run_id_new)
                            results.append(result)
                            save_result(result, out_dir)
                            write_run_artifacts(result, str(run_dir_this), cached=False)
                        finally:
                            clear_run_log_path()
                        if use_cache:
                            cache_path.parent.mkdir(parents=True, exist_ok=True)
                            with open(cache_path, "w") as f:
                                json.dump({
                                    "run_id": result.run_id,
                                    "config_hash": ch,
                                    "accuracy": result.accuracy,
                                    "m_score": result.metrics_v2.get("m_score") if result.metrics_v2 else None,
                                    "metrics_v2": result.metrics_v2,
                                }, f, indent=2)

    run_id = str(uuid.uuid4())[:8]
    config_hash = hashlib.sha256(json.dumps({
        "policies": suite_config.policies,
        "scenarios": suite_config.scenarios,
        "seeds": suite_config.seeds,
        "stress_modes": suite_config.stress_modes,
    }, sort_keys=True).encode()).hexdigest()[:16]

    # Aggregated CSV
    import pandas as pd
    rows = []
    for r in results:
        row = {
            "run_id": r.run_id,
            "scenario": r.config.scenario_type,
            "policy": r.config.policy,
            "seed": r.config.seed,
            "stress_mode": r.config.stress_mode or "",
            "accuracy": r.accuracy,
            "token_estimate": r.token_estimate,
            "memory_items_stored": r.memory_items_stored,
        }
        if r.metrics_v2:
            row["m_score"] = r.metrics_v2.get("m_score")
            row["retention_half_life"] = r.metrics_v2.get("retention_half_life")
            row["contradiction_rate"] = r.metrics_v2.get("contradiction_rate")
        rows.append(row)
    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, f"suite_{run_id}_aggregated.csv")
    df.to_csv(csv_path, index=False)
    try:
        df.to_parquet(csv_path.replace(".csv", ".parquet"), index=False)
    except Exception:
        pass

    manifest = {
        "suite_run_id": run_id,
        "config_hash": config_hash,
        "n_runs": len(results),
        "policies": suite_config.policies,
        "scenarios": suite_config.scenarios,
        "seeds": suite_config.seeds,
        "stress_modes": suite_config.stress_modes,
        "aggregated_csv": csv_path,
    }
    manifest_path = os.path.join(out_dir, f"suite_{run_id}_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return SuiteRunResult(
        run_id=run_id,
        config_hash=config_hash,
        results=results,
        aggregated_csv_path=csv_path,
        manifest_path=manifest_path,
    )


def load_suite_results(out_dir: str = "data/runs") -> list[dict]:
    """Load all suite manifests and aggregated CSVs from out_dir."""
    out = []
    p = Path(out_dir)
    if not p.exists():
        return out
    for f in p.glob("suite_*_manifest.json"):
        try:
            with open(f) as fp:
                out.append(json.load(fp))
        except Exception:
            pass
    return out
