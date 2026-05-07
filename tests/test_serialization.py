"""Tests for benchmark serialization and config validation."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.schemas import BenchmarkConfig, BenchmarkResult, RunRecord  # noqa: E402
from bench.serialization import (  # noqa: E402
    benchmark_result_to_dict,
    dict_to_benchmark_result,
    normalize_retrieved,
)


def _sample_result() -> BenchmarkResult:
    config = BenchmarkConfig(
        scenario_type="personal_assistant",
        policy="full_log",
        seed=42,
        number_of_days=3,
        wm_size=10,
        top_k=5,
        llm_mode="mock",
    )
    record = RunRecord(
        day=1,
        turn=1,
        question="What drink does user prefer?",
        gold_answer="oat milk",
        answer="oat milk",
        citations=["ep_1"],
        gold_fact_ids=["ep_1"],
        retrieved=[("ep_1", "episodic_decay_salience", 0.88)],
        correct=True,
        citation_precision=1.0,
        citation_recall=1.0,
        latency_retrieve_s=0.01,
        latency_llm_s=0.09,
        prompt_text="prompt",
        memory_updates=[{"id": "ep_1"}],
    )
    return BenchmarkResult(
        config=config,
        accuracy=1.0,
        citation_precision=1.0,
        citation_recall=1.0,
        hallucination_rate=0.0,
        memory_items_stored=5,
        token_estimate=123,
        retrieval_latency_avg_s=0.01,
        forgetting_curve=[(1, 1.0)],
        run_records=[record],
        run_id="abc123",
        metrics_v2={"m_score": 0.9},
    )


def test_result_roundtrip_serialization():
    result = _sample_result()
    payload = benchmark_result_to_dict(result)
    restored = dict_to_benchmark_result(payload)
    assert restored.run_id == result.run_id
    assert restored.config.policy == result.config.policy
    assert restored.run_records[0].retrieved[0][0] == "ep_1"
    assert restored.metrics_v2 == {"m_score": 0.9}


def test_normalize_retrieved_supports_dict_shape():
    raw = [{"id": "ep_9", "reason": "semantic_primary", "score": 0.75}]
    normalized = normalize_retrieved(raw)
    assert normalized == [("ep_9", "semantic_primary", 0.75)]


def test_benchmark_config_validation_rejects_bad_values():
    with pytest.raises(ValueError):
        BenchmarkConfig(number_of_days=0)
    with pytest.raises(ValueError):
        BenchmarkConfig(top_k=0)
    with pytest.raises(ValueError):
        BenchmarkConfig(salience_threshold=1.2)
    with pytest.raises(ValueError):
        BenchmarkConfig(llm_mode="invalid")
