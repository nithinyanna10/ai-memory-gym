"""Unit tests for benchmark scoring and runner."""

import pytest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench.metrics import answer_correct, citation_precision, citation_recall, compute_metrics, normalize_answer
from bench.schemas import RunRecord, BenchmarkConfig, BenchmarkResult
from bench.runner import run_benchmark


def test_normalize_answer():
    assert normalize_answer("  Hello,  world.  ") == "hello world"
    assert normalize_answer("") == ""


def test_answer_correct():
    assert answer_correct("oat milk", "The user prefers oat milk.") is True
    assert answer_correct("PST", "PST timezone") is True
    assert answer_correct("wrong", "right") is False


def test_citation_precision():
    assert citation_precision(["ep_1"], ["ep_1"]) == 1.0
    assert citation_precision(["ep_1", "ep_2"], ["ep_1"]) == 0.5
    assert citation_precision([], ["ep_1"]) == 0.0
    assert citation_precision([], []) == 1.0


def test_citation_recall():
    assert citation_recall(["ep_1"], ["ep_1"]) == 1.0
    assert citation_recall(["ep_1"], ["ep_1", "ep_2"]) == 0.5
    assert citation_recall([], []) == 1.0


def test_compute_metrics():
    records = [
        RunRecord(day=1, turn=1, question="Q?", gold_answer="A", answer="A", citations=["ep_1"], gold_fact_ids=["ep_1"],
                  retrieved=[], correct=True, citation_precision=1.0, citation_recall=1.0, latency_retrieve_s=0.01, latency_llm_s=0.1),
        RunRecord(day=2, turn=2, question="Q2?", gold_answer="B", answer="wrong", citations=[], gold_fact_ids=[],
                  retrieved=[], correct=False, citation_precision=1.0, citation_recall=1.0, latency_retrieve_s=0.01, latency_llm_s=0.1),
    ]
    m = compute_metrics(records)
    assert m["accuracy"] == 0.5
    assert m["citation_precision"] == 1.0
    assert "forgetting_curve" in m
    assert len(m["forgetting_curve"]) >= 1


def test_run_benchmark_smoke():
    config = BenchmarkConfig(scenario_type="personal_assistant", policy="full_log", seed=42, number_of_days=5, use_mock_llm=True)
    result = run_benchmark(config)
    assert isinstance(result, BenchmarkResult)
    assert 0 <= result.accuracy <= 1
    assert len(result.run_records) >= 1
    assert result.config.policy == "full_log"
