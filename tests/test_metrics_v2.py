"""Tests for V2 metrics and M-score."""

import pytest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench.schemas import RunRecord, BenchmarkResult, BenchmarkConfig
from bench.metrics_v2 import (
    retention_half_life,
    interference_rate,
    grounding_rate,
    contradiction_rate,
    cost_per_correct,
    m_score,
    compute_metrics_v2,
    right_to_be_forgotten_score,
)


def test_retention_half_life():
    assert retention_half_life([(1, 0.9), (2, 0.6), (3, 0.4)]) == 3
    assert retention_half_life([(1, 1.0)]) is None


def test_interference_rate():
    records = [
        RunRecord(1, 1, "Q?", "A", "wrong", [], [], [(1, "r", 0.5)], False, 0.0, 0.0, 0.01, 0.1),
        RunRecord(2, 2, "Q2?", "B", "B", [], [], [], True, 1.0, 1.0, 0.01, 0.1),
    ]
    assert interference_rate(records) == 0.5


def test_grounding_rate():
    records = [
        RunRecord(1, 1, "Q?", "A", "A", [], [], [], True, 1.0, 1.0, 0.01, 0.1),
        RunRecord(2, 2, "Q2?", "B", "wrong", [], [], [], False, 0.0, 0.0, 0.01, 0.1),
    ]
    assert grounding_rate(records) == 0.5


def test_m_score():
    m = m_score(accuracy=0.8, token_estimate=500, pii_leakage_rate_val=0.0, contradiction_rate_val=0.1)
    assert m < 0.8
    assert m > 0.5


def test_compute_metrics_v2():
    config = BenchmarkConfig()
    records = [
        RunRecord(1, 1, "Q?", "A", "A", [], [], [], True, 1.0, 1.0, 0.01, 0.1),
    ]
    result = BenchmarkResult(
        config=config,
        accuracy=1.0,
        citation_precision=1.0,
        citation_recall=1.0,
        hallucination_rate=0.0,
        memory_items_stored=5,
        token_estimate=100,
        retrieval_latency_avg_s=0.01,
        forgetting_curve=[(1, 1.0)],
        run_records=records,
    )
    m = compute_metrics_v2(result)
    assert "m_score" in m
    assert "retention_half_life" in m
    assert "interference_rate" in m


def test_right_to_be_forgotten_score():
    records = [
        RunRecord(1, 1, "Q?", "A", "A", ["ep_2"], [], [], True, 1.0, 1.0, 0.01, 0.1),
    ]
    # deleted ep_1; we cited ep_2 so no bad citation
    assert right_to_be_forgotten_score(records, {"ep_1"}) == 1.0
    assert right_to_be_forgotten_score(records, {"ep_2"}) == 0.0
