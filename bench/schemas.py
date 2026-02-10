"""Schemas for benchmark config and results."""

from dataclasses import dataclass, field
from typing import Optional, Any


@dataclass
class BenchmarkConfig:
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
    llm_mode: str = "mock"  # mock | rule | real
    stress_mode: Optional[str] = None  # None | corruption | distraction_flood | contradiction | distribution_shift
    stress_kwargs: dict = field(default_factory=dict)  # e.g. p_drop, k_noise, p_contradict, style_switch_day


@dataclass
class RunRecord:
    day: int
    turn: int
    question: Optional[str]
    gold_answer: Optional[str]
    answer: str
    citations: list[str]
    gold_fact_ids: list[str]
    retrieved: list[tuple[str, str, float]]
    correct: bool
    citation_precision: float
    citation_recall: float
    latency_retrieve_s: float
    latency_llm_s: float


@dataclass
class BenchmarkResult:
    config: BenchmarkConfig
    accuracy: float
    citation_precision: float
    citation_recall: float
    hallucination_rate: float
    memory_items_stored: int
    token_estimate: int
    retrieval_latency_avg_s: float
    forgetting_curve: list[tuple[int, float]]
    run_records: list[RunRecord] = field(default_factory=list)
    run_id: Optional[str] = None
    metrics_v2: Optional[dict] = None  # V2 metrics when computed


@dataclass
class SuiteRunConfig:
    """Config for batch suite: policies x scenarios x seeds x days x stress_modes."""
    policies: list[str]
    scenarios: list[str]
    seeds: list[int]
    number_of_days: int
    stress_modes: list[Optional[str]] = field(default_factory=lambda: [None])
    stress_kwargs: dict = field(default_factory=dict)
    llm_mode: str = "mock"
    wm_size: int = 10
    top_k: int = 5


@dataclass
class SuiteRunResult:
    run_id: str
    config_hash: str
    results: list[BenchmarkResult]
    aggregated_csv_path: Optional[str] = None
    manifest_path: Optional[str] = None
