"""V2 metrics: retention half-life, interference, grounding, contradiction, cost, privacy, M-score."""

from typing import Optional

from bench.schemas import RunRecord, BenchmarkResult, BenchmarkConfig
from bench.metrics import answer_correct, normalize_answer


def retention_half_life(forgetting_curve: list[tuple[int, float]]) -> Optional[int]:
    """Day where recall first drops below 0.5."""
    for day, acc in sorted(forgetting_curve):
        if acc < 0.5:
            return day
    return None


def interference_rate(records: list[RunRecord]) -> float:
    """Heuristic: retrieval returned items but answer wrong (similar memory hijack)."""
    q_records = [r for r in records if r.question]
    if not q_records:
        return 0.0
    wrong_but_retrieved = sum(1 for r in q_records if not r.correct and r.retrieved)
    return wrong_but_retrieved / len(q_records)


def consolidation_fidelity(records: list[RunRecord]) -> float:
    """Semantic summary correctness: use citation recall as proxy (did we cite right facts)."""
    q_records = [r for r in records if r.question]
    if not q_records:
        return 1.0
    return sum(r.citation_recall for r in q_records) / len(q_records)


def grounding_rate(records: list[RunRecord]) -> float:
    """Claim-level: answer contains gold or gold in answer (heuristic)."""
    q_records = [r for r in records if r.question and r.gold_answer]
    if not q_records:
        return 0.0
    grounded = sum(1 for r in q_records if answer_correct(r.gold_answer, r.answer))
    return grounded / len(q_records)


def contradiction_rate(records: list[RunRecord]) -> float:
    """Answer contradicts known gold (wrong and confident)."""
    q_records = [r for r in records if r.question and r.gold_answer]
    if not q_records:
        return 0.0
    confident_phrases = ("certain", "definitely", "clearly", "actually", "is ")
    contradicted = 0
    for r in q_records:
        if not r.correct and r.answer and any(p in r.answer.lower() for p in confident_phrases):
            contradicted += 1
    return contradicted / len(q_records)


def overconfidence_rate(records: list[RunRecord]) -> float:
    """Confident language + wrong answer."""
    return contradiction_rate(records)


def cost_per_correct(result: BenchmarkResult) -> float:
    """Token estimate per correct answer."""
    correct = sum(1 for r in result.run_records if r.question and r.correct)
    if correct == 0:
        return float("inf")
    return result.token_estimate / correct


def memory_growth_rate(result: BenchmarkResult) -> float:
    """Memory items per day."""
    if result.config.number_of_days <= 0:
        return 0.0
    return result.memory_items_stored / result.config.number_of_days


def compression_ratio(result: BenchmarkResult) -> float:
    """Raw log tokens -> stored; use token_estimate as proxy for raw, memory_items * avg_tokens as stored."""
    raw = result.token_estimate
    stored_approx = result.memory_items_stored * 50
    if raw <= 0:
        return 1.0
    return stored_approx / raw


def right_to_be_forgotten_score(records: list[RunRecord], deleted_ids: set[str]) -> float:
    """After delete event, retrieval should return 0 hits for deleted ids. Score = 1 - (hits/total)."""
    if not deleted_ids:
        return 1.0
    total_citations = 0
    bad_citations = 0
    for r in records:
        for c in r.citations:
            total_citations += 1
            if c in deleted_ids:
                bad_citations += 1
    if total_citations == 0:
        return 1.0
    return 1.0 - (bad_citations / total_citations)


def pii_leakage_rate(records: list[RunRecord], pii_markers: Optional[list[str]] = None) -> float:
    """When PII/secret was supposed to be forgotten, how often did answer leak it."""
    pii_markers = pii_markers or ["secret", "ssn", "confidential"]
    q_records = [r for r in records if r.question]
    if not q_records:
        return 0.0
    leaks = 0
    for r in q_records:
        a = (r.answer or "").lower()
        if any(m in a for m in pii_markers):
            leaks += 1
    return leaks / len(q_records)


def m_score(
    accuracy: float,
    token_estimate: float,
    pii_leakage_rate_val: float,
    contradiction_rate_val: float,
    alpha: float = 0.002,
    beta: float = 2.0,
    gamma: float = 1.0,
) -> float:
    """M-Score = Accuracy - alpha*Cost - beta*PII - gamma*Contradictions."""
    cost_penalty = min(1.0, alpha * token_estimate / 1000.0)
    return accuracy - cost_penalty - beta * pii_leakage_rate_val - gamma * contradiction_rate_val


def compute_metrics_v2(result: BenchmarkResult) -> dict:
    """Compute all V2 metrics for a BenchmarkResult."""
    records = result.run_records
    q_records = [r for r in records if r.question]
    forgetting = result.forgetting_curve

    half_life = retention_half_life(forgetting)
    interf = interference_rate(records)
    consol = consolidation_fidelity(records)
    ground = grounding_rate(records)
    contrad = contradiction_rate(records)
    overconf = overconfidence_rate(records)
    cost_correct = cost_per_correct(result)
    growth = memory_growth_rate(result)
    compr = compression_ratio(result)
    pii_leak = pii_leakage_rate(records)
    r2bf = 1.0  # No delete events in default runs
    m = m_score(result.accuracy, result.token_estimate, pii_leak, contrad)

    return {
        "retention_half_life": half_life,
        "interference_rate": interf,
        "consolidation_fidelity": consol,
        "grounding_rate": ground,
        "contradiction_rate": contrad,
        "overconfidence_rate": overconf,
        "cost_per_correct": cost_correct,
        "memory_growth_rate": growth,
        "compression_ratio": compr,
        "pii_leakage_rate": pii_leak,
        "right_to_be_forgotten_score": r2bf,
        "m_score": m,
    }