"""Benchmark metrics: accuracy, citation precision/recall, hallucination, cost."""

from typing import Optional
from bench.schemas import RunRecord


def normalize_answer(s: str) -> str:
    s = (s or "").strip().lower()
    for c in ".,!?;:":
        s = s.replace(c, " ")
    return " ".join(s.split())


def answer_correct(gold: Optional[str], answer: str) -> bool:
    if not gold:
        return False
    g = normalize_answer(gold)
    a = normalize_answer(answer)
    if g in a or a in g:
        return True
    g_tokens = set(g.split())
    a_tokens = set(a.split())
    overlap = len(g_tokens & a_tokens) / max(len(g_tokens), 1)
    return overlap >= 0.5


def citation_precision(cited: list[str], gold_ids: list[str]) -> float:
    if not cited:
        return 1.0 if not gold_ids else 0.0
    hit = sum(1 for c in cited if c in gold_ids)
    return hit / len(cited)


def citation_recall(cited: list[str], gold_ids: list[str]) -> float:
    if not gold_ids:
        return 1.0
    hit = sum(1 for g in gold_ids if g in cited)
    return hit / len(gold_ids)


def hallucination_heuristic(gold: Optional[str], answer: str, confident: bool = True) -> bool:
    if not confident or not answer or answer.lower().startswith("i don't"):
        return False
    return not answer_correct(gold, answer)


def compute_metrics(records: list[RunRecord]) -> dict:
    q_records = [r for r in records if r.question]
    if not q_records:
        return {
            "accuracy": 0.0,
            "citation_precision": 0.0,
            "citation_recall": 0.0,
            "hallucination_rate": 0.0,
            "retrieval_latency_avg_s": 0.0,
            "forgetting_curve": [],
        }
    correct = sum(1 for r in q_records if r.correct)
    acc = correct / len(q_records)
    cp = sum(r.citation_precision for r in q_records) / len(q_records)
    cr = sum(r.citation_recall for r in q_records) / len(q_records)
    hall = sum(1 for r in q_records if hallucination_heuristic(r.gold_answer, r.answer)) / len(q_records)
    lat = sum(r.latency_retrieve_s for r in records) / max(len(records), 1)
    by_day: dict[int, list[RunRecord]] = {}
    for r in q_records:
        by_day.setdefault(r.day, []).append(r)
    forgetting_curve = []
    for day in sorted(by_day.keys()):
        recs = by_day[day]
        acc_day = sum(1 for r in recs if r.correct) / len(recs)
        forgetting_curve.append((day, acc_day))
    return {
        "accuracy": acc,
        "citation_precision": cp,
        "citation_recall": cr,
        "hallucination_rate": hall,
        "retrieval_latency_avg_s": lat,
        "forgetting_curve": forgetting_curve,
    }
