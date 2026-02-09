"""Scoring and decay utilities for memory."""

import math
from typing import Sequence


def decay_score(age: int, lambda_: float) -> float:
    """priority = salience * exp(-lambda * age). Age in days/steps."""
    return math.exp(-lambda_ * max(0, age))


def crowding_penalty(
    embedding: list[float],
    all_embeddings: Sequence[list[float]],
    exclude_self: list[float] | None = None,
    k: int = 5,
) -> float:
    """Penalize retrieval when many similar memories compete (interference)."""
    sims = []
    for other in all_embeddings:
        if other is exclude_self or other == exclude_self:
            continue
        s = _cosine_sim(embedding, other)
        sims.append(s)
    if not sims:
        return 0.0
    sims.sort(reverse=True)
    top = sims[:k]
    return min(0.5, sum(top) / max(len(top), 1) * 0.5)


def _cosine_sim(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return max(-1.0, min(1.0, dot / (na * nb)))
