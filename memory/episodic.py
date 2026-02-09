"""Episodic memory: events with timestamps, tags, salience, embeddings."""

from memory.base import MemoryItem, RetrievalResult, BaseMemory
from memory.scoring import decay_score, crowding_penalty
import math


class EpisodicMemory(BaseMemory):
    """Stores events with id, timestamp, text, tags, salience, embedding, links_to."""

    def __init__(self, decay_lambda: float = 0.1, use_crowding: bool = True):
        self.decay_lambda = decay_lambda
        self.use_crowding = use_crowding
        self._items: dict[str, MemoryItem] = {}
        self._id_counter = 0
        self._current_day = 0

    def set_current_day(self, day: int) -> None:
        self._current_day = day

    def _next_id(self) -> str:
        self._id_counter += 1
        return f"ep_{self._id_counter}"

    def store(self, item: MemoryItem) -> None:
        if not item.id:
            item.id = self._next_id()
        self._items[item.id] = item

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        query_embedding: list[float] | None = None,
        day: int | None = None,
        **kwargs,
    ) -> list[RetrievalResult]:
        day = day if day is not None else self._current_day
        items = list(self._items.values())
        if not items:
            return []

        scored: list[tuple[MemoryItem, float]] = []
        for item in items:
            age = day - item.timestamp
            base = item.salience_score * decay_score(age, self.decay_lambda)
            if query_embedding and item.embedding:
                sim = _cosine_sim(query_embedding, item.embedding)
                base = base * (0.5 + 0.5 * max(0, sim))
            scored.append((item, base))

        scored.sort(key=lambda x: -x[1])
        if self.use_crowding:
            all_embs = [item.embedding for item, _ in scored if item.embedding]
            if len(all_embs) > 1:
                penalized = []
                for item, s in scored:
                    if item.embedding:
                        others = [e for e in all_embs if e is not item.embedding]
                        pen = crowding_penalty(item.embedding, others)
                        penalized.append((item, s * (1.0 - pen)))
                    else:
                        penalized.append((item, s))
                penalized.sort(key=lambda x: -x[1])
                scored = penalized

        top = scored[:top_k]
        return [
            RetrievalResult(item=item, score=score, reason="episodic_decay_salience")
            for item, score in top
        ]

    def list_items(self, day: int | None = None, **kwargs) -> list[MemoryItem]:
        return list(self._items.values())

    def get_by_id(self, id: str) -> MemoryItem | None:
        return self._items.get(id)

    def prune_below_priority(self, threshold: float, day: int) -> int:
        to_remove = []
        for id_, item in self._items.items():
            age = day - item.timestamp
            p = item.salience_score * decay_score(age, self.decay_lambda)
            if p < threshold:
                to_remove.append(id_)
        for id_ in to_remove:
            del self._items[id_]
        return len(to_remove)


def _cosine_sim(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
