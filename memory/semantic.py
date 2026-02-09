"""Semantic memory: distilled facts as key-value statements with confidence."""

from dataclasses import dataclass
from memory.base import MemoryItem, RetrievalResult, BaseMemory


@dataclass
class SemanticFact:
    id: str
    key: str
    value: str
    confidence: float
    provenance_ids: list[str]
    timestamp: int


class SemanticMemory(BaseMemory):
    def __init__(self):
        self._facts: dict[str, SemanticFact] = {}
        self._id_counter = 0

    def _next_id(self) -> str:
        self._id_counter += 1
        return f"sem_{self._id_counter}"

    def store_fact(self, key: str, value: str, confidence: float, provenance_ids: list[str], timestamp: int) -> str:
        fid = self._next_id()
        self._facts[fid] = SemanticFact(
            id=fid, key=key, value=value, confidence=confidence,
            provenance_ids=provenance_ids, timestamp=timestamp,
        )
        return fid

    def store(self, item: MemoryItem) -> None:
        key = item.metadata.get("key", item.id)
        value = item.text
        conf = item.salience_score
        prov = item.links_to or []
        ts = item.timestamp
        self.store_fact(key, value, conf, prov, ts)

    def retrieve(self, query: str, top_k: int = 5, **kwargs) -> list[RetrievalResult]:
        ql = query.lower()
        scored = []
        for fid, fact in self._facts.items():
            score = 0.0
            if ql in fact.key.lower():
                score += 0.5
            if ql in fact.value.lower():
                score += 0.5
            qw = set(ql.split())
            kw = set(fact.key.lower().split()) | set(fact.value.lower().split())
            overlap = len(qw & kw) / max(len(qw), 1)
            score = score * 0.5 + overlap * 0.5
            score *= fact.confidence
            scored.append((fact, score))
        scored.sort(key=lambda x: -x[1])
        results = []
        for fact in scored[:top_k]:
            mi = MemoryItem(
                id=fact[0].id,
                timestamp=fact[0].timestamp,
                text=f"{fact[0].key}: {fact[0].value}",
                tags=[],
                salience_score=fact[0].confidence,
                links_to=fact[0].provenance_ids,
            )
            results.append(RetrievalResult(item=mi, score=fact[1], reason="semantic_match"))
        return results

    def list_items(self, **kwargs) -> list[MemoryItem]:
        out = []
        for f in self._facts.values():
            out.append(MemoryItem(
                id=f.id, timestamp=f.timestamp, text=f"{f.key}: {f.value}",
                tags=[], salience_score=f.confidence, links_to=f.provenance_ids,
            ))
        return out

    def get_by_id(self, id: str) -> SemanticFact | None:
        return self._facts.get(id)
