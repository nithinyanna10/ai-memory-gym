"""Procedural memory: reusable skills/checklists from repeated patterns."""

from dataclasses import dataclass
from memory.base import MemoryItem, RetrievalResult, BaseMemory


@dataclass
class Skill:
    id: str
    name: str
    steps: list[str]
    success_count: int
    provenance_ids: list[str]
    timestamp: int


class ProceduralMemory(BaseMemory):
    def __init__(self):
        self._skills: dict[str, Skill] = {}
        self._id_counter = 0

    def _next_id(self) -> str:
        self._id_counter += 1
        return f"proc_{self._id_counter}"

    def store_skill(self, name: str, steps: list[str], success_count: int, provenance_ids: list[str], timestamp: int) -> str:
        sid = self._next_id()
        self._skills[sid] = Skill(
            id=sid, name=name, steps=steps, success_count=success_count,
            provenance_ids=provenance_ids, timestamp=timestamp,
        )
        return sid

    def store(self, item: MemoryItem) -> None:
        name = item.metadata.get("skill_name", item.id)
        steps = item.metadata.get("steps", [item.text])
        if isinstance(steps, str):
            steps = [steps]
        self.store_skill(name, steps, int(item.salience_score), item.links_to or [], item.timestamp)

    def retrieve(self, query: str, top_k: int = 5, **kwargs) -> list[RetrievalResult]:
        ql = query.lower()
        scored = []
        for sid, skill in self._skills.items():
            score = 0.0
            if ql in skill.name.lower():
                score += 0.6
            for step in skill.steps:
                if ql in step.lower():
                    score += 0.4 / max(len(skill.steps), 1)
            score *= (0.5 + 0.5 * min(1.0, skill.success_count / 5.0))
            scored.append((skill, score))
        scored.sort(key=lambda x: -x[1])
        results = []
        for skill in scored[:top_k]:
            text = f"{skill[0].name}: " + " | ".join(skill[0].steps)
            mi = MemoryItem(
                id=skill[0].id, timestamp=skill[0].timestamp, text=text,
                tags=[], salience_score=float(skill[0].success_count), links_to=skill[0].provenance_ids,
            )
            results.append(RetrievalResult(item=mi, score=skill[1], reason="procedural_match"))
        return results

    def list_items(self, **kwargs) -> list[MemoryItem]:
        out = []
        for s in self._skills.values():
            text = f"{s.name}: " + " | ".join(s.steps)
            out.append(MemoryItem(
                id=s.id, timestamp=s.timestamp, text=text,
                tags=[], salience_score=float(s.success_count), links_to=s.provenance_ids,
            ))
        return out
