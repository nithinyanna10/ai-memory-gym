"""Consolidation pass: extract semantic facts, prune episodes, update procedural skills."""

from memory.episodic import EpisodicMemory
from memory.semantic import SemanticMemory
from memory.procedural import ProceduralMemory
from memory.scoring import decay_score
from typing import Callable, Optional
import re


def run_consolidation(
    day: int,
    episodic: EpisodicMemory,
    semantic: SemanticMemory,
    procedural: ProceduralMemory,
    *,
    top_n_episodes: int = 20,
    prune_threshold: float = 0.05,
    extract_facts: bool = True,
    extract_llm: Optional[Callable[[list[str], int], list[tuple[str, str]]]] = None,
) -> dict:
    episodic.set_current_day(day)
    items = episodic.list_items()
    if not items:
        return {"episodes_pruned": 0, "facts_added": 0, "skills_updated": 0}

    scored = [(item, item.salience_score * decay_score(day - item.timestamp, episodic.decay_lambda)) for item in items]
    scored.sort(key=lambda x: -x[1])
    top = [item for item, _ in scored[:top_n_episodes]]

    facts_added = 0
    if extract_facts:
        for item in top:
            extracted = _rule_based_extract(item.text)
            if extract_llm and item.text:
                try:
                    llm_facts = extract_llm([item.text], day)
                    for k, v in llm_facts:
                        extracted.append((k, v))
                except Exception:
                    pass
            for key, value in extracted:
                if key and value:
                    semantic.store_fact(key, value, item.salience_score, [item.id], day)
                    facts_added += 1

    episodes_pruned = episodic.prune_below_priority(prune_threshold, day)

    skills_updated = 0
    step_patterns: dict[str, list[str]] = {}
    for item in top:
        steps = _extract_steps(item.text)
        if steps:
            key = "|".join(steps[:3])
            step_patterns.setdefault(key, []).append(item.id)
    for pattern, prov_ids in step_patterns.items():
        if len(prov_ids) >= 2:
            steps = pattern.split("|")
            procedural.store_skill(f"procedure_{day}", steps, len(prov_ids), prov_ids, day)
            skills_updated += 1

    return {"episodes_pruned": episodes_pruned, "facts_added": facts_added, "skills_updated": skills_updated}


def _rule_based_extract(text: str) -> list[tuple[str, str]]:
    out = []
    text_lower = text.lower()
    for m in re.finditer(r"(\w+(?:\s+\w+)*)\s+is\s+([^.]+?)(?:\.|$)", text, re.IGNORECASE):
        k, v = m.group(1).strip(), m.group(2).strip()
        if len(k) > 2 and len(v) > 1:
            out.append((f"fact_{k.replace(' ', '_')}", v))
    for m in re.finditer(r"(\w+(?:\s+\w+)*)\s*=\s*([^.]+?)(?:\.|$)", text):
        k, v = m.group(1).strip(), m.group(2).strip()
        if len(k) > 2 and len(v) > 1:
            out.append((f"fact_{k.replace(' ', '_')}", v))
    for m in re.finditer(r"(?:prefers?|likes?)\s+([^.]+?)(?:\.|$)", text_lower):
        v = m.group(1).strip()
        if len(v) > 1:
            out.append(("preference", v))
    for m in re.finditer(r"reminder[:\s]+([^.]+?)(?:\.|$)", text_lower):
        out.append(("reminder", m.group(1).strip()))
    return out


def _extract_steps(text: str) -> list[str]:
    steps = []
    for m in re.finditer(r"(?:step\s*\d+|step\s*[a-z]|\d+\.)\s*[:\-]?\s*([^.]+?)(?:\.|$)", text, re.IGNORECASE):
        steps.append(m.group(1).strip())
    return steps
