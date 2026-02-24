"""Meta-memory analytics and summaries for AI Memory Gym.

This module provides utilities to introspect a BrainState across policies and runs:
- Per-store statistics (counts, avg salience, age distribution)
- Simple text summaries for visualization in UI / traces
- Safety-focused views (PII/secret-tagged counts)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from memory.base import MemoryItem
from memory.policies import BrainState


@dataclass
class StoreStats:
    count: int
    avg_salience: float
    min_timestamp: Optional[int]
    max_timestamp: Optional[int]


@dataclass
class SafetyStats:
    pii_like_items: int
    secret_like_items: int
    recent_pii_like_items: int


@dataclass
class MetaSnapshot:
    working: StoreStats
    episodic: StoreStats
    semantic: StoreStats
    procedural: StoreStats
    safety: SafetyStats
    notes: List[str]


def _store_stats(items: List[MemoryItem]) -> StoreStats:
    if not items:
        return StoreStats(count=0, avg_salience=0.0, min_timestamp=None, max_timestamp=None)
    count = len(items)
    avg_salience = sum(getattr(i, "salience_score", 1.0) for i in items) / max(count, 1)
    timestamps = [getattr(i, "timestamp", 0) for i in items]
    return StoreStats(
        count=count,
        avg_salience=avg_salience,
        min_timestamp=min(timestamps) if timestamps else None,
        max_timestamp=max(timestamps) if timestamps else None,
    )


def _safety_flags_from_text(text: str) -> Tuple[bool, bool]:
    t = (text or "").lower()
    pii_markers = ("ssn", "passport", "social security", "credit card", "dob ")
    secret_markers = ("password", "api key", "secret", "token")
    pii = any(m in t for m in pii_markers)
    secret = any(m in t for m in secret_markers)
    return pii, secret


def _safety_stats(all_items: List[MemoryItem], current_day: int, recent_window: int = 2) -> SafetyStats:
    pii_like = 0
    secret_like = 0
    recent_pii = 0
    for item in all_items:
        pii, secret = _safety_flags_from_text(getattr(item, "text", "") or "")
        if pii:
            pii_like += 1
            if current_day - getattr(item, "timestamp", current_day) <= recent_window:
                recent_pii += 1
        if secret:
            secret_like += 1
    return SafetyStats(
        pii_like_items=pii_like,
        secret_like_items=secret_like,
        recent_pii_like_items=recent_pii,
    )


def snapshot_brain_state(state: Optional[BrainState], current_day: int) -> Optional[MetaSnapshot]:
    """Create a compact snapshot of the given BrainState for debugging / UI."""
    if state is None:
        return None

    working_items = state.working.list_items()
    episodic_items = state.episodic.list_items()
    semantic_items = state.semantic.list_items()
    procedural_items = state.procedural.list_items()

    all_items: List[MemoryItem] = []
    all_items.extend(working_items)
    all_items.extend(episodic_items)
    all_items.extend(semantic_items)
    all_items.extend(procedural_items)

    safety = _safety_stats(all_items, current_day=current_day)

    notes: List[str] = []
    if safety.pii_like_items > 0:
        notes.append(f"Detected {safety.pii_like_items} PII-like memories; consider stricter policies.")
    if safety.secret_like_items > 0:
        notes.append(f"Detected {safety.secret_like_items} secret-like memories.")
    if safety.recent_pii_like_items > 0:
        notes.append(f"{safety.recent_pii_like_items} PII-like memories were stored in the last few days.")
    if len(episodic_items) > 500:
        notes.append("High episodic load; consolidation / pruning may be needed.")

    return MetaSnapshot(
        working=_store_stats(working_items),
        episodic=_store_stats(episodic_items),
        semantic=_store_stats(semantic_items),
        procedural=_store_stats(procedural_items),
        safety=safety,
        notes=notes,
    )


def summarize_snapshot(snapshot: MetaSnapshot) -> Dict[str, str]:
    """Return short human-readable summaries (for logs/UI)."""
    parts: Dict[str, str] = {}
    parts["counts"] = (
        f"WM={snapshot.working.count}, EP={snapshot.episodic.count}, "
        f"SEM={snapshot.semantic.count}, PROC={snapshot.procedural.count}"
    )
    parts["salience"] = (
        f"avg_salience: WM={snapshot.working.avg_salience:.2f}, "
        f"EP={snapshot.episodic.avg_salience:.2f}, "
        f"SEM={snapshot.semantic.avg_salience:.2f}, "
        f"PROC={snapshot.procedural.avg_salience:.2f}"
    )
    parts["safety"] = (
        f"PII-like={snapshot.safety.pii_like_items}, "
        f"secret-like={snapshot.safety.secret_like_items}, "
        f"recent PII-like={snapshot.safety.recent_pii_like_items}"
    )
    if snapshot.notes:
        parts["notes"] = " | ".join(snapshot.notes)
    else:
        parts["notes"] = "No notable meta-memory issues detected."
    return parts

