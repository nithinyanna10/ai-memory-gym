"""Tests for advanced memory policies: semantic_first, procedure_centric, long_term_focus."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memory.policies import (  # type: ignore  # noqa: E402
    BrainState,
    PolicyContext,
    create_brain_state,
    semantic_first,
    procedure_centric,
    long_term_focus,
)
from memory.base import MemoryItem  # type: ignore  # noqa: E402


def _make_state_with_procedure_and_fact() -> BrainState:
    state = create_brain_state(wm_size=10, decay_lambda=0.1)
    # Episodic items to consolidate into semantic/procedural in a lightweight way.
    state.episodic.set_current_day(1)
    state.episodic.store(
        MemoryItem(
            id="ep_proc",
            timestamp=1,
            text="Step 1: Check service health dashboard. Step 2: Page on-call.",
        )
    )
    state.episodic.store(
        MemoryItem(
            id="ep_fact",
            timestamp=1,
            text="User prefers coffee with oat milk.",
        )
    )
    # Manually mirror into semantic/procedural stores to avoid invoking consolidation.
    state.semantic.store(
        MemoryItem(
            id="sem_pref",
            timestamp=1,
            text="preference is oat milk",
        )
    )
    state.procedural.store(
        MemoryItem(
            id="proc_incident",
            timestamp=1,
            text="Incident runbook: Step 1: Check service health dashboard. Step 2: Page on-call.",
            metadata={"skill_name": "incident_runbook"},
        )
    )
    return state


def test_semantic_first_prioritizes_semantic():
    state = _make_state_with_procedure_and_fact()
    ctx = PolicyContext(current_day=3, current_turn=1, top_k=3)

    results = semantic_first("What does the user prefer for coffee?", state, ctx)
    assert results
    # The top result should come from semantic memory when available.
    top = results[0]
    assert "semantic" in top.reason


def test_procedure_centric_triggers_on_how_query():
    state = _make_state_with_procedure_and_fact()
    ctx = PolicyContext(current_day=3, current_turn=1, top_k=3)

    results = procedure_centric("How do I handle an incident?", state, ctx)
    assert results
    assert any("procedural" in r.reason for r in results)


def test_procedure_centric_falls_back_for_non_procedural():
    state = _make_state_with_procedure_and_fact()
    ctx = PolicyContext(current_day=3, current_turn=1, top_k=3)

    results = procedure_centric("What is the user coffee preference?", state, ctx)
    assert results
    # Fallback path should still retrieve something, but not label everything as procedural_primary.
    assert any("episodic" in r.reason or "working" in r.reason or "semantic" in r.reason for r in results)


def test_long_term_focus_prefers_older_items_when_available():
    state = create_brain_state(wm_size=10, decay_lambda=0.1)
    # Add two episodic items, one older and one recent.
    state.episodic.set_current_day(1)
    state.episodic.store(
        MemoryItem(
            id="ep_old",
            timestamp=1,
            text="Day 1: Key finding was p-value 0.03 for the primary endpoint.",
            salience_score=1.0,
        )
    )
    state.episodic.store(
        MemoryItem(
            id="ep_recent",
            timestamp=4,
            text="Day 4: Random meeting chatter.",
            salience_score=0.9,
        )
    )

    ctx = PolicyContext(current_day=5, current_turn=1, top_k=2)
    results = long_term_focus("What was the p-value for the primary endpoint?", state, ctx)
    assert results
    # Long-term focused policy should surface the older item with the answer.
    ids = [r.item.id for r in results]
    assert "ep_old" in ids

