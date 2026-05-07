"""Switchable memory policies."""

from dataclasses import dataclass
from typing import Optional

from memory.base import MemoryItem, RetrievalResult
from memory.working import WorkingMemory
from memory.episodic import EpisodicMemory
from memory.semantic import SemanticMemory
from memory.procedural import ProceduralMemory
from memory.embeddings import TFIDFEmbedder, embed_text
from memory.consolidation import run_consolidation


@dataclass
class PolicyContext:
    current_day: int
    current_turn: int
    top_k: int = 5
    wm_size: int = 10
    decay_lambda: float = 0.1
    salience_threshold: float = 0.3
    rehearsal_frequency: int = 3
    embedder: Optional[TFIDFEmbedder] = None


@dataclass
class BrainState:
    working: WorkingMemory
    episodic: EpisodicMemory
    semantic: SemanticMemory
    procedural: ProceduralMemory
    running_summary: str = ""
    last_rehearsal_day: int = -1


def no_memory(query: str, state: Optional[BrainState], ctx: PolicyContext) -> list[RetrievalResult]:
    return []


def full_log(query: str, state: Optional[BrainState], ctx: PolicyContext) -> list[RetrievalResult]:
    if state is None:
        return []
    return state.working.retrieve(query, top_k=ctx.top_k)


def rolling_summary(query: str, state: Optional[BrainState], ctx: PolicyContext) -> list[RetrievalResult]:
    if state is None:
        return []
    wm_results = state.working.retrieve(query, top_k=ctx.top_k)
    if state.running_summary:
        summary_item = MemoryItem(id="summary", timestamp=ctx.current_day, text=state.running_summary, salience_score=1.0)
        wm_results.insert(0, RetrievalResult(item=summary_item, score=1.0, reason="running_summary"))
    return wm_results[: ctx.top_k]


def vector_rag(query: str, state: Optional[BrainState], ctx: PolicyContext) -> list[RetrievalResult]:
    if state is None:
        return []
    state.episodic.set_current_day(ctx.current_day)
    query_emb = embed_text(query)
    return state.episodic.retrieve(query, top_k=ctx.top_k, query_embedding=query_emb, day=ctx.current_day)


def hybrid_brain(query: str, state: Optional[BrainState], ctx: PolicyContext) -> list[RetrievalResult]:
    if state is None:
        return []
    state.episodic.set_current_day(ctx.current_day)
    query_emb = embed_text(query)
    results = []
    wm = state.working.retrieve(query, top_k=min(3, ctx.top_k))
    ep = state.episodic.retrieve(query, top_k=ctx.top_k, query_embedding=query_emb, day=ctx.current_day)
    sem = state.semantic.retrieve(query, top_k=min(3, ctx.top_k))
    proc = state.procedural.retrieve(query, top_k=min(2, ctx.top_k))
    for r in wm:
        r.reason = "working"
        results.append(r)
    for r in ep:
        r.reason = "episodic"
        results.append(r)
    for r in sem:
        results.append(r)
    for r in proc:
        results.append(r)
    results.sort(key=lambda x: -x.score)
    return results[: ctx.top_k]


def salience_only(query: str, state: Optional[BrainState], ctx: PolicyContext) -> list[RetrievalResult]:
    if state is None:
        return []
    state.episodic.set_current_day(ctx.current_day)
    all_ep = state.episodic.retrieve(query, top_k=ctx.top_k * 2, day=ctx.current_day)
    return [r for r in all_ep if r.item.salience_score >= ctx.salience_threshold][: ctx.top_k]


def rehearsal(query: str, state: Optional[BrainState], ctx: PolicyContext) -> list[RetrievalResult]:
    return hybrid_brain(query, state, ctx)


def semantic_first(query: str, state: Optional[BrainState], ctx: PolicyContext) -> list[RetrievalResult]:
    """Hybrid policy that prioritizes distilled semantic facts, then episodic context, then working/procedural.

    Useful for questions that ask for crisp facts or preferences where consolidation has already run.
    """
    if state is None:
        return []
    state.episodic.set_current_day(ctx.current_day)

    query_emb = embed_text(query)
    sem_results = state.semantic.retrieve(query, top_k=ctx.top_k)
    ep_results = state.episodic.retrieve(
        query,
        top_k=ctx.top_k * 2,
        query_embedding=query_emb,
        day=ctx.current_day,
    )
    wm_results = state.working.retrieve(query, top_k=min(2, ctx.top_k))
    proc_results = state.procedural.retrieve(query, top_k=min(2, ctx.top_k))

    tiered_results: list[tuple[int, RetrievalResult]] = []
    for r in sem_results:
        r.reason = "semantic_primary"
        tiered_results.append((0, r))
    for r in ep_results:
        # keep original reason from EpisodicMemory, just nudge a bit lower than semantic
        r.score *= 0.95
        tiered_results.append((1, r))
    for r in wm_results:
        r.reason = "working_context"
        r.score *= 0.9
        tiered_results.append((2, r))
    for r in proc_results:
        r.reason = "procedural_support"
        r.score *= 0.9
        tiered_results.append((3, r))

    # Deduplicate repeated items surfaced by different stores.
    deduped: dict[str, tuple[int, RetrievalResult]] = {}
    for tier, result in tiered_results:
        existing = deduped.get(result.item.id)
        if existing is None:
            deduped[result.item.id] = (tier, result)
            continue
        existing_tier, existing_result = existing
        if tier < existing_tier or (tier == existing_tier and result.score > existing_result.score):
            deduped[result.item.id] = (tier, result)

    ranked = [r for _, r in sorted(deduped.values(), key=lambda tr: (tr[0], -tr[1].score))]
    return ranked[: ctx.top_k]


def procedure_centric(query: str, state: Optional[BrainState], ctx: PolicyContext) -> list[RetrievalResult]:
    """Policy that routes 'how-to' and procedure-style questions through ProceduralMemory first.

    For non-procedural queries it falls back to hybrid_brain.
    """
    if state is None:
        return []

    q_lower = query.lower()
    procedural_markers = (
        "how do i",
        "how to",
        "step",
        "steps",
        "runbook",
        "procedure",
        "incident",
        "playbook",
        "checklist",
    )
    is_procedural = any(m in q_lower for m in procedural_markers)
    if not is_procedural:
        return hybrid_brain(query, state, ctx)

    state.episodic.set_current_day(ctx.current_day)
    query_emb = embed_text(query)

    proc_results = state.procedural.retrieve(query, top_k=ctx.top_k)
    ep_results = state.episodic.retrieve(
        query,
        top_k=ctx.top_k * 2,
        query_embedding=query_emb,
        day=ctx.current_day,
    )
    wm_results = state.working.retrieve(query, top_k=min(2, ctx.top_k))

    results: list[RetrievalResult] = []
    for r in proc_results:
        r.reason = "procedural_primary"
        r.score *= 1.2
        results.append(r)
    for r in ep_results:
        r.reason = "episodic_support"
        r.score *= 0.95
        results.append(r)
    for r in wm_results:
        r.reason = "working_context"
        r.score *= 0.9
        results.append(r)

    results.sort(key=lambda x: -x.score)
    return results[: ctx.top_k]


def long_term_focus(query: str, state: Optional[BrainState], ctx: PolicyContext) -> list[RetrievalResult]:
    """Bias toward older episodic memories (long-horizon recall) while still allowing recent context.

    This is useful for scenarios like research_long or legal_contract where the question
    might refer back to information introduced many days earlier.
    """
    if state is None:
        return []

    state.episodic.set_current_day(ctx.current_day)
    query_emb = embed_text(query)

    ep_results = state.episodic.retrieve(
        query,
        top_k=ctx.top_k * 3,
        query_embedding=query_emb,
        day=ctx.current_day,
    )
    if not ep_results:
        # Fallback to hybrid if nothing stored yet
        return hybrid_brain(query, state, ctx)

    # Prefer items at least 2 "days" old; if not enough, fill with the best of the rest.
    long_horizon: list[RetrievalResult] = []
    recent: list[RetrievalResult] = []
    for r in ep_results:
        age = ctx.current_day - r.item.timestamp
        if age >= 2:
            r.reason = "episodic_long_term"
            long_horizon.append(r)
        else:
            r.reason = "episodic_recent"
            recent.append(r)

    long_horizon.sort(key=lambda x: -x.score)
    recent.sort(key=lambda x: -x.score)

    selected: list[RetrievalResult] = long_horizon[: ctx.top_k]
    if len(selected) < ctx.top_k:
        needed = ctx.top_k - len(selected)
        selected.extend(recent[:needed])

    # Provide a very small amount of working/semantic context for grounding.
    wm_results = state.working.retrieve(query, top_k=1)
    sem_results = state.semantic.retrieve(query, top_k=1)
    for r in wm_results:
        r.reason = "working_support"
        r.score *= 0.8
        selected.append(r)
    for r in sem_results:
        r.reason = "semantic_support"
        r.score *= 0.9
        selected.append(r)

    selected.sort(key=lambda x: -x.score)
    return selected[: ctx.top_k]


POLICIES = {
    "no_memory": no_memory,
    "full_log": full_log,
    "rolling_summary": rolling_summary,
    "vector_rag": vector_rag,
    "hybrid_brain": hybrid_brain,
    "salience_only": salience_only,
    "rehearsal": rehearsal,
    "semantic_first": semantic_first,
    "procedure_centric": procedure_centric,
    "long_term_focus": long_term_focus,
}


def get_policy(name: str):
    return POLICIES.get(name, full_log)


def create_brain_state(wm_size: int = 10, decay_lambda: float = 0.1) -> BrainState:
    return BrainState(
        working=WorkingMemory(capacity=wm_size),
        episodic=EpisodicMemory(decay_lambda=decay_lambda),
        semantic=SemanticMemory(),
        procedural=ProceduralMemory(),
    )
