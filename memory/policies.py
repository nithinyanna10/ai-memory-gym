"""Switchable memory policies."""

from dataclasses import dataclass
from typing import Optional
from memory.base import MemoryItem, RetrievalResult
from memory.working import WorkingMemory
from memory.episodic import EpisodicMemory
from memory.semantic import SemanticMemory
from memory.procedural import ProceduralMemory
from memory.embeddings import embed_text, TFIDFEmbedder
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


POLICIES = {
    "no_memory": no_memory,
    "full_log": full_log,
    "rolling_summary": rolling_summary,
    "vector_rag": vector_rag,
    "hybrid_brain": hybrid_brain,
    "salience_only": salience_only,
    "rehearsal": rehearsal,
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
