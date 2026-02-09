"""Unit tests for memory store/retrieval and scoring."""

import pytest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from memory.base import MemoryItem, RetrievalResult
from memory.working import WorkingMemory
from memory.episodic import EpisodicMemory
from memory.semantic import SemanticMemory
from memory.procedural import ProceduralMemory
from memory.scoring import decay_score, crowding_penalty
from memory.embeddings import TFIDFEmbedder, embed_text


def test_working_memory_store_retrieve():
    wm = WorkingMemory(capacity=3)
    wm.store(MemoryItem(id="1", timestamp=1, text="a"))
    wm.store(MemoryItem(id="2", timestamp=2, text="b"))
    wm.store(MemoryItem(id="3", timestamp=3, text="c"))
    results = wm.retrieve("any", top_k=5)
    assert len(results) == 3
    assert results[0].item.id == "3"
    wm.store(MemoryItem(id="4", timestamp=4, text="d"))
    results = wm.retrieve("any", top_k=5)
    assert len(results) == 3
    ids = [r.item.id for r in results]
    assert "1" not in ids
    assert "4" in ids


def test_episodic_memory_decay():
    ep = EpisodicMemory(decay_lambda=0.1)
    ep.set_current_day(5)
    ep.store(MemoryItem(id="e1", timestamp=1, text="old", salience_score=1.0))
    ep.store(MemoryItem(id="e2", timestamp=4, text="recent", salience_score=1.0))
    results = ep.retrieve("recent", top_k=5, day=5)
    assert len(results) >= 1
    assert max(r.score for r in results) > 0


def test_semantic_memory_store_retrieve():
    sem = SemanticMemory()
    sem.store_fact("user_coffee", "oat milk", 0.9, ["ep_1"], 1)
    sem.store_fact("user_tz", "PST", 0.8, ["ep_2"], 1)
    results = sem.retrieve("coffee", top_k=2)
    assert len(results) >= 1
    assert any("oat milk" in r.item.text for r in results)


def test_procedural_memory():
    proc = ProceduralMemory()
    proc.store_skill("incident_check", ["check dashboard", "page on-call"], 3, ["ep_1", "ep_2"], 1)
    results = proc.retrieve("incident", top_k=2)
    assert len(results) >= 1
    assert any("incident" in r.item.text.lower() for r in results)


def test_decay_score():
    assert decay_score(0, 0.1) == 1.0
    assert decay_score(1, 0.1) == pytest.approx(0.9048, rel=1e-2)
    assert decay_score(10, 0.1) < 0.5


def test_crowding_penalty():
    emb = [1.0, 0.0, 0.0]
    others_same = [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0]]
    pen = crowding_penalty(emb, others_same)
    assert pen > 0
    others_diff = [[0.0, 1.0, 0.0]]
    pen2 = crowding_penalty(emb, others_diff)
    assert pen2 < pen


def test_embed_text():
    v = embed_text("hello world")
    assert isinstance(v, list)
    assert len(v) == 64
    assert all(isinstance(x, float) for x in v)
