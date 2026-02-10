"""Tests for stress modes and suite runner."""

import pytest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sim.base import Scenario, ScenarioStep
from sim.stress import distraction_flood, contradiction_injection, distribution_shift, memory_corruption
from memory.policies import create_brain_state
from memory.base import MemoryItem


def test_distraction_flood_deterministic():
    scenario = Scenario(name="t", steps=[ScenarioStep(1, 1, "Hello", None, None, None)], ground_truth=[])
    out1 = distraction_flood(scenario, k_noise=3, seed=42)
    out2 = distraction_flood(scenario, k_noise=3, seed=42)
    assert len(out1.steps) == len(out2.steps)
    assert len(out1.steps) == 1 + 3


def test_contradiction_injection():
    from sim.base import GroundTruth
    gt = [GroundTruth("q1", "Q?", "A", [], 2, 1)]
    scenario = Scenario(name="t", steps=[ScenarioStep(1, 1, "Fact: A", None, None, None)], ground_truth=gt)
    out = contradiction_injection(scenario, p_contradict=1.0, seed=42)
    assert len(out.steps) >= 2


def test_distribution_shift():
    scenario = Scenario(name="t", steps=[ScenarioStep(1, 1, "Formal text"), ScenarioStep(2, 2, "More")], ground_truth=[])
    out = distribution_shift(scenario, style_switch_day=2)
    assert "[casual]" in out.steps[1].event_text
    assert "[casual]" not in out.steps[0].event_text


def test_memory_corruption_episodic():
    state = create_brain_state()
    state.episodic.store(MemoryItem(id="ep_1", timestamp=1, text="hello world"))
    state.episodic.store(MemoryItem(id="ep_2", timestamp=2, text="foo bar"))
    n_before = len(state.episodic.list_items())
    memory_corruption(state, p_drop=0.99, p_mutate=0.0, seed=42)
    n_after = len(state.episodic.list_items())
    assert n_after <= n_before
