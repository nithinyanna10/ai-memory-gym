"""Stress testing modes: corruption, distraction flood, contradiction injection, distribution shift."""

import random
import re
from typing import Optional, Tuple

from sim.base import Scenario, ScenarioStep


NOISE_TEMPLATES = [
    "Reminder: sync with team.",
    "Note: backlog item #%d pending.",
    "FYI: no update today.",
    "Slack: someone posted in #random.",
    "Calendar: optional standup at 10.",
    "Email: newsletter received.",
    "Task: review doc when free.",
    "Update: status unchanged.",
]

# Noise that is lexically similar to common target words (for similarity_to_target stress)
SIMILAR_NOISE = [
    "The rate limit is 500/min for trial.",
    "Budget discussion: we mentioned $20k earlier.",
    "API limit was set to 2000/min in doc.",
]


def distraction_flood(
    scenario: Scenario,
    k_noise: int,
    seed: Optional[int] = None,
    similarity_to_target: float = 0.0,
) -> Scenario:
    """Add k_noise irrelevant steps across days. similarity_to_target in [0,1] = fraction of noise that is similar to target (distractor)."""
    rng = random.Random(seed)
    steps = list(scenario.steps)
    if not steps:
        return scenario
    max_day = max(s.day for s in steps)
    n_similar = int(k_noise * max(0.0, min(1.0, similarity_to_target)))
    for i in range(k_noise):
        day = rng.randint(1, max(1, max_day))
        turn = max((s.turn for s in steps if s.day == day), default=0) + 1 + i
        if i < n_similar and SIMILAR_NOISE:
            text = rng.choice(SIMILAR_NOISE)
        else:
            text = rng.choice(NOISE_TEMPLATES)
            if "%d" in text:
                text = text % rng.randint(1, 999)
        steps.append(ScenarioStep(day=day, turn=turn, event_text=text, question=None, gold_answer=None, gold_fact_ids=None))
    steps.sort(key=lambda s: (s.day, s.turn))
    return Scenario(name=scenario.name, steps=steps, ground_truth=scenario.ground_truth)


def contradiction_injection(
    scenario: Scenario,
    p_contradict: float,
    seed: Optional[int] = None,
    targeted: bool = True,
) -> Scenario:
    """With probability p_contradict, add a step that contradicts an earlier fact. If targeted=True, place near the question day."""
    rng = random.Random(seed)
    if rng.random() > p_contradict or not scenario.ground_truth:
        return scenario
    gt = rng.choice(scenario.ground_truth)
    contradict_text = f"Correction: Actually the answer is NOT {gt.gold_answer}. Disregard previous info."
    steps = list(scenario.steps)
    max_turn = max((s.turn for s in steps), default=0)
    if targeted:
        q_day = gt.day_asked
        turn = max_turn + 1
    else:
        q_day = rng.randint(1, max(1, max(s.day for s in steps)))
        turn = max((s.turn for s in steps if s.day == q_day), default=0) + 1
    steps.append(ScenarioStep(day=q_day, turn=turn, event_text=contradict_text, question=None, gold_answer=None, gold_fact_ids=None))
    steps.sort(key=lambda s: (s.day, s.turn))
    return Scenario(name=scenario.name, steps=steps, ground_truth=scenario.ground_truth)


def distribution_shift(
    scenario: Scenario,
    style_switch_day: int,
    shift_style: Tuple[str, ...] = ("slack", "formal", "noisy"),
    seed: Optional[int] = None,
) -> Scenario:
    """From style_switch_day onward, mutate event text to a different style. shift_style: (slack, formal, noisy) applied in order by day."""
    if style_switch_day < 1:
        return scenario
    rng = random.Random(seed)
    styles = list(shift_style) if shift_style else ["slack"]
    new_steps = []
    for s in scenario.steps:
        if s.day >= style_switch_day and s.event_text:
            idx = (s.day - style_switch_day) % len(styles)
            style = styles[idx]
            if style == "slack":
                text = s.event_text + " [slack]"
            elif style == "formal":
                text = "Formal record: " + s.event_text
            elif style == "noisy":
                text = s.event_text + " @@ " + str(rng.randint(1, 999))
            else:
                text = s.event_text + " [" + style + "]"
        else:
            text = s.event_text
        new_steps.append(ScenarioStep(day=s.day, turn=s.turn, event_text=text, question=s.question, gold_answer=s.gold_answer, gold_fact_ids=s.gold_fact_ids))
    return Scenario(name=scenario.name, steps=new_steps, ground_truth=scenario.ground_truth)


def memory_corruption(
    state,
    p_drop: float,
    p_mutate: float,
    seed: Optional[int] = None,
    mutate_strength: float = 1.0,
) -> None:
    """In-place: randomly drop or mutate items in episodic memory. mutate_strength in [0,1]: fraction of words to corrupt (0=one word, 1=all)."""
    rng = random.Random(seed)
    if not state:
        return
    store = getattr(state, "episodic", None)
    if store is None or not hasattr(store, "_items") or not isinstance(store._items, dict):
        return
    items = list(store._items.items())
    for id_, item in items:
        if rng.random() < p_drop:
            del store._items[id_]
        elif rng.random() < p_mutate and getattr(item, "text", None):
            words = item.text.split()
            if words:
                n_corrupt = max(1, int(len(words) * max(0.0, min(1.0, mutate_strength))))
                n_corrupt = min(n_corrupt, len(words))
                indices = set(rng.sample(range(len(words)), n_corrupt))
                words = [("???" if i in indices else w) for i, w in enumerate(words)]
                item.text = " ".join(words)
