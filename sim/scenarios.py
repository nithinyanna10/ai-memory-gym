"""Scenario types: PersonalAssistant, Research, Ops."""

import random
from typing import Optional
from sim.base import Scenario, ScenarioStep, GroundTruth, BaseScenario


def _make_step(day: int, turn: int, event: str, question: Optional[str] = None, gold: Optional[str] = None, fact_ids: Optional[list[str]] = None) -> ScenarioStep:
    return ScenarioStep(
        day=day, turn=turn, event_text=event,
        question=question, gold_answer=gold, gold_fact_ids=fact_ids or [],
    )


class PersonalAssistantScenario(BaseScenario):
    PREFS = [
        ("coffee", "User prefers coffee with oat milk."),
        ("meeting_time", "User likes morning meetings before 10am."),
        ("timezone", "User is in PST timezone."),
    ]
    DISTRACTORS = [
        "Reminder: check email.",
        "Note: weather will be sunny tomorrow.",
        "Reminder: team standup at 9am.",
    ]

    def generate(self, num_days: int, seed: Optional[int] = None) -> Scenario:
        rng = random.Random(seed)
        steps = []
        gt = []
        turn = 0
        for key, text in self.PREFS:
            turn += 1
            steps.append(_make_step(1, turn, text))
        steps.append(_make_step(1, turn + 1, rng.choice(self.DISTRACTORS)))
        steps.append(_make_step(2, turn + 2, "Constraint: Do not schedule meetings on Fridays."))
        turn += 2
        for day in range(3, min(6, num_days + 1)):
            steps.append(_make_step(day, turn + 1, rng.choice(self.DISTRACTORS)))
            turn += 1
        q_day = min(7, num_days)
        steps.append(_make_step(q_day, turn + 1, "The assistant is asked: What does the user prefer for coffee?",
                                question="What does the user prefer for coffee?", gold="oat milk", fact_ids=["ep_1"]))
        gt.append(GroundTruth(question_id="q_coffee", question="What does the user prefer for coffee?",
                              gold_answer="oat milk", gold_fact_ids=["ep_1"], day_asked=q_day, day_introduced=1))
        turn += 1
        steps.append(_make_step(q_day, turn + 1, "The assistant is asked: What timezone is the user in?",
                                question="What timezone is the user in?", gold="PST", fact_ids=["ep_3"]))
        gt.append(GroundTruth(question_id="q_tz", question="What timezone is the user in?",
                              gold_answer="PST", gold_fact_ids=["ep_3"], day_asked=q_day, day_introduced=1))
        return Scenario(name="personal_assistant", steps=steps, ground_truth=gt)


class ResearchScenario(BaseScenario):
    SNIPPETS_DAY1 = [
        "Document A: The project deadline is March 15.",
        "Document B: Budget for Q1 is $50k.",
        "Document C: Key contact is alice@company.com.",
    ]
    SNIPPETS_DAY3 = [
        "Document D: The actual deadline was revised to March 20.",
        "Document E: Budget was increased to $55k in February.",
    ]
    CONTRADICTIONS = ["Memo: Ignore previous deadline; use March 10 as final."]

    def generate(self, num_days: int, seed: Optional[int] = None) -> Scenario:
        rng = random.Random(seed)
        steps = []
        gt = []
        turn = 0
        for t in self.SNIPPETS_DAY1:
            turn += 1
            steps.append(_make_step(1, turn, t))
        for t in self.SNIPPETS_DAY3:
            turn += 1
            steps.append(_make_step(3, turn, t))
        steps.append(_make_step(4, turn + 1, rng.choice(self.CONTRADICTIONS)))
        turn += 1
        q_day = min(6, num_days)
        steps.append(_make_step(q_day, turn + 1, "Question: What is the Q1 budget?",
                                question="What is the Q1 budget?", gold="$55k", fact_ids=[]))
        gt.append(GroundTruth(question_id="q_budget", question="What is the Q1 budget?", gold_answer="$55k",
                              gold_fact_ids=[], day_asked=q_day, day_introduced=3))
        return Scenario(name="research", steps=steps, ground_truth=gt)


class OpsScenario(BaseScenario):
    PROCEDURE = [
        "Step 1: Check service health dashboard.",
        "Step 2: If error rate > 5%, page on-call.",
        "Step 3: Create incident ticket and post in #incidents.",
    ]
    INCIDENT = [
        "Incident started at 14:00 UTC. Service X down.",
        "At 14:30 root cause identified: deployment bug.",
        "At 15:00 rollback completed. Service restored.",
    ]

    def generate(self, num_days: int, seed: Optional[int] = None) -> Scenario:
        rng = random.Random(seed)
        steps = []
        gt = []
        turn = 0
        for t in self.PROCEDURE:
            turn += 1
            steps.append(_make_step(1, turn, t))
        for t in self.INCIDENT:
            turn += 1
            steps.append(_make_step(2, turn, t))
        for day in range(3, min(5, num_days + 1)):
            steps.append(_make_step(day, turn + 1, f"Daily check day {day}."))
            turn += 1
        q_day = min(5, num_days)
        steps.append(_make_step(q_day, turn + 1, "Question: What is step 2 of the incident procedure?",
                                question="What is step 2 of the incident procedure?", gold="page on-call", fact_ids=[]))
        gt.append(GroundTruth(question_id="q_step2", question="What is step 2 of the incident procedure?",
                              gold_answer="page on-call", gold_fact_ids=[], day_asked=q_day, day_introduced=1))
        return Scenario(name="ops", steps=steps, ground_truth=gt)


def get_scenario(scenario_type: str) -> BaseScenario:
    m = {
        "personal_assistant": PersonalAssistantScenario(),
        "research": ResearchScenario(),
        "ops": OpsScenario(),
    }
    return m.get(scenario_type, PersonalAssistantScenario())
