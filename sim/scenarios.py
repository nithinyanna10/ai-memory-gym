"""Scenario types: PersonalAssistant, Research, Ops."""

import random
from typing import Optional
from sim.base import Scenario, ScenarioStep, GroundTruth, BaseScenario


def _make_step(day: int, turn: int, event: str, question: Optional[str] = None, gold: Optional[str] = None, fact_ids: Optional[list[str]] = None, tags: Optional[dict] = None) -> ScenarioStep:
    return ScenarioStep(
        day=day, turn=turn, event_text=event,
        question=question, gold_answer=gold, gold_fact_ids=fact_ids or [], tags=tags,
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


class SalesCRMScenario(BaseScenario):
    """Customer requirements across calls; recall constraints later."""

    def generate(self, num_days: int, seed: Optional[int] = None) -> Scenario:
        rng = random.Random(seed)
        steps = []
        gt = []
        turn = 0
        steps.append(_make_step(1, 1, "Call with Acme Corp: They need API rate limit of 1000/min.", tags={"constraint": True}))
        steps.append(_make_step(1, 2, "Acme Corp: Budget cap $20k for Q1."))
        steps.append(_make_step(2, 3, "Email from Acme: Prefer Slack integration."))
        turn = 3
        for d in range(3, min(6, num_days + 1)):
            steps.append(_make_step(d, turn + 1, f"Check-in day {d}: no new constraints."))
            turn += 1
        q_day = min(6, num_days)
        steps.append(_make_step(q_day, turn + 1, "Question: What is Acme's API rate limit requirement?",
                                question="What is Acme's API rate limit requirement?", gold="1000/min", fact_ids=[]))
        gt.append(GroundTruth(question_id="q_rate", question="What is Acme's API rate limit requirement?",
                              gold_answer="1000/min", gold_fact_ids=[], day_asked=q_day, day_introduced=1))
        return Scenario(name="sales_crm", steps=steps, ground_truth=gt)


class LegalContractScenario(BaseScenario):
    """Clauses, versions, redlines; gotcha traps."""

    def generate(self, num_days: int, seed: Optional[int] = None) -> Scenario:
        rng = random.Random(seed)
        steps = []
        gt = []
        steps.append(_make_step(1, 1, "Contract v1: Liability cap is $500k."))
        steps.append(_make_step(2, 2, "Redline: Liability cap changed to $1M in Section 4.2."))
        steps.append(_make_step(3, 3, "Final signed: Section 4.2 liability cap $1M."))
        q_day = min(5, num_days)
        steps.append(_make_step(q_day, 4, "Question: What is the signed liability cap?",
                                question="What is the signed liability cap?", gold="$1M", fact_ids=[]))
        gt.append(GroundTruth(question_id="q_liability", question="What is the signed liability cap?",
                              gold_answer="$1M", gold_fact_ids=[], day_asked=q_day, day_introduced=3))
        return Scenario(name="legal_contract", steps=steps, ground_truth=gt)


class MeetingMemoryScenario(BaseScenario):
    """Action items across meetings; later asked what we decided."""

    def generate(self, num_days: int, seed: Optional[int] = None) -> Scenario:
        rng = random.Random(seed)
        steps = []
        gt = []
        steps.append(_make_step(1, 1, "Meeting 1: Action item - Sarah to own the dashboard by Friday."))
        steps.append(_make_step(2, 2, "Meeting 2: We decided to use Postgres for the new service."))
        steps.append(_make_step(3, 3, "Standup: Dashboard delayed to next week."))
        q_day = min(5, num_days)
        steps.append(_make_step(q_day, 4, "Question: What did we decide about the database for the new service?",
                                question="What did we decide about the database for the new service?", gold="Postgres", fact_ids=[]))
        gt.append(GroundTruth(question_id="q_db", question="What did we decide about the database for the new service?",
                              gold_answer="Postgres", gold_fact_ids=[], day_asked=q_day, day_introduced=2))
        return Scenario(name="meeting_memory", steps=steps, ground_truth=gt)


class MultiAgentHandoffScenario(BaseScenario):
    """Agent A stores memory; agent B must retrieve (same scenario, two phases)."""

    def generate(self, num_days: int, seed: Optional[int] = None) -> Scenario:
        steps = []
        gt = []
        steps.append(_make_step(1, 1, "Agent A stores: Customer ticket #4421 resolution = refund approved."))
        steps.append(_make_step(1, 2, "Handoff to Agent B."))
        steps.append(_make_step(2, 3, "Agent B is asked: Was ticket #4421 refund approved?",
                                question="Was ticket #4421 refund approved?", gold="yes", fact_ids=[]))
        gt.append(GroundTruth(question_id="q_4421", question="Was ticket #4421 refund approved?",
                              gold_answer="yes", gold_fact_ids=[], day_asked=2, day_introduced=1))
        return Scenario(name="multi_agent_handoff", steps=steps, ground_truth=gt)


class AdversarialInjectionScenario(BaseScenario):
    """Someone says false fact later; does agent overwrite?"""

    def generate(self, num_days: int, seed: Optional[int] = None) -> Scenario:
        rng = random.Random(seed)
        steps = []
        gt = []
        steps.append(_make_step(1, 1, "Fact: The launch date is March 15."))
        steps.append(_make_step(2, 2, "Someone says: Actually the launch date is March 1.", tags={"contradiction": True}))
        steps.append(_make_step(3, 3, "Question: When is the launch date?",
                                question="When is the launch date?", gold="March 15", fact_ids=[]))
        gt.append(GroundTruth(question_id="q_launch", question="When is the launch date?",
                              gold_answer="March 15", gold_fact_ids=[], day_asked=3, day_introduced=1))
        return Scenario(name="adversarial_injection", steps=steps, ground_truth=gt)


def get_scenario(scenario_type: str) -> BaseScenario:
    m = {
        "personal_assistant": PersonalAssistantScenario(),
        "research": ResearchScenario(),
        "ops": OpsScenario(),
        "sales_crm": SalesCRMScenario(),
        "legal_contract": LegalContractScenario(),
        "meeting_memory": MeetingMemoryScenario(),
        "multi_agent_handoff": MultiAgentHandoffScenario(),
        "adversarial_injection": AdversarialInjectionScenario(),
    }
    return m.get(scenario_type, PersonalAssistantScenario())
