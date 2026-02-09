"""Base types for simulation scenarios."""

from dataclasses import dataclass, field
from typing import Optional
from abc import ABC, abstractmethod


@dataclass
class ScenarioStep:
    day: int
    turn: int
    event_text: str
    question: Optional[str] = None
    gold_answer: Optional[str] = None
    gold_fact_ids: Optional[list[str]] = None


@dataclass
class GroundTruth:
    question_id: str
    question: str
    gold_answer: str
    gold_fact_ids: list[str]
    day_asked: int
    day_introduced: int


@dataclass
class Scenario:
    name: str
    steps: list[ScenarioStep]
    ground_truth: list[GroundTruth] = field(default_factory=list)

    def iter_steps(self):
        for s in self.steps:
            yield s


class BaseScenario(ABC):
    @abstractmethod
    def generate(self, num_days: int, seed: Optional[int] = None) -> Scenario:
        pass
