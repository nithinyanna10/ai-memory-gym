"""Synthetic task generator."""

from typing import Optional
from sim.base import Scenario, ScenarioStep
from sim.scenarios import get_scenario


def generate_scenario_steps(
    scenario_type: str,
    num_days: int,
    seed: Optional[int] = None,
) -> Scenario:
    gen = get_scenario(scenario_type)
    return gen.generate(num_days=num_days, seed=seed)
