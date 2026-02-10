"""RuleLLM: deterministic but realistic error patterns (forget, interference, contradiction susceptibility)."""

import random
from typing import Optional

from agent.llm import BaseLLM


class RuleLLM(BaseLLM):
    """Semi-realistic deterministic LLM: decays recall with time gap and distraction, interference, contradiction susceptibility."""

    def __init__(
        self,
        ground_truth_hints: Optional[list[str]] = None,
        seed: Optional[int] = 42,
        forget_decay: float = 0.15,
        interference_prob: float = 0.2,
        contradiction_susceptible: bool = True,
        verification_mode: bool = False,
    ):
        self.ground_truth_hints = ground_truth_hints or []
        self.rng = random.Random(seed)
        self.forget_decay = forget_decay
        self.interference_prob = interference_prob
        self.contradiction_susceptible = contradiction_susceptible
        self.verification_mode = verification_mode

    def complete(self, prompt: str, system: Optional[str] = None, **kwargs) -> str:
        prompt_lower = prompt.lower()
        # If verification mode and we see "Correction:" or "Disregard", prefer the correction (don't be susceptible)
        if self.verification_mode and ("correction:" in prompt_lower or "disregard" in prompt_lower):
            # Still try to use gold if present and not contradicted
            for h in self.ground_truth_hints:
                if h and h.lower() in prompt_lower and "not " + h.lower() not in prompt_lower:
                    return h.strip()
            return "I need to verify; previous info may have been corrected."

        # Contradiction injection: if prompt says "Actually the answer is NOT X", sometimes return wrong
        if self.contradiction_susceptible and "not " in prompt_lower and "actually" in prompt_lower:
            if self.rng.random() < 0.4:
                # Return the contradicted (wrong) answer
                for h in self.ground_truth_hints:
                    if h and ("not " + h.lower() in prompt_lower or "disregard" in prompt_lower):
                        return h.strip()
                return "Based on the correction, I no longer recall the original."

        # Interference: sometimes return a near-miss (different hint)
        if self.ground_truth_hints and len(self.ground_truth_hints) > 1 and self.rng.random() < self.interference_prob:
            wrong = self.rng.choice(self.ground_truth_hints)
            return wrong.strip()

        # Forget: with probability that increases with prompt length (proxy for time/density)
        word_count = len(prompt.split())
        forget_p = min(0.5, word_count * self.forget_decay * 0.01)
        if self.rng.random() < forget_p:
            return "I don't recall."
        if self.rng.random() < 0.1:
            return "I'm not sure."

        # Normal: match hint to prompt
        for h in self.ground_truth_hints:
            if h and h.lower() in prompt_lower:
                return h.strip()
        if self.ground_truth_hints:
            return self.ground_truth_hints[0].strip()
        if "?" in prompt:
            return "Based on my memory, I believe so."
        return "Acknowledged."
