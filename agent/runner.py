"""AgentRunner: consumes step, composes context from memory policy, calls LLM, stores memories."""

import time
from dataclasses import dataclass
from typing import Optional

from memory.base import MemoryItem, RetrievalResult
from memory.policies import get_policy, PolicyContext, BrainState, create_brain_state, run_consolidation
from memory.embeddings import embed_text
from agent.llm import BaseLLM
from agent.prompts import build_context_prompt, build_system_prompt


@dataclass
class StepInput:
    day: int
    turn: int
    event_text: str
    question: Optional[str] = None
    gold_answer: Optional[str] = None
    gold_fact_ids: Optional[list[str]] = None


@dataclass
class StepOutput:
    answer: str
    citations: list[str]
    retrieved: list[tuple[str, str, float]]
    latency_retrieve_s: float = 0.0
    latency_llm_s: float = 0.0


class AgentRunner:
    def __init__(
        self,
        policy_name: str = "full_log",
        wm_size: int = 10,
        top_k: int = 5,
        decay_lambda: float = 0.1,
        salience_threshold: float = 0.3,
        rehearsal_frequency: int = 3,
        run_consolidation_at_end_of_day: bool = True,
    ):
        self.policy_name = policy_name
        self.wm_size = wm_size
        self.top_k = top_k
        self.decay_lambda = decay_lambda
        self.salience_threshold = salience_threshold
        self.rehearsal_frequency = rehearsal_frequency
        self.run_consolidation_at_end_of_day = run_consolidation_at_end_of_day
        self.llm: Optional[BaseLLM] = None
        self.state: Optional[BrainState] = None
        self._policy = get_policy(policy_name)
        self._last_day_consolidated = -1

    def set_llm(self, llm: BaseLLM) -> None:
        self.llm = llm

    def reset_state(self) -> None:
        self.state = create_brain_state(wm_size=self.wm_size, decay_lambda=self.decay_lambda)
        self._policy = get_policy(self.policy_name)
        self._last_day_consolidated = -1

    def run_step(self, step: StepInput, llm: Optional[BaseLLM] = None) -> StepOutput:
        if self.state is None:
            self.reset_state()
        llm = llm or self.llm
        if llm is None:
            from agent.llm import MockLLM
            llm = MockLLM()
            self.llm = llm

        ctx = PolicyContext(
            current_day=step.day,
            current_turn=step.turn,
            top_k=self.top_k,
            wm_size=self.wm_size,
            decay_lambda=self.decay_lambda,
            salience_threshold=self.salience_threshold,
            rehearsal_frequency=self.rehearsal_frequency,
        )
        self.state.episodic.set_current_day(step.day)

        t0 = time.perf_counter()
        query = step.question or step.event_text
        results = self._policy(query, self.state, ctx)
        latency_retrieve = time.perf_counter() - t0

        memory_snippets = [(r.item.id, r.item.text[:200]) for r in results]
        retrieved_log = [(r.item.id, r.reason, r.score) for r in results]

        prompt = build_context_prompt(memory_snippets, step.event_text)
        if step.question:
            prompt += "\n\nQuestion: " + step.question
        t1 = time.perf_counter()
        answer = llm.complete(prompt, system=build_system_prompt())
        latency_llm = time.perf_counter() - t1

        salience = 0.5
        if step.gold_answer or step.gold_fact_ids:
            salience = 0.9
        emb = embed_text(step.event_text)
        ep_item = MemoryItem(
            id="",
            timestamp=step.day,
            text=step.event_text,
            tags=[],
            salience_score=salience,
            source_turn=step.turn,
            embedding=emb,
            links_to=[],
        )
        self.state.episodic.store(ep_item)
        self.state.working.store(ep_item)

        citations = []
        for r in results:
            if r.item.id and f"[{r.item.id}]" in answer:
                citations.append(r.item.id)

        if self.run_consolidation_at_end_of_day and self.policy_name in ("hybrid_brain", "rehearsal", "vector_rag"):
            if step.day > self._last_day_consolidated:
                run_consolidation(
                    step.day,
                    self.state.episodic,
                    self.state.semantic,
                    self.state.procedural,
                    top_n_episodes=20,
                    prune_threshold=0.05,
                )
                self._last_day_consolidated = step.day

        return StepOutput(
            answer=answer,
            citations=citations,
            retrieved=retrieved_log,
            latency_retrieve_s=latency_retrieve,
            latency_llm_s=latency_llm,
        )
