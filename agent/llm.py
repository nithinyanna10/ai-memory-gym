"""LLM client: OpenAI-compatible or MockLLM fallback (no key required)."""

import os
from abc import ABC, abstractmethod
from typing import Optional


class BaseLLM(ABC):
    @abstractmethod
    def complete(self, prompt: str, system: Optional[str] = None, **kwargs) -> str:
        pass


class MockLLM(BaseLLM):
    def __init__(self, ground_truth_hints: Optional[list[str]] = None):
        self.ground_truth_hints = ground_truth_hints or []

    def set_hints(self, hints: list[str]) -> None:
        self.ground_truth_hints = hints

    def complete(self, prompt: str, system: Optional[str] = None, **kwargs) -> str:
        if self.ground_truth_hints:
            for h in self.ground_truth_hints:
                if h and h.lower() in prompt.lower():
                    return h.strip()
            return self.ground_truth_hints[0].strip() if self.ground_truth_hints else "I don't recall."
        if "?" in prompt:
            return "Based on my memory, I believe so."
        return "Acknowledged."


class OpenAILLM(BaseLLM):
    """OpenAI-compatible API: OpenAI, Ollama, Ollama Cloud, Azure, etc."""

    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "") or "ollama"
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"

    def complete(self, prompt: str, system: Optional[str] = None, **kwargs) -> str:
        try:
            from openai import OpenAI
        except ImportError:
            return "[openai package not installed]"
        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = client.chat.completions.create(model=self.model, messages=messages, **kwargs)
        return (resp.choices[0].message.content or "").strip()


def get_llm(
    use_mock: bool = True,
    ground_truth_hints: Optional[list[str]] = None,
    llm_mode: Optional[str] = None,
    **kwargs,
) -> BaseLLM:
    mode = llm_mode or ("mock" if use_mock else "real")
    if mode == "rule":
        from agent.rule_llm import RuleLLM
        return RuleLLM(ground_truth_hints=ground_truth_hints, seed=kwargs.get("seed", 42))
    if mode == "mock":
        return MockLLM(ground_truth_hints=ground_truth_hints)
    key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL") or kwargs.get("base_url")
    model = os.environ.get("OPENAI_MODEL") or kwargs.get("model", "gpt-4o-mini")
    if not key and not base_url:
        return MockLLM(ground_truth_hints=ground_truth_hints)
    return OpenAILLM(
        model=model,
        api_key=key or "ollama",
        base_url=base_url,
        **{k: v for k, v in kwargs.items() if k not in ("model", "api_key", "base_url")},
    )
