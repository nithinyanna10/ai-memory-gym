"""Prompts for agent context and completion."""

def build_context_prompt(memory_snippets: list[tuple[str, str]], current_step: str) -> str:
    parts = []
    if memory_snippets:
        parts.append("Relevant memories:")
        for mid, text in memory_snippets:
            parts.append(f"  [{mid}] {text}")
        parts.append("")
    parts.append("Current step:")
    parts.append(current_step)
    return "\n".join(parts)


def build_system_prompt() -> str:
    return """You are an assistant with access to your past memories. Use the provided relevant memories to answer accurately. If you don't have a memory that answers the question, say so. Cite memory IDs when you use them (e.g. [ep_1])."""
