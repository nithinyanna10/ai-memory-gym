"""Working memory: last N turns (simple list)."""

from memory.base import MemoryItem, RetrievalResult, BaseMemory


class WorkingMemory(BaseMemory):
    """Keeps the last N turns as a simple FIFO list."""

    def __init__(self, capacity: int = 10):
        self.capacity = capacity
        self._items: list[MemoryItem] = []

    def store(self, item: MemoryItem) -> None:
        self._items.append(item)
        while len(self._items) > self.capacity:
            self._items.pop(0)

    def retrieve(self, query: str, top_k: int = 5, **kwargs) -> list[RetrievalResult]:
        n = min(top_k, len(self._items))
        recent = list(reversed(self._items))[:n]
        return [
            RetrievalResult(item=item, score=1.0 - (i / max(n, 1)), reason="working_memory_recent")
            for i, item in enumerate(recent)
        ]

    def list_items(self, **kwargs) -> list[MemoryItem]:
        return list(self._items)

    def clear(self) -> None:
        self._items.clear()
