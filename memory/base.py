"""Base types for memory systems."""

from dataclasses import dataclass, field
from typing import Any, Optional
from abc import ABC, abstractmethod


@dataclass
class MemoryItem:
    """Base memory item with common fields."""
    id: str
    timestamp: int  # day/step
    text: str
    tags: list[str] = field(default_factory=list)
    salience_score: float = 1.0
    source_turn: Optional[int] = None
    embedding: Optional[list[float]] = None
    links_to: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = {
            "id": self.id,
            "timestamp": self.timestamp,
            "text": self.text,
            "tags": self.tags,
            "salience_score": self.salience_score,
            "source_turn": self.source_turn,
            "links_to": self.links_to,
            "metadata": self.metadata,
        }
        if self.embedding is not None:
            d["embedding"] = self.embedding
        return d


@dataclass
class RetrievalResult:
    """Result of a memory retrieval with score and item."""
    item: MemoryItem
    score: float
    reason: str = ""


class BaseMemory(ABC):
    """Abstract base for memory stores."""

    @abstractmethod
    def store(self, item: MemoryItem) -> None:
        """Store a memory item."""
        pass

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5, **kwargs) -> list[RetrievalResult]:
        """Retrieve relevant items for a query."""
        pass

    @abstractmethod
    def list_items(self, **kwargs) -> list[MemoryItem]:
        """List stored items (for inspection)."""
        pass
