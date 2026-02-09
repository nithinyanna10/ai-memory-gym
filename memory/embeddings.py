"""Lightweight embedding: TF-IDF vectorizer fallback (no GPU)."""

import hashlib
import math
from typing import Optional


class TFIDFEmbedder:
    def __init__(self, dim: int = 64):
        self.dim = dim
        self._vocab: dict[str, int] = {}
        self._idf: dict[str, float] = {}
        self._doc_count = 0
        self._doc_freq: dict[str, int] = {}

    def _tokenize(self, text: str) -> list[str]:
        return [t.lower() for t in text.replace(".", " ").replace(",", " ").split() if len(t) > 1]

    def fit(self, texts: list[str]) -> None:
        self._doc_count = len(texts)
        self._doc_freq = {}
        for text in texts:
            seen = set()
            for t in self._tokenize(text):
                self._doc_freq[t] = self._doc_freq.get(t, 0) + (1 if t not in seen else 0)
                seen.add(t)
        for t, df in self._doc_freq.items():
            self._idf[t] = math.log((self._doc_count + 1) / (df + 1)) + 1
        vocab_list = sorted(self._idf.keys())[: self.dim * 4]
        self._vocab = {w: i % self.dim for i, w in enumerate(vocab_list)}

    def embed(self, text: str) -> list[float]:
        tokens = self._tokenize(text)
        vec = [0.0] * self.dim
        for t in tokens:
            if t in self._vocab:
                idx = self._vocab[t]
                idf = self._idf.get(t, 1.0)
                vec[idx] += idf
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def embed_without_fit(self, text: str) -> list[float]:
        tokens = self._tokenize(text)
        vec = [0.0] * self.dim
        for t in tokens:
            h = int(hashlib.md5(t.encode()).hexdigest(), 16) % (self.dim * 1000)
            idx = h % self.dim
            vec[idx] += 1.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]


_global_embedder: Optional[TFIDFEmbedder] = None


def get_embedder(fit_texts: Optional[list[str]] = None, dim: int = 64) -> TFIDFEmbedder:
    global _global_embedder
    if _global_embedder is None:
        _global_embedder = TFIDFEmbedder(dim=dim)
        if fit_texts:
            _global_embedder.fit(fit_texts)
    return _global_embedder


def embed_text(text: str, fit_texts: Optional[list[str]] = None) -> list[float]:
    emb = get_embedder(fit_texts=fit_texts)
    if emb._vocab:
        return emb.embed(text)
    return emb.embed_without_fit(text)
