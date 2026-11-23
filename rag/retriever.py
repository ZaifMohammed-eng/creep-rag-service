"""BM25, dense and hybrid (reciprocal rank fusion) retrieval."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from .embeddings import Embedder
from .text import tokenize
from .vectorstore import FaissStore, VectorStore


@dataclass
class Hit:
    doc_id: str
    score: float
    bm25: float | None = None
    dense: float | None = None


class BM25Index:
    def __init__(self, ids: list[str], texts: list[str], k1: float = 1.2, b: float = 0.75):
        self.ids = ids
        self.bm25 = BM25Okapi([tokenize(t) for t in texts], k1=k1, b=b)

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        top = np.argsort(-scores)[:k]
        return [(self.ids[i], float(scores[i])) for i in top]


class HybridRetriever:
    """Runs BM25 and dense search, then fuses ranks with RRF.

    RRF uses ranks only, so the two score scales never need calibrating against each
    other. `dense_weight` scales the dense list's contribution; it and `rrf_k` are
    tuned on the calibration split by `eval/run_creep_eval.py` and `eval/run_squad_eval.py` (Cormack et al. 2009 use
    rrf_k=60, weight 1).
    """

    def __init__(self, ids: list[str], texts: list[str], embedder: Embedder,
                 store: VectorStore | None = None, rrf_k: int = 60, dense_weight: float = 1.0,
                 depth: int = 50, titles: list[str] | None = None):
        self.texts = dict(zip(ids, texts))
        self.titles = dict(zip(ids, titles)) if titles else {}
        self.embedder = embedder
        self.bm25 = BM25Index(ids, texts)
        self.store = store or FaissStore(embedder.dim, "flat")
        if len(self.store) == 0:
            self.store.add(ids, embedder.embed(texts))
        self.rrf_k, self.dense_weight, self.depth = rrf_k, dense_weight, depth

    def search(self, query: str, k: int = 5, mode: str = "hybrid",
               qvec: np.ndarray | None = None) -> list[Hit]:
        if mode == "bm25":
            return [Hit(d, s, bm25=s) for d, s in self.bm25.search(query, k)]
        if qvec is None:
            qvec = self.embedder.embed([query])
        dense = self.store.search(qvec.reshape(1, -1), self.depth if mode == "hybrid" else k)[0]
        if mode == "dense":
            return [Hit(d, s, dense=s) for d, s in dense[:k]]
        lex = self.bm25.search(query, self.depth)
        return self.fuse(lex, dense, k)

    def fuse(self, lex: list[tuple[str, float]], dense: list[tuple[str, float]], k: int) -> list[Hit]:
        fused: dict[str, Hit] = {}
        for rank, (d, s) in enumerate(lex):
            h = fused.setdefault(d, Hit(d, 0.0))
            h.score += 1.0 / (self.rrf_k + rank + 1)
            h.bm25 = s
        for rank, (d, s) in enumerate(dense):
            h = fused.setdefault(d, Hit(d, 0.0))
            h.score += self.dense_weight / (self.rrf_k + rank + 1)
            h.dense = s
        return sorted(fused.values(), key=lambda h: -h.score)[:k]
