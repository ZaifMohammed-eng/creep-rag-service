"""Retrieve -> gate -> generate -> verify citations."""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

import numpy as np

from .gate import Gate, signals, title_match
from .llm import LLMProvider
from .retriever import HybridRetriever
from .text import tokenize

SYSTEM = (
    "You answer questions using only the numbered context passages you are given. "
    "Give the shortest answer that fully answers the question, then cite the passage id "
    "in square brackets, for example: 1912 [a03p012]. "
    "If the passages do not contain the answer, reply exactly INSUFFICIENT_CONTEXT. "
    "Do not use outside knowledge."
)
_CITE = re.compile(r"\[([A-Za-z0-9_\-:.#]+)\]")


def build_prompt(question: str, passages: list[tuple[str, str]]) -> str:
    ctx = "\n".join(f"[{pid}] {text}" for pid, text in passages)
    return f"Context:\n{ctx}\n\nQuestion: {question}"


@dataclass
class Answer:
    query: str
    answer: str | None
    abstained: bool
    reason: str
    citations: list[str] = field(default_factory=list)
    retrieved: list[dict] = field(default_factory=list)
    gate_prob: float | None = None
    timings_ms: dict[str, float] = field(default_factory=dict)
    usage: dict[str, float | int | None] = field(default_factory=dict)


class RAGPipeline:
    def __init__(self, retriever: HybridRetriever, provider: LLMProvider, gate: Gate | None = None,
                 k: int = 5, depth: int = 50, reranker=None):
        self.retriever, self.provider, self.gate, self.k, self.depth = retriever, provider, gate, k, depth
        self.reranker = reranker          # optional, e.g. rag.metadata.MetadataFilter

    def retrieve(self, query: str, k: int | None = None, qvec: np.ndarray | None = None):
        r = self.retriever
        if qvec is None:
            qvec = r.embedder.embed([query])
        lex = r.bm25.search(query, self.depth)
        dense = r.store.search(qvec.reshape(1, -1), self.depth)[0]
        k = k or self.k
        hits = r.fuse(lex, dense, self.depth if self.reranker else k)
        if self.reranker is not None:
            hits = self.reranker.rerank(query, hits, k)
        qt = tokenize(query, drop_stop=False)
        tm = title_match(qt, tokenize(r.titles.get(hits[0].doc_id, ""), drop_stop=False)) if hits else 0.0
        fm = self.reranker.match_score(query) if self.reranker is not None else None
        return hits, signals(lex, dense, len(tokenize(query)), tm, fm)

    def answer(self, query: str, k: int | None = None) -> Answer:
        t0 = time.perf_counter()
        hits, sig = self.retrieve(query, k)
        t1 = time.perf_counter()
        retrieved = [{"doc_id": h.doc_id, "score": round(h.score, 5)} for h in hits]
        timings = {"retrieve": (t1 - t0) * 1000}

        # a question that names an entity the index holds is in scope by construction;
        # the learned gate decides only for everything else
        verified = self.reranker is not None and self.reranker.entity(query) is not None
        if self.gate is not None and not verified:
            ok, p = self.gate.allow(sig)
            if not ok:
                return Answer(query, None, True, "low_retrieval_confidence", [], retrieved, p,
                              {**timings, "total": timings["retrieve"]})
        else:
            p = 1.0 if verified else None

        passages = [(h.doc_id, self.retriever.texts[h.doc_id]) for h in hits]
        resp = self.provider.generate(SYSTEM, build_prompt(query, passages))
        t2 = time.perf_counter()
        timings.update(generate=(t2 - t1) * 1000, total=(t2 - t0) * 1000)
        usage = {"model": resp.model, "input_tokens": resp.input_tokens,
                 "output_tokens": resp.output_tokens, "cost_usd": resp.cost_usd}

        if resp.text.strip().startswith("INSUFFICIENT_CONTEXT"):
            return Answer(query, None, True, "model_declined", [], retrieved, p, timings, usage)
        allowed = {h.doc_id for h in hits}
        cites = list(dict.fromkeys(_CITE.findall(resp.text)))
        valid = [c for c in cites if c in allowed]
        text = _CITE.sub("", resp.text).strip()
        if not valid:
            # an answer that cites nothing we retrieved is not grounded: do not return it
            return Answer(query, None, True, "ungrounded_answer", cites, retrieved, p, timings, usage)
        return Answer(query, text, False, "ok", valid, retrieved, p, timings, usage)
