"""FastAPI service.

    POST /v1/query       question -> grounded answer with citations, or an abstention
    POST /v1/documents   add documents (chunked, embedded, indexed)
    GET  /healthz        liveness + index size
    GET  /metrics        Prometheus metrics
"""
from __future__ import annotations

import json
import os
import threading
import time
from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from .pipeline import RAGPipeline
from .retriever import BM25Index
from .text import chunk_words

ROOT = Path(__file__).resolve().parent.parent


class QueryIn(BaseModel):
    query: str = Field(min_length=3, max_length=2000)
    k: int = Field(default=5, ge=1, le=20)


class DocIn(BaseModel):
    doc_id: str = Field(pattern=r"^[A-Za-z0-9_\-.]{1,64}$")
    title: str = ""
    text: str = Field(min_length=1, max_length=200_000)


class DocsIn(BaseModel):
    documents: list[DocIn] = Field(min_length=1, max_length=500)


def build_pipeline(corpus_name: str | None = None) -> RAGPipeline:
    """Build the index for RAG_CORPUS (creep by default, or squad) with its tuned config."""
    from .embeddings import get_embedder
    from .gate import Gate
    from .llm import get_provider
    from .retriever import HybridRetriever

    name = corpus_name or os.getenv("RAG_CORPUS", "creep")
    cfg_path = ROOT / "results" / f"{name}_config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    emb = get_embedder(os.getenv("RAG_EMBEDDER", "wordllama"))
    reranker = None
    if name == "creep":
        from .creep import build
        from .metadata import MetadataFilter

        docs = build().docs
        ids, texts, titles = [d.doc_id for d in docs], [d.text for d in docs], [d.title for d in docs]
        reranker = MetadataFilter({d.doc_id: dict(d.meta) for d in docs if d.meta},
                                  {d.doc_id: d.alloy for d in docs if d.alloy})
    elif name == "squad":
        from . import corpus

        col = corpus.load()
        ids, texts, titles = [d.doc_id for d in col.docs], [f"{d.title}. {d.text}" for d in col.docs], None
    else:
        raise ValueError(f"unknown corpus {name!r}")
    retr = HybridRetriever(ids, texts, emb, rrf_k=cfg.get("rrf_k", 60), dense_weight=cfg.get("dense_weight", 1.0),
                           titles=titles)
    gate = Gate(**cfg["gate"]) if "gate" in cfg else None
    return RAGPipeline(retr, get_provider(), gate, k=cfg.get("k", 5), reranker=reranker)


def create_app(pipeline: RAGPipeline | None = None, cache_size: int = 1024,
               log_path: Path | None = None) -> FastAPI:
    app = FastAPI(title="creep-rag-service", version="0.1.0")
    state: dict = {"pipeline": pipeline}
    cache: OrderedDict[tuple, dict] = OrderedDict()
    lock = threading.Lock()
    reg = CollectorRegistry()
    m_req = Counter("rag_requests_total", "Queries by outcome", ["outcome"], registry=reg)
    m_lat = Histogram("rag_request_seconds", "End-to-end query latency", registry=reg,
                      buckets=(.002, .005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5, 10))
    m_tok = Counter("rag_llm_tokens_total", "LLM tokens", ["direction"], registry=reg)
    m_hit = Counter("rag_cache_hits_total", "Answer cache hits", registry=reg)

    def pipe() -> RAGPipeline:
        if state["pipeline"] is None:
            state["pipeline"] = build_pipeline()
        return state["pipeline"]

    @app.get("/healthz")
    def healthz():
        p = pipe()
        return {"status": "ok", "corpus": os.getenv("RAG_CORPUS", "creep"), "documents": len(p.retriever.texts),
                "provider": p.provider.name,
                "embedder": p.retriever.embedder.name, "gate": p.gate is not None}

    @app.post("/v1/query")
    def query(q: QueryIn):
        t = time.perf_counter()
        key = (q.query.strip().lower(), q.k)
        with lock:
            if key in cache:
                cache.move_to_end(key)
                m_hit.inc()
                m_req.labels("cache_hit").inc()
                return {**cache[key], "cached": True}
        try:
            ans = pipe().answer(q.query, q.k)
        except Exception as e:  # provider outage: fail loudly, never fabricate
            m_req.labels("error").inc()
            raise HTTPException(503, f"generation failed: {type(e).__name__}") from e
        out = asdict(ans)
        m_req.labels(ans.reason).inc()
        m_tok.labels("in").inc(ans.usage.get("input_tokens") or 0)
        m_tok.labels("out").inc(ans.usage.get("output_tokens") or 0)
        m_lat.observe(time.perf_counter() - t)
        with lock:
            cache[key] = out
            if len(cache) > cache_size:
                cache.popitem(last=False)
        if log_path:
            with lock, open(log_path, "a") as f:
                f.write(json.dumps({"ts": time.time(), "query": q.query, "reason": ans.reason,
                                    "gate_prob": ans.gate_prob, "citations": ans.citations,
                                    "timings_ms": ans.timings_ms, "usage": ans.usage}) + "\n")
        return {**out, "cached": False}

    @app.post("/v1/documents")
    def add_documents(body: DocsIn):
        p = pipe()
        r = p.retriever
        ids, texts = [], []
        for d in body.documents:
            for i, chunk in enumerate(chunk_words(d.text)):
                cid = f"{d.doc_id}#{i}"
                if cid in r.texts:
                    raise HTTPException(409, f"{cid} already indexed")
                ids.append(cid)
                texts.append(f"{d.title}. {chunk}" if d.title else chunk)
        with lock:
            r.store.add(ids, r.embedder.embed(texts))
            r.texts.update(zip(ids, texts))
            # BM25 statistics are corpus-wide, so the lexical index is rebuilt (fine to ~100k chunks)
            r.bm25 = BM25Index(list(r.texts), list(r.texts.values()))
            cache.clear()
        return {"added_chunks": len(ids), "documents": len(r.texts)}

    @app.get("/metrics")
    def metrics():
        return PlainTextResponse(generate_latest(reg), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app(log_path=Path(os.getenv("RAG_REQUEST_LOG", "/tmp/rag_requests.jsonl")))
