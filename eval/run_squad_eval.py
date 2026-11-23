"""Offline evaluation: retrieval quality, fusion tuning, vector index trade-offs, the
abstention gate, and end-to-end pipeline latency.

Generic benchmark for the retrieval engine on SQuAD v1.1 dev (Wikipedia). Every tuned
quantity (fusion weight, RRF constant, gate) is fit on calibration articles and reported
once on test articles. Writes results/squad_metrics.json and results/squad_config.json.

    python scripts/get_data.py && python eval/run_squad_eval.py
"""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag import corpus  # noqa: E402
from rag.embeddings import get_embedder  # noqa: E402
from rag.gate import FEATURES, Gate, matrix, signals  # noqa: E402
from rag.llm import ExtractiveProvider  # noqa: E402
from rag.pipeline import RAGPipeline  # noqa: E402
from rag.retriever import HybridRetriever  # noqa: E402
from rag.text import tokenize  # noqa: E402
from rag.vectorstore import FaissStore, QdrantStore  # noqa: E402

RESULTS = ROOT / "results"
DEPTH = 50


def rank_metrics(ranked: list[list[str]], gold: list[str]) -> dict[str, float]:
    ranks = []
    for r, g in zip(ranked, gold):
        ranks.append(r.index(g) + 1 if g in r else None)
    def recall(k):
        return float(np.mean([x is not None and x <= k for x in ranks]))
    mrr = float(np.mean([1 / x if x and x <= 10 else 0 for x in ranks]))
    ndcg = float(np.mean([1 / np.log2(x + 1) if x and x <= 10 else 0 for x in ranks]))
    return {"recall@1": recall(1), "recall@5": recall(5), "recall@10": recall(10),
            "mrr@10": mrr, "ndcg@10": ndcg, "n": len(gold)}


def fuse_ids(lex, dense, rrf_k, w, k=10):
    f: dict[str, float] = {}
    for r, (d, _) in enumerate(lex):
        f[d] = f.get(d, 0) + 1 / (rrf_k + r + 1)
    for r, (d, _) in enumerate(dense):
        f[d] = f.get(d, 0) + w / (rrf_k + r + 1)
    return sorted(f, key=lambda d: -f[d])[:k]


def pct(a, q):
    return round(float(np.percentile(a, q)), 3)


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    s = np.concatenate([pos, neg])
    ranks = s.argsort().argsort() + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    col = corpus.load()
    ids = [d.doc_id for d in col.docs]
    texts = [f"{d.title}. {d.text}" for d in col.docs]
    emb = get_embedder("wordllama")

    t = time.perf_counter()
    retr = HybridRetriever(ids, texts, emb)
    build_s = time.perf_counter() - t

    in_cal, oos_cal = col.queries("calib")
    in_te, oos_te = col.queries("test")
    groups = {"in_cal": in_cal, "oos_cal": oos_cal, "in_te": in_te, "oos_te": oos_te}
    print({k: len(v) for k, v in groups.items()}, "docs", len(ids))

    # one pass of both retrievers per query; everything downstream reuses these lists
    runs = {}
    for name, qs in groups.items():
        Q = emb.embed([q.text for q in qs])
        lex = [retr.bm25.search(q.text, DEPTH) for q in qs]
        dense = retr.store.search(Q, DEPTH)
        runs[name] = (qs, Q, lex, dense)

    # ---- fusion tuning on calibration articles -------------------------------------
    qs, _, lex, dense = runs["in_cal"]
    gold = [q.gold_doc for q in qs]
    grid = []
    for rrf_k in (5, 10, 30, 60):
        for w in (0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0):
            m = rank_metrics([fuse_ids(l, d, rrf_k, w) for l, d in zip(lex, dense)], gold)
            grid.append({"rrf_k": rrf_k, "dense_weight": w, **m})
    best = max(grid, key=lambda g: g["mrr@10"])
    rrf_k, w = best["rrf_k"], best["dense_weight"]
    print("tuned fusion", rrf_k, w, round(best["mrr@10"], 4))

    # ---- retrieval quality on test articles ----------------------------------------
    qs, Q, lex, dense = runs["in_te"]
    gold = [q.gold_doc for q in qs]
    retrieval = {
        "bm25": rank_metrics([[d for d, _ in l[:10]] for l in lex], gold),
        "dense (wordllama)": rank_metrics([[d for d, _ in x[:10]] for x in dense], gold),
        "hybrid RRF (k=60, w=1)": rank_metrics([fuse_ids(l, d, 60, 1.0) for l, d in zip(lex, dense)], gold),
        f"hybrid RRF tuned (k={rrf_k}, w={w})": rank_metrics(
            [fuse_ids(l, d, rrf_k, w) for l, d in zip(lex, dense)], gold),
    }
    # weaker embedding baseline: averaged word vectors
    try:
        sp = get_embedder("spacy")
        sp_store = FaissStore(sp.dim)
        sp_store.add(ids, sp.embed(texts))
        retrieval["dense (spaCy word vectors)"] = rank_metrics(
            [[d for d, _ in r] for r in sp_store.search(sp.embed([q.text for q in qs]), 10)], gold)
    except (ImportError, OSError):
        print("spaCy baseline skipped (pip install -e '.[baselines]')")
    for k_, v in retrieval.items():
        print(f"{k_:38s} R@1 {v['recall@1']:.3f} R@5 {v['recall@5']:.3f} MRR {v['mrr@10']:.3f}")

    # ---- vector index trade-offs ---------------------------------------------------
    D = emb.embed(texts)
    flat = retr.store
    index_cmp = {}
    exact = flat.search(Q, 10)
    for name, store in (("faiss-flat", flat), ("faiss-hnsw", FaissStore(emb.dim, "hnsw"))):
        if name != "faiss-flat":
            store.add(ids, D)
        t = time.perf_counter()
        for i in range(len(Q)):
            store.search(Q[i:i + 1], 10)
        dt = (time.perf_counter() - t) / len(Q) * 1000
        res = store.search(Q, 10)
        overlap = np.mean([len({a for a, _ in x} & {a for a, _ in y}) / 10 for x, y in zip(res, exact)])
        index_cmp[name] = {"ms_per_query": round(dt, 4), "recall@10_vs_exact": round(float(overlap), 4)}
    qd = QdrantStore(emb.dim)
    qd.add(ids, D)
    sub = Q[:300]
    t = time.perf_counter()
    qres = qd.search(sub, 10)
    dt = (time.perf_counter() - t) / len(sub) * 1000
    overlap = np.mean([len({a for a, _ in x} & {a for a, _ in y}) / 10 for x, y in zip(qres, exact[:300])])
    index_cmp["qdrant (local mode)"] = {"ms_per_query": round(dt, 4), "recall@10_vs_exact": round(float(overlap), 4)}
    print(index_cmp)

    # ---- abstention gate -----------------------------------------------------------
    def sigs(name):
        qs, _, lex, dense = runs[name]
        return matrix([signals(l, d, len(tokenize(q.text))) for q, l, d in zip(qs, lex, dense)])

    Xc = np.vstack([sigs("in_cal"), sigs("oos_cal")])
    yc = np.r_[np.ones(len(in_cal)), np.zeros(len(oos_cal))]
    gate = Gate.fit(Xc, yc, keep_in_scope=0.95)
    p_in = np.array([gate.prob(dict(zip(FEATURES, x)))
                     for x in sigs("in_te")])
    p_oos = np.array([gate.prob(dict(zip(FEATURES, x)))
                      for x in sigs("oos_te")])
    answered_in = p_in >= gate.threshold
    # among in-scope questions the gate lets through, how often is the gold passage in the top 5?
    fused_te = [fuse_ids(l, d, rrf_k, w, 5) for l, d in zip(runs["in_te"][2], runs["in_te"][3])]
    hit5 = np.array([q.gold_doc in f for q, f in zip(in_te, fused_te)])
    gate_res = {
        "auroc_in_vs_out_of_scope": round(auroc(p_in, p_oos), 4),
        "threshold": round(gate.threshold, 4),
        "in_scope_answered": round(float(answered_in.mean()), 4),
        "out_of_scope_abstained": round(float((p_oos < gate.threshold).mean()), 4),
        "recall@5_when_answered": round(float(hit5[answered_in].mean()), 4),
        "recall@5_when_abstained": round(float(hit5[~answered_in].mean()), 4),
        "n_in_scope_test": int(len(p_in)), "n_out_of_scope_test": int(len(p_oos)),
    }
    print(gate_res)

    # ---- answer-bearing context and extractive baseline ----------------------------
    pipe = RAGPipeline(HybridRetriever(ids, texts, emb, store=flat, rrf_k=rrf_k, dense_weight=w),
                       ExtractiveProvider(), gate, k=5)
    ctx_hit, sent_hit, lat = [], [], []
    sample = in_te[::4]
    for q in sample:
        hits, _ = pipe.retrieve(q.text)
        ctx = " ".join(pipe.retriever.texts[h.doc_id] for h in hits)
        ctx_hit.append(any(a in ctx for a in q.answers))
        t = time.perf_counter()
        a = pipe.answer(q.text)
        lat.append((time.perf_counter() - t) * 1000)
        sent_hit.append(bool(a.answer) and any(x in a.answer for x in q.answers))
    e2e = {
        "answer_string_in_top5_context": round(float(np.mean(ctx_hit)), 4),
        "extractive_baseline_sentence_contains_answer": round(float(np.mean(sent_hit)), 4),
        "pipeline_latency_ms_p50": pct(lat, 50), "pipeline_latency_ms_p95": pct(lat, 95),
        "n": len(sample),
    }
    print(e2e)

    cfg = {"embedder": emb.name, "rrf_k": rrf_k, "dense_weight": w, "k": 5,
           "gate": {"mean": gate.mean, "scale": gate.scale, "coef": gate.coef,
                    "intercept": gate.intercept, "threshold": gate.threshold}}
    (RESULTS / "squad_config.json").write_text(json.dumps(cfg, indent=2))
    out = {
        "corpus": {"indexed_passages": len(ids), "in_scope_questions": len(col.in_scope),
                   "out_of_scope_questions": len(col.out_of_scope), "index_build_s": round(build_s, 2)},
        "fusion_grid_calibration": grid, "tuned_fusion": {"rrf_k": rrf_k, "dense_weight": w},
        "retrieval_test": retrieval, "vector_index": index_cmp, "gate_test": gate_res,
        "end_to_end_test": e2e,
        "machine": {"python": platform.python_version(), "cpu": platform.processor() or platform.machine()},
    }
    (RESULTS / "squad_metrics.json").write_text(json.dumps(out, indent=2))
    print("wrote results/squad_metrics.json and results/squad_config.json")


if __name__ == "__main__":
    main()
