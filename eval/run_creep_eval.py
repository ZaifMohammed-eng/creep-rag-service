"""Evaluate RAG on the creep corpus.

Fusion weights and the abstention gate are fit on calibration alloys only; every
number reported is from test alloys and held-out alloys. Writes results/creep_metrics.json
and results/creep_config.json (the config the API loads for RAG_CORPUS=creep).

    python eval/run_creep_eval.py
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.creep import CONFLICT_QUESTION, answer_matches, answer_matches_any, build  # noqa: E402
from rag.embeddings import get_embedder  # noqa: E402
from rag.gate import FEATURES, Gate, matrix, signals, title_match  # noqa: E402
from rag.llm import ExtractiveProvider  # noqa: E402
from rag.metadata import MetadataFilter  # noqa: E402
from rag.retriever import Hit  # noqa: E402
from rag.pipeline import RAGPipeline  # noqa: E402
from rag.retriever import HybridRetriever  # noqa: E402
from rag.text import tokenize  # noqa: E402

RESULTS = ROOT / "results"
DEPTH = 50
LOOKUP_TYPES = {"test_lookup", "approx_lookup", "composition", "temp_range", "descriptor", "method"}


def fuse_ids(lex, dense, rrf_k, w, k=10):
    f: dict[str, float] = {}
    for r, (d, _) in enumerate(lex):
        f[d] = f.get(d, 0) + 1 / (rrf_k + r + 1)
    for r, (d, _) in enumerate(dense):
        f[d] = f.get(d, 0) + w / (rrf_k + r + 1)
    return sorted(f, key=lambda d: -f[d])[:k]


def rank_metrics(ranked: list[list[str]], gold: list[frozenset]) -> dict:
    first = []
    for r, g in zip(ranked, gold):
        pos = [i + 1 for i, d in enumerate(r) if d in g]
        first.append(pos[0] if pos else None)
    rec = lambda k: float(np.mean([x is not None and x <= k for x in first]))  # noqa: E731
    return {"recall@1": round(rec(1), 4), "recall@5": round(rec(5), 4), "recall@10": round(rec(10), 4),
            "mrr@10": round(float(np.mean([1 / x if x and x <= 10 else 0 for x in first])), 4), "n": len(gold)}


def auroc(pos, neg) -> float:
    s = np.concatenate([pos, neg])
    ranks = s.argsort().argsort() + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    c = build()
    ids, texts, titles = [d.doc_id for d in c.docs], [d.text for d in c.docs], [d.title for d in c.docs]
    emb = get_embedder("wordllama")
    t = time.perf_counter()
    retr = HybridRetriever(ids, texts, emb, titles=titles)
    build_s = time.perf_counter() - t
    print(f"{len(ids)} docs indexed in {build_s:.1f}s; held-out alloys: {c.held_out_alloys}")

    qs = [q for q in c.questions if q.qtype != "aggregation"]
    Q = emb.embed([q.text for q in qs])
    lex = {q.qid: retr.bm25.search(q.text, DEPTH) for q in qs}
    dense = {q.qid: d for q, d in zip(qs, retr.store.search(Q, DEPTH))}

    # ---- fusion tuning on calibration alloys ----------------------------------------
    cal = [q for q in qs if q.split == "calib" and q.in_scope]
    grid = []
    for rrf_k in (5, 10, 30, 60):
        for w in (0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0):
            m = rank_metrics([fuse_ids(lex[q.qid], dense[q.qid], rrf_k, w) for q in cal], [q.gold_docs for q in cal])
            grid.append({"rrf_k": rrf_k, "dense_weight": w, **m})
    best = max(grid, key=lambda g: g["mrr@10"])
    rrf_k, w = best["rrf_k"], best["dense_weight"]
    print("tuned fusion:", rrf_k, w, best["mrr@10"])

    mf = MetadataFilter({d.doc_id: dict(d.meta) for d in c.docs if d.meta},
                        {d.doc_id: d.alloy for d in c.docs if d.alloy})

    def final(q, k=10):
        fused = fuse_ids(lex[q.qid], dense[q.qid], rrf_k, w, DEPTH)
        return [h.doc_id for h in mf.rerank(q.text, [Hit(d, 0.0) for d in fused], k)]

    # ---- retrieval on test alloys, overall and by question type ---------------------
    te = [q for q in qs if q.split == "test" and q.in_scope]
    runs = {
        "bm25": lambda q: [d for d, _ in lex[q.qid][:10]],
        "dense (wordllama)": lambda q: [d for d, _ in dense[q.qid][:10]],
        "hybrid RRF (k=60, w=1)": lambda q: fuse_ids(lex[q.qid], dense[q.qid], 60, 1.0),
        f"hybrid RRF tuned (k={rrf_k}, w={w})": lambda q: fuse_ids(lex[q.qid], dense[q.qid], rrf_k, w),
        "tuned hybrid + metadata filter": final,
    }
    retrieval = {}
    for name, fn in runs.items():
        by = {"all": rank_metrics([fn(q) for q in te], [q.gold_docs for q in te])}
        for qt in sorted(LOOKUP_TYPES):
            sub = [q for q in te if q.qtype == qt]
            if sub:
                by[qt] = rank_metrics([fn(q) for q in sub], [q.gold_docs for q in sub])
        retrieval[name] = by
        print(f"{name:32s} approx R@1 {by['approx_lookup']['recall@1']:.3f} |", f"R@1 {by['all']['recall@1']:.3f} R@5 {by['all']['recall@5']:.3f} MRR {by['all']['mrr@10']:.3f}")

    # ---- abstention gate: in-scope vs held-out-alloy questions ----------------------
    def feats(q):
        top = final(q, 1)
        tm = title_match(tokenize(q.text, drop_stop=False),
                         tokenize(retr.titles.get(top[0], ""), drop_stop=False)) if top else 0.0
        return signals(lex[q.qid], dense[q.qid], len(tokenize(q.text)), tm, mf.match_score(q.text))

    def group(split, ins):
        return [q for q in qs if q.split == split and q.in_scope == ins]

    Xc = np.vstack([matrix([feats(q) for q in group("calib", True)]), matrix([feats(q) for q in group("calib", False)])])
    yc = np.r_[np.ones(len(group("calib", True))), np.zeros(len(group("calib", False)))]
    gate = Gate.fit(Xc, yc, keep_in_scope=0.95)
    # serving rule: a question naming an indexed alloy skips the learned gate (probability 1)
    def served_prob(q):
        return 1.0 if mf.entity(q.text) is not None else gate.prob(feats(q))
    p_in = np.array([served_prob(q) for q in group("test", True)])
    p_out = np.array([served_prob(q) for q in group("test", False)])
    learned_only = {"auroc": round(auroc(np.array([gate.prob(feats(q)) for q in group("test", True)]),
                                         np.array([gate.prob(feats(q)) for q in group("test", False)])), 4)}
    false_entity = [q.text for q in group("test", False) if mf.entity(q.text) is not None]
    gate_res = {"auroc": round(auroc(p_in, p_out), 4), "threshold": round(gate.threshold, 4),
                "in_scope_answered": round(float((p_in >= gate.threshold).mean()), 4),
                "held_out_alloy_questions_declined": round(float((p_out < gate.threshold).mean()), 4),
                "n_in_scope": len(p_in), "n_held_out": len(p_out), "learned_gate_alone": learned_only,
                "held_out_questions_matched_to_an_indexed_alloy": len(false_entity),
                "coefficients": dict(zip(FEATURES, [round(x, 3) for x in gate.coef]))}
    print("gate", gate_res)

    # ---- end to end with the offline extractive reader ------------------------------
    pipe = RAGPipeline(HybridRetriever(ids, texts, emb, store=retr.store, rrf_k=rrf_k, dense_weight=w,
                                       titles=titles), ExtractiveProvider(), gate, k=5, reranker=mf)
    e2e = defaultdict(lambda: {"n": 0, "answered": 0, "correct": 0})
    lat = []
    for q in te + group("test", False) + [q for q in c.questions if q.qtype == "aggregation"]:
        t0 = time.perf_counter()
        a = pipe.answer(q.text)
        lat.append((time.perf_counter() - t0) * 1000)
        key = q.qtype if q.in_scope else "held_out_alloy"
        e = e2e[key]
        e["n"] += 1
        e["answered"] += not a.abstained
        if q.in_scope and not a.abstained:
            ok = (answer_matches_any(a.answer, q.answers) if q.qtype in ("test_lookup", "approx_lookup")
                  else answer_matches(a.answer, q.answers))
            e["correct"] += ok
    e2e = {k: {**v, "answered_rate": round(v["answered"] / v["n"], 3),
               "correct_when_answered": round(v["correct"] / v["answered"], 3) if v["answered"] else None}
           for k, v in e2e.items()}
    for k, v in e2e.items():
        print(f"{k:16s} n={v['n']:4d} answered {v['answered_rate']:.3f} correct|answered {v['correct_when_answered']}")

    # ---- conflicting sources --------------------------------------------------------
    text, gold = CONFLICT_QUESTION
    hits, _ = pipe.retrieve(text)
    conflict = {"question": text, "top5": [h.doc_id for h in hits],
                "conflicting_notes_in_top5": sorted(set(h.doc_id for h in hits) & gold), "notes_in_corpus": sorted(gold)}
    print("conflict", conflict)

    cfg = {"corpus": "creep", "embedder": emb.name, "rrf_k": rrf_k, "dense_weight": w, "k": 5,
           "gate": {"mean": gate.mean, "scale": gate.scale, "coef": gate.coef, "intercept": gate.intercept,
                    "threshold": gate.threshold}}
    (RESULTS / "creep_config.json").write_text(json.dumps(cfg, indent=2))
    out = {"corpus": {"documents": len(ids), "by_kind": {k: sum(d.kind == k for d in c.docs)
                                                         for k in ("record", "sheet", "note")},
                      "held_out_alloys": list(c.held_out_alloys), "questions": len(c.questions),
                      "index_build_s": round(build_s, 2)},
           "tuned_fusion": {"rrf_k": rrf_k, "dense_weight": w}, "fusion_grid_calibration": grid,
           "retrieval_test": retrieval, "gate_test": gate_res, "end_to_end_test": e2e,
           "pipeline_latency_ms": {"p50": round(float(np.percentile(lat, 50)), 2),
                                   "p95": round(float(np.percentile(lat, 95)), 2)},
           "conflicting_sources": conflict}
    (RESULTS / "creep_metrics.json").write_text(json.dumps(out, indent=2))
    print("wrote results/creep_metrics.json and results/creep_config.json")


if __name__ == "__main__":
    main()
