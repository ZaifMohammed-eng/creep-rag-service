"""End-to-end answer evaluation on the SQuAD benchmark with a chosen LLM provider.

Scores SQuAD exact match / F1 on answered in-scope test questions, abstention on
out-of-scope ones, citation validity, tokens, cost and latency.

    python eval/run_generation_eval.py --provider extractive --n 300          # offline baseline
    ANTHROPIC_API_KEY=... python eval/run_generation_eval.py --provider anthropic --n 300
    OPENAI_API_KEY=...    python eval/run_generation_eval.py --provider openai --n 300

Writes results/generation_<provider>.json.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
import string
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag import corpus  # noqa: E402
from rag.embeddings import get_embedder  # noqa: E402
from rag.gate import Gate  # noqa: E402
from rag.llm import get_provider  # noqa: E402
from rag.pipeline import RAGPipeline  # noqa: E402
from rag.retriever import HybridRetriever  # noqa: E402


def normalize(s: str) -> str:  # official SQuAD v1.1 normalisation
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1(pred: str, gold: str) -> float:
    p, g = normalize(pred).split(), normalize(gold).split()
    common = collections.Counter(p) & collections.Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="extractive", choices=["extractive", "anthropic", "openai"])
    ap.add_argument("--n", type=int, default=300, help="in-scope questions; out-of-scope uses n/3")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = json.loads((ROOT / "results" / "squad_config.json").read_text())
    col = corpus.load()
    retr = HybridRetriever([d.doc_id for d in col.docs], [f"{d.title}. {d.text}" for d in col.docs],
                           get_embedder("wordllama"), rrf_k=cfg["rrf_k"], dense_weight=cfg["dense_weight"])
    pipe = RAGPipeline(retr, get_provider(args.provider), Gate(**cfg["gate"]), k=cfg["k"])

    rng = random.Random(args.seed)
    in_te, oos_te = col.queries("test")
    in_s, oos_s = rng.sample(in_te, args.n), rng.sample(oos_te, args.n // 3)

    rows = []
    for q, scope in [(q, "in") for q in in_s] + [(q, "out") for q in oos_s]:
        a = pipe.answer(q.text)
        row = {"qid": q.qid, "scope": scope, "abstained": a.abstained, "reason": a.reason,
               "latency_ms": a.timings_ms.get("total"), **{k: a.usage.get(k) for k in
                                                           ("input_tokens", "output_tokens", "cost_usd")}}
        if scope == "in" and a.answer:
            row["em"] = max(float(normalize(a.answer) == normalize(g)) for g in q.answers)
            row["f1"] = max(f1(a.answer, g) for g in q.answers)
            row["cited_gold"] = q.gold_doc in a.citations
        rows.append(row)

    ins = [r for r in rows if r["scope"] == "in"]
    outs = [r for r in rows if r["scope"] == "out"]
    ans = [r for r in ins if not r["abstained"]]
    gen = [r for r in rows if r["input_tokens"]]
    costs = [r["cost_usd"] for r in gen if r["cost_usd"] is not None]
    summary = {
        "provider": pipe.provider.name, "n_in_scope": len(ins), "n_out_of_scope": len(outs),
        "in_scope_answered": round(len(ans) / len(ins), 4),
        "exact_match_when_answered": round(float(np.mean([r["em"] for r in ans])), 4) if ans else None,
        "f1_when_answered": round(float(np.mean([r["f1"] for r in ans])), 4) if ans else None,
        "cites_gold_passage_when_answered": round(float(np.mean([r["cited_gold"] for r in ans])), 4) if ans else None,
        "out_of_scope_abstained": round(float(np.mean([r["abstained"] for r in outs])), 4),
        "abstain_reasons": dict(collections.Counter(r["reason"] for r in rows)),
        "latency_ms_p50": round(float(np.percentile([r["latency_ms"] for r in rows], 50)), 2),
        "latency_ms_p95": round(float(np.percentile([r["latency_ms"] for r in rows], 95)), 2),
        "mean_input_tokens": round(float(np.mean([r["input_tokens"] for r in gen])), 1) if gen else 0,
        "mean_output_tokens": round(float(np.mean([r["output_tokens"] for r in gen])), 1) if gen else 0,
        "cost_usd_per_1k_queries": round(1000 * float(np.mean(costs)), 4) if costs else None,
    }
    name = args.provider
    (ROOT / "results" / f"generation_{name}.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
