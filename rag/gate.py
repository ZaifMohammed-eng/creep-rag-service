"""Abstention gate: decide from retrieval signals alone whether the index can answer.

The gate never sees the answer. It looks at how strongly and how consistently the two
retrievers agree, and returns P(the question is covered by the indexed documents).
If that probability is under the threshold, the service abstains instead of asking the
LLM to write an answer from weak context, which is where most hallucinated answers in a
RAG system come from.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

FEATURES = ("bm25_top1_per_term", "bm25_margin", "dense_top1", "dense_margin", "top5_agreement",
            "same_top1", "title_match", "filter_match")


def title_match(query_tokens: list[str], title_tokens: list[str]) -> float:
    """Share of the top document's title tokens (e.g. its alloy name) that the query mentions.
    A question naming an entity the index does not hold scores low even when numbers match.
    Documents with no entity (empty title) return 1.0: there is nothing to contradict."""
    if not title_tokens:
        return 1.0
    q = set(query_tokens)
    return sum(t in q for t in title_tokens) / len(title_tokens)


def signals(lex: list[tuple[str, float]], dense: list[tuple[str, float]], n_terms: int,
            title_overlap: float = 1.0, filter_match: float | None = None) -> dict[str, float]:
    b = [s for _, s in lex[:2]] + [0.0, 0.0]
    d = [s for _, s in dense[:2]] + [0.0, 0.0]
    top5_l = {x for x, _ in lex[:5]}
    top5_d = {x for x, _ in dense[:5]}
    return {
        "bm25_top1_per_term": b[0] / max(1, n_terms),
        "bm25_margin": b[0] - b[1],
        "dense_top1": d[0],
        "dense_margin": d[0] - d[1],
        "top5_agreement": len(top5_l & top5_d) / 5.0,
        "same_top1": float(bool(lex) and bool(dense) and lex[0][0] == dense[0][0]),
        "title_match": title_overlap,
        # closeness of the best record to the parsed entity + conditions; title match when not applicable
        "filter_match": title_overlap if filter_match is None else filter_match,
    }


@dataclass
class Gate:
    mean: list[float]
    scale: list[float]
    coef: list[float]
    intercept: float
    threshold: float

    def prob(self, sig: dict[str, float]) -> float:
        x = (np.array([sig[f] for f in FEATURES]) - self.mean) / self.scale
        return float(1.0 / (1.0 + np.exp(-(x @ np.array(self.coef) + self.intercept))))

    def allow(self, sig: dict[str, float]) -> tuple[bool, float]:
        p = self.prob(sig)
        return p >= self.threshold, p

    @classmethod
    def fit(cls, X: np.ndarray, y: np.ndarray, keep_in_scope: float = 0.95) -> "Gate":
        """Fit logistic regression; set the threshold so `keep_in_scope` of the
        calibration in-scope questions are still answered."""
        from sklearn.linear_model import LogisticRegression

        mean, scale = X.mean(0), X.std(0) + 1e-9
        lr = LogisticRegression(C=1.0, max_iter=1000).fit((X - mean) / scale, y)
        p = lr.predict_proba((X - mean) / scale)[:, 1]
        thr = float(np.quantile(p[y == 1], 1.0 - keep_in_scope))
        return cls(mean.tolist(), scale.tolist(), lr.coef_[0].tolist(), float(lr.intercept_[0]), thr)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> "Gate":
        return cls(**json.loads(path.read_text()))


def matrix(sigs: list[dict[str, float]]) -> np.ndarray:
    return np.array([[s[f] for f in FEATURES] for s in sigs], dtype=float)
