"""Metadata filter for record documents (the "self-query" pattern).

Text retrieval cannot tell that 550 MPa is close to 552 MPa: to BM25 they are different
tokens, and small embedding models represent numbers poorly. So the question is parsed for
an entity (alloy name from the index's own catalogue), a stress and a temperature, and the
records of that entity are re-ranked by numeric distance. Questions with no parsed numbers
pass through unchanged.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .retriever import Hit

_MPA = re.compile(r"(\d+(?:\.\d+)?)\s*mpa", re.I)
_K = re.compile(r"(\d+(?:\.\d+)?)\s*k\b", re.I)
_C = re.compile(r"(\d+(?:\.\d+)?)\s*(?:°\s*c|deg(?:rees)?\s*c|celsius)\b", re.I)   # not "713C" in a name


@dataclass
class Parsed:
    entity: str | None
    mpa: float | None
    T: float | None


def _pattern(name: str) -> re.Pattern:
    chunks = re.findall(r"[a-z0-9]+", name.lower())
    return re.compile(r"(?<![a-z0-9])" + r"[\s\-.]*".join(map(re.escape, chunks)) + r"(?![a-z0-9])")


class MetadataFilter:
    def __init__(self, doc_meta: dict[str, dict], doc_entity: dict[str, str]):
        self.meta = doc_meta                       # doc_id -> {"mpa": .., "T": ..}
        self.by_entity: dict[str, list[str]] = {}
        for d, e in doc_entity.items():
            if d in doc_meta:
                self.by_entity.setdefault(e, []).append(d)
        # longest names first so "Nimonic 80A" wins over "Nimonic 80"
        self.patterns = sorted(((e, _pattern(e)) for e in self.by_entity), key=lambda x: -len(x[0]))

    def parse(self, q: str) -> Parsed:
        ql = q.lower()
        exact = [e for e in self.by_entity if e.lower() in ql]
        entity = max(exact, key=len) if exact else next((e for e, p in self.patterns if p.search(ql)), None)
        if entity is not None:                       # numbers inside the name are not conditions
            q = dict(self.patterns)[entity].sub(" ", ql)
        mpa = _MPA.search(q)
        k, c = _K.search(q), _C.search(q)
        T = float(k.group(1)) if k else (float(c.group(1)) + 273.15 if c else None)
        return Parsed(entity, float(mpa.group(1)) if mpa else None, T)

    def entity(self, query: str) -> str | None:
        """The catalogued entity the question names, if any."""
        return self.parse(query).entity

    def match_score(self, query: str) -> float | None:
        """1 / (1 + distance to the closest record of the named entity), or None when the
        question names no catalogued entity with a stress or temperature."""
        p = self.parse(query)
        if p.entity is None or (p.mpa is None and p.T is None):
            return None
        return 1.0 / (1.0 + min(self._dist(d, p) for d in self.by_entity[p.entity]))

    def _dist(self, d: str, p: Parsed) -> float:
        m = self.meta[d]
        ds = abs(m["mpa"] - p.mpa) / max(10.0, 0.03 * p.mpa) if p.mpa is not None else 0.0
        dt = abs(m["T"] - p.T) / 10.0 if p.T is not None else 0.0
        return ds + dt

    def rerank(self, query: str, hits: list[Hit], k: int) -> list[Hit]:
        p = self.parse(query)
        if p.entity is None or (p.mpa is None and p.T is None):
            return hits[:k]

        def dist(d: str) -> float:
            return self._dist(d, p)

        ranked = sorted(self.by_entity[p.entity], key=dist)
        top = [Hit(d, 1.0 / (1.0 + dist(d))) for d in ranked[:k]]
        seen = {h.doc_id for h in top}
        return (top + [h for h in hits if h.doc_id not in seen])[:k]
