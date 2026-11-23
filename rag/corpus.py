"""Load SQuAD v1.1 dev as a document collection plus labelled queries.

Each paragraph becomes one document. Each question is labelled with the id of the
paragraph it was written from, which gives a gold target for retrieval.

Articles are split, not questions, so nothing about a held-out article leaks into
calibration:
  * indexed articles  -> paragraphs go into the index; questions are "in scope"
  * held-out articles -> paragraphs are never indexed; questions are "out of scope"
Both groups are split again by article into a calibration half and a test half.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data" / "raw" / "dev-v1.1.json"


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str


@dataclass(frozen=True)
class Query:
    qid: str
    text: str
    gold_doc: str | None          # None for out-of-scope questions
    answers: tuple[str, ...]
    article: str
    split: str                    # "calib" or "test"


@dataclass
class Collection:
    docs: list[Document]
    in_scope: list[Query] = field(default_factory=list)
    out_of_scope: list[Query] = field(default_factory=list)

    def queries(self, split: str) -> tuple[list[Query], list[Query]]:
        return ([q for q in self.in_scope if q.split == split],
                [q for q in self.out_of_scope if q.split == split])


def load(path: Path = DATA, n_holdout: int = 8, seed: int = 0) -> Collection:
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run `python scripts/get_data.py`")
    articles = json.loads(path.read_text())["data"]
    rng = random.Random(seed)
    order = list(range(len(articles)))
    rng.shuffle(order)
    holdout = set(order[:n_holdout])

    # alternate calib/test assignment by article within each group
    split_of: dict[int, str] = {}
    for group in ([i for i in order if i in holdout], [i for i in order if i not in holdout]):
        for k, i in enumerate(group):
            split_of[i] = "calib" if k % 2 == 0 else "test"

    col = Collection(docs=[])
    for ai, art in enumerate(articles):
        title = art["title"].replace("_", " ")
        for pi, para in enumerate(art["paragraphs"]):
            doc_id = f"a{ai:02d}p{pi:03d}"
            if ai not in holdout:
                col.docs.append(Document(doc_id, title, para["context"]))
            for qa in para["qas"]:
                q = Query(
                    qid=qa["id"],
                    text=qa["question"].strip(),
                    gold_doc=None if ai in holdout else doc_id,
                    answers=tuple(dict.fromkeys(a["text"] for a in qa["answers"])),
                    article=title,
                    split=split_of[ai],
                )
                (col.out_of_scope if ai in holdout else col.in_scope).append(q)
    return col
