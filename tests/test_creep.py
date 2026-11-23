import pytest

from rag.creep import answer_matches, answer_matches_any, build, load_df
from rag.llm import ExtractiveProvider
from rag.metadata import MetadataFilter
from rag.pipeline import RAGPipeline
from rag.retriever import Hit, HybridRetriever


@pytest.fixture(scope="module")
def corpus():
    return build()


@pytest.fixture(scope="module")
def mf(corpus):
    return MetadataFilter({d.doc_id: dict(d.meta) for d in corpus.docs if d.meta},
                          {d.doc_id: d.alloy for d in corpus.docs if d.alloy})


def test_corpus_shape_and_holdout(corpus):
    df = load_df()
    held = set(corpus.held_out_alloys)
    assert len(held) == 6
    assert not any(d.alloy in held for d in corpus.docs)
    assert sum(d.kind == "record" for d in corpus.docs) == int((~df.alloy.isin(held)).sum())
    assert sum(d.kind == "sheet" for d in corpus.docs) == df.alloy.nunique() - 6
    assert all(not q.gold_docs for q in corpus.questions if not q.in_scope)
    ids = {d.doc_id for d in corpus.docs}
    assert all(q.gold_docs <= ids for q in corpus.questions)


def test_alloy_split_is_by_alloy(corpus):
    split_of = {}
    for q in corpus.questions:
        if q.alloy:
            assert split_of.setdefault(q.alloy, q.split) == q.split


def test_parser_units_and_names(mf):
    p = mf.parse("How long did Inconel 713C last at 600 MPa and 1000 K?")
    assert p.entity == "Inconel 713C" and p.mpa == 600 and p.T == 1000      # "713C" is not 713 deg C
    p = mf.parse("Rupture life of Nimonic 80A at 760 °C and 300 MPa?")
    assert p.entity == "Nimonic 80A" and p.T == pytest.approx(1033.15)
    assert mf.parse("How much chromium does Inconel 713C contain?").T is None
    assert mf.parse("What is the median rupture time?").entity is None


def test_rerank_puts_nearest_record_first(corpus, mf):
    df = load_df()
    r = df[df.alloy == "Waspaloy"].iloc[0]
    q = f"Roughly how long does Waspaloy last at about {round(r.mpa, -1):.0f} MPa and {r.degc} °C?"
    top = mf.rerank(q, [Hit("sheet-waspaloy", 1.0)], 3)
    m = mf.meta[top[0].doc_id]
    assert abs(m["mpa"] - r.mpa) <= 10 and abs(m["T"] - r["T"]) <= 10
    assert mf.match_score(q) > 0.3


def test_answer_matching():
    assert answer_matches("ruptured after 1,557.9 hours", (1557.9,))
    assert not answer_matches("ruptured after 1,500.0 hours", (1557.9,))
    assert answer_matches_any("after 12.0 hours", (11.0, 12.0))
    assert answer_matches("tested from 800.00 K to 920.00 K", (800.0, 920.0))


def test_end_to_end_lookup_and_entity_rule(corpus, mf, embedder):
    docs = corpus.docs
    retr = HybridRetriever([d.doc_id for d in docs], [d.text for d in docs], embedder,
                           titles=[d.title for d in docs], rrf_k=5, dense_weight=0.0)
    pipe = RAGPipeline(retr, ExtractiveProvider(), None, k=5, reranker=mf)
    a = pipe.answer("How much chromium does Waspaloy contain?")
    assert a.citations == ["sheet-waspaloy"] and "Cr (chromium)" in a.answer
    held = corpus.held_out_alloys[0]
    assert mf.entity(f"How much chromium does {held} contain?") is None   # held-out alloys are not catalogued
