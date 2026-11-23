import numpy as np
import pytest

from rag.gate import FEATURES, Gate, matrix, signals
from rag.text import chunk_words, split_sentences, tokenize
from rag.vectorstore import FaissStore, QdrantStore


def test_tokenize_drops_stopwords():
    assert tokenize("What is the role of the BM25 function?") == ["role", "bm25", "function"]


def test_chunking_overlaps_and_covers():
    words = [f"w{i}" for i in range(500)]
    chunks = chunk_words(" ".join(words), max_words=200, overlap=50)
    assert all(len(c.split()) <= 200 for c in chunks)
    assert chunks[0].split()[-50:] == chunks[1].split()[:50]
    assert set(" ".join(chunks).split()) == set(words)
    with pytest.raises(ValueError):
        chunk_words("a b", max_words=10, overlap=10)


def test_sentences():
    assert split_sentences("One here. Two there! 3 is last?") == ["One here.", "Two there!", "3 is last?"]


@pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid"])
def test_each_mode_finds_obvious_document(retriever, mode):
    hits = retriever.search("which nickel superalloy is used in jet engine discs", k=3, mode=mode)
    assert hits[0].doc_id == "inco"


def test_fusion_rewards_agreement(retriever):
    fused = retriever.fuse([("a", 9.0), ("b", 5.0)], [("b", 0.9), ("c", 0.8)], k=3)
    assert fused[0].doc_id == "b"           # ranked by both lists
    assert {h.doc_id for h in fused} == {"a", "b", "c"}


def test_qdrant_matches_faiss_flat(embedder):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 16)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    ids = [f"d{i}" for i in range(200)]
    f, q = FaissStore(16), QdrantStore(16)
    f.add(ids, X)
    q.add(ids, X)
    a, b = f.search(X[:20], 5), q.search(X[:20], 5)
    assert [[d for d, _ in r] for r in a] == [[d for d, _ in r] for r in b]


def test_hnsw_close_to_exact():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(2000, 32)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    ids = [str(i) for i in range(2000)]
    flat, hnsw = FaissStore(32), FaissStore(32, "hnsw", ef_search=128)
    flat.add(ids, X)
    hnsw.add(ids, X)
    ex, ap = flat.search(X[:100], 10), hnsw.search(X[:100], 10)
    overlap = np.mean([len({d for d, _ in x} & {d for d, _ in y}) / 10 for x, y in zip(ex, ap)])
    assert overlap > 0.9


def test_gate_separates_and_roundtrips(tmp_path):
    good = signals([("a", 30.0), ("b", 10.0)], [("a", 0.8), ("c", 0.5)], 3)
    bad = signals([("x", 4.0), ("y", 3.9)], [("z", 0.3), ("w", 0.29)], 3)
    rng = np.random.default_rng(0)
    X = np.vstack([matrix([good]) + rng.normal(0, .05, (50, len(FEATURES))),
                   matrix([bad]) + rng.normal(0, .05, (50, len(FEATURES)))])
    y = np.r_[np.ones(50), np.zeros(50)]
    gate = Gate.fit(X, y, keep_in_scope=0.95)
    assert gate.allow(good)[0] and not gate.allow(bad)[0]
    gate.save(tmp_path / "g.json")
    assert Gate.load(tmp_path / "g.json").prob(good) == pytest.approx(gate.prob(good))
