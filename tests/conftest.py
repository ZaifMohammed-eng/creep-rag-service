import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DOCS = {
    "inco": "Inconel 718. Inconel 718 is a nickel-based superalloy strengthened by gamma double prime precipitates and used in jet engine discs.",
    "liion": "Lithium-ion cells. Lithium plating on graphite anodes during fast charging at low temperature causes capacity fade.",
    "faiss": "FAISS. FAISS is a library for efficient similarity search of dense vectors, with flat and HNSW indexes.",
    "bm25": "BM25. Okapi BM25 is a ranking function that scores documents by term frequency and inverse document frequency.",
    "rust": "Rust language. Rust is a systems programming language focused on memory safety without a garbage collector.",
}


@pytest.fixture(scope="session")
def embedder():
    from rag.embeddings import get_embedder

    return get_embedder("wordllama")


@pytest.fixture()
def retriever(embedder):
    from rag.retriever import HybridRetriever

    return HybridRetriever(list(DOCS), list(DOCS.values()), embedder, depth=5)
