"""Embedding models behind one interface.

`WordLlamaEmbedder` is the default: 256-d embeddings built from token embeddings
distilled from an LLM (WordLlama, l2_supercat), with the weights shipped inside the
pip package, so it runs offline on CPU with no model download. It is much weaker
than a trained sentence encoder, which is why the hybrid retriever keeps BM25 in
the loop. `SpacyEmbedder` (averaged GloVe-style word vectors) is kept as a weaker
baseline. `SentenceTransformerEmbedder` and `OpenAIEmbedder` are drop-in
replacements when those models are reachable.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Protocol

import numpy as np

from .text import STOPWORDS


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


def _normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (x / n).astype(np.float32)


class SpacyEmbedder:
    def __init__(self, model: str = "en_core_web_md"):
        import spacy

        self._nlp = spacy.load(model, exclude=["parser", "ner", "tagger", "lemmatizer",
                                               "attribute_ruler", "senter", "tok2vec"])
        self.name = f"spacy:{model}"
        self.dim = int(self._nlp.vocab.vectors.shape[1])

    def _one(self, doc) -> np.ndarray:
        vecs = [t.vector for t in doc
                if t.has_vector and not t.is_punct and t.lower_ not in STOPWORDS]
        return np.mean(vecs, axis=0) if vecs else np.zeros(self.dim, np.float32)

    def embed(self, texts: list[str]) -> np.ndarray:
        docs = self._nlp.tokenizer.pipe(texts, batch_size=256)
        return _normalize(np.vstack([self._one(d) for d in docs]))


class WordLlamaEmbedder:
    def __init__(self, dim: int = 256):
        import wordllama
        from wordllama import WordLlama

        # load the weights bundled in the package; never reach for the network
        pkg = os.path.dirname(wordllama.__file__)
        self._wl = WordLlama.load(cache_dir=pkg, disable_download=True, dim=dim)
        self.name, self.dim = f"wordllama:l2_supercat_{dim}", dim

    def embed(self, texts: list[str]) -> np.ndarray:
        return _normalize(np.asarray(self._wl.embed(texts, batch_size=64), dtype=np.float32))


class SentenceTransformerEmbedder:  # pragma: no cover - needs model download
    def __init__(self, model: str = "BAAI/bge-small-en-v1.5"):
        from sentence_transformers import SentenceTransformer

        self._m = SentenceTransformer(model)
        self.name = f"st:{model}"
        self.dim = self._m.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        return _normalize(self._m.encode(texts, batch_size=64, normalize_embeddings=True))


class OpenAIEmbedder:  # pragma: no cover - needs API key
    def __init__(self, model: str = "text-embedding-3-small", dim: int = 1536):
        import httpx

        self._client = httpx.Client(
            base_url="https://api.openai.com/v1",
            headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}, timeout=60)
        self.name, self.model, self.dim = f"openai:{model}", model, dim

    def embed(self, texts: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), 256):
            r = self._client.post("/embeddings", json={"model": self.model, "input": texts[i:i + 256]})
            r.raise_for_status()
            out.extend(d["embedding"] for d in r.json()["data"])
        return _normalize(np.asarray(out, dtype=np.float32))


@lru_cache(maxsize=4)
def get_embedder(kind: str = "wordllama") -> Embedder:
    if kind == "wordllama":
        return WordLlamaEmbedder()
    if kind == "spacy":
        return SpacyEmbedder()
    if kind.startswith("st:"):
        return SentenceTransformerEmbedder(kind[3:])
    if kind.startswith("openai:"):
        return OpenAIEmbedder(kind[7:])
    raise ValueError(f"unknown embedder {kind!r}")
