"""Vector stores behind one interface: FAISS (flat or HNSW) and Qdrant.

Vectors are L2-normalised, so inner product equals cosine similarity.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np


class VectorStore(Protocol):
    name: str

    def add(self, ids: list[str], vectors: np.ndarray) -> None: ...

    def search(self, queries: np.ndarray, k: int) -> list[list[tuple[str, float]]]: ...


class FaissStore:
    def __init__(self, dim: int, kind: str = "flat", hnsw_m: int = 32, ef_search: int = 64):
        import faiss

        if kind == "flat":
            self.index = faiss.IndexFlatIP(dim)
        elif kind == "hnsw":
            self.index = faiss.IndexHNSWFlat(dim, hnsw_m, faiss.METRIC_INNER_PRODUCT)
            self.index.hnsw.efSearch = ef_search
        else:
            raise ValueError(kind)
        self.name = f"faiss-{kind}"
        self._ids: list[str] = []

    def add(self, ids: list[str], vectors: np.ndarray) -> None:
        self.index.add(np.ascontiguousarray(vectors, dtype=np.float32))
        self._ids.extend(ids)

    def search(self, queries: np.ndarray, k: int) -> list[list[tuple[str, float]]]:
        k = min(k, len(self._ids))
        scores, idx = self.index.search(np.ascontiguousarray(queries, dtype=np.float32), k)
        return [[(self._ids[j], float(s)) for j, s in zip(row_i, row_s) if j >= 0]
                for row_i, row_s in zip(idx, scores)]

    def __len__(self) -> int:
        return len(self._ids)


class QdrantStore:
    """Qdrant in local mode (in-process, no server). Point `url` at a server in production."""

    def __init__(self, dim: int, collection: str = "docs", url: str | None = None):
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        self.client = QdrantClient(url=url) if url else QdrantClient(location=":memory:")
        self.collection = collection
        if self.client.collection_exists(collection):
            self.client.delete_collection(collection)
        self.client.create_collection(collection, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))
        self.name = "qdrant"
        self._n = 0

    def add(self, ids: list[str], vectors: np.ndarray) -> None:
        from qdrant_client.models import PointStruct

        points = [PointStruct(id=self._n + i, vector=v.tolist(), payload={"doc_id": d})
                  for i, (d, v) in enumerate(zip(ids, vectors))]
        for i in range(0, len(points), 512):
            self.client.upsert(self.collection, points=points[i:i + 512])
        self._n += len(points)

    def search(self, queries: np.ndarray, k: int) -> list[list[tuple[str, float]]]:
        out = []
        for q in queries:
            res = self.client.query_points(self.collection, query=q.tolist(), limit=k).points
            out.append([(p.payload["doc_id"], float(p.score)) for p in res])
        return out

    def __len__(self) -> int:
        return self._n
