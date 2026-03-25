"""FAISS-based similarity search for geometry retrieval."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single search result."""
    index: int
    distance: float
    score: float  # similarity score (higher = more similar)
    metadata: dict[str, Any] | None = None


class FAISSIndex:
    """FAISS-based vector index for fast similarity search.

    Supports multiple index types for different speed/accuracy tradeoffs.
    """

    def __init__(
        self,
        dim: int,
        index_type: str = "Flat",
        metric: str = "cosine",
        nlist: int = 100,
    ):
        if not HAS_FAISS:
            raise ImportError("faiss required. Install with: pip install faiss-cpu")

        self.dim = dim
        self.index_type = index_type
        self.metric = metric

        # Build index
        if metric == "cosine":
            # For cosine similarity, normalize vectors and use inner product
            if index_type == "Flat":
                self.index = faiss.IndexFlatIP(dim)
            elif index_type == "IVFFlat":
                quantizer = faiss.IndexFlatIP(dim)
                self.index = faiss.IndexIVFFlat(quantizer, dim, nlist)
            elif index_type == "IVFPQ":
                quantizer = faiss.IndexFlatIP(dim)
                self.index = faiss.IndexIVFPQ(quantizer, dim, nlist, 16, 8)
            else:
                raise ValueError(f"Unknown index type: {index_type}")
        else:  # L2
            if index_type == "Flat":
                self.index = faiss.IndexFlatL2(dim)
            elif index_type == "IVFFlat":
                quantizer = faiss.IndexFlatL2(dim)
                self.index = faiss.IndexIVFFlat(quantizer, dim, nlist)
            else:
                raise ValueError(f"Unknown index type: {index_type}")

        self.is_trained = index_type == "Flat"

    def train(self, vectors: np.ndarray) -> None:
        """Train the index (required for IVF-based indices)."""
        if self.metric == "cosine":
            vectors = self._normalize(vectors)
        if not self.index.is_trained:
            self.index.train(vectors.astype(np.float32))
            self.is_trained = True

    def add(self, vectors: np.ndarray) -> None:
        """Add vectors to the index."""
        if self.metric == "cosine":
            vectors = self._normalize(vectors)
        if not self.is_trained:
            self.train(vectors)
        self.index.add(vectors.astype(np.float32))

    def search(
        self, query: np.ndarray, top_k: int = 10
    ) -> tuple[np.ndarray, np.ndarray]:
        """Search for nearest neighbors.

        Args:
            query: (Q, dim) query vectors
            top_k: number of results per query

        Returns:
            distances: (Q, top_k) distances/similarities
            indices: (Q, top_k) nearest neighbor indices
        """
        if self.metric == "cosine":
            query = self._normalize(query)

        if hasattr(self.index, "nprobe"):
            self.index.nprobe = 10

        distances, indices = self.index.search(
            query.astype(np.float32), top_k
        )
        return distances, indices

    def save(self, path: str) -> None:
        """Save index to disk."""
        faiss.write_index(self.index, path)

    def load(self, path: str) -> None:
        """Load index from disk."""
        self.index = faiss.read_index(path)
        self.is_trained = True

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        """L2 normalize vectors for cosine similarity."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        return vectors / norms

    @property
    def ntotal(self) -> int:
        return self.index.ntotal


class SimilaritySearch:
    """High-level similarity search interface.

    Combines FAISS index with embedding store for rich search results.
    """

    def __init__(
        self,
        dim: int = 256,
        index_type: str = "Flat",
        metric: str = "cosine",
    ):
        self.faiss_index = FAISSIndex(dim, index_type, metric)
        self.metadata: list[dict[str, Any]] = []
        self.labels: np.ndarray | None = None

    def build_index(
        self,
        embeddings: np.ndarray,
        metadata: list[dict] | None = None,
        labels: np.ndarray | None = None,
    ) -> None:
        """Build search index from embeddings.

        Args:
            embeddings: (N, dim) embedding vectors
            metadata: Optional list of metadata dicts per sample
            labels: Optional (N,) label array
        """
        self.faiss_index.add(embeddings)
        self.metadata = metadata or [{}] * embeddings.shape[0]
        self.labels = labels
        logger.info(f"Built search index with {embeddings.shape[0]} entries")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
    ) -> list[SearchResult]:
        """Search for similar items.

        Args:
            query_embedding: (1, dim) or (dim,) query vector
            top_k: number of results

        Returns:
            List of SearchResult objects
        """
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        distances, indices = self.faiss_index.search(query_embedding, top_k)

        results = []
        for i in range(top_k):
            idx = int(indices[0, i])
            if idx < 0:
                continue
            dist = float(distances[0, i])
            # For cosine similarity (inner product), distance IS the score
            score = dist if self.faiss_index.metric == "cosine" else 1.0 / (1.0 + dist)

            meta = self.metadata[idx] if idx < len(self.metadata) else None
            results.append(SearchResult(
                index=idx,
                distance=dist,
                score=score,
                metadata=meta,
            ))

        return results

    def search_batch(
        self,
        query_embeddings: np.ndarray,
        top_k: int = 10,
    ) -> list[list[SearchResult]]:
        """Batch search for similar items."""
        distances, indices = self.faiss_index.search(query_embeddings, top_k)

        all_results = []
        for q in range(query_embeddings.shape[0]):
            results = []
            for i in range(top_k):
                idx = int(indices[q, i])
                if idx < 0:
                    continue
                dist = float(distances[q, i])
                score = dist if self.faiss_index.metric == "cosine" else 1.0 / (1.0 + dist)
                meta = self.metadata[idx] if idx < len(self.metadata) else None
                results.append(SearchResult(
                    index=idx, distance=dist, score=score, metadata=meta
                ))
            all_results.append(results)

        return all_results

    def save(self, path: str) -> None:
        """Save search index and metadata."""
        import json
        from pathlib import Path

        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        self.faiss_index.save(str(save_dir / "faiss.index"))

        with open(save_dir / "search_metadata.json", "w") as f:
            json.dump(self.metadata, f)

        if self.labels is not None:
            np.save(save_dir / "search_labels.npy", self.labels)

    def load(self, path: str) -> None:
        """Load search index and metadata."""
        import json
        from pathlib import Path

        load_dir = Path(path)
        self.faiss_index.load(str(load_dir / "faiss.index"))

        with open(load_dir / "search_metadata.json") as f:
            self.metadata = json.load(f)

        labels_path = load_dir / "search_labels.npy"
        if labels_path.exists():
            self.labels = np.load(labels_path)
