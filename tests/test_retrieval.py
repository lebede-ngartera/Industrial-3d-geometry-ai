"""Tests for retrieval system — embedding store, FAISS index, search."""

import tempfile

import numpy as np

from geofusion.retrieval.embeddings import EmbeddingStore
from geofusion.retrieval.search import FAISSIndex, SimilaritySearch

# ---------------------------------------------------------------------------
# EmbeddingStore tests
# ---------------------------------------------------------------------------


class TestEmbeddingStore:
    def test_add_and_len(self):
        store = EmbeddingStore()
        emb = np.random.randn(5, 64).astype(np.float32)
        labels = np.array([0, 1, 2, 3, 4])
        store.add_embeddings(emb, labels=labels)
        assert len(store) == 5

    def test_add_incremental(self):
        store = EmbeddingStore()
        store.add_embeddings(np.random.randn(3, 32).astype(np.float32), labels=np.array([0, 1, 2]))
        store.add_embeddings(np.random.randn(2, 32).astype(np.float32), labels=np.array([3, 4]))
        assert len(store) == 5
        assert store.embeddings.shape == (5, 32)

    def test_save_load(self):
        store = EmbeddingStore()
        embs = np.random.randn(10, 16).astype(np.float32)
        labels = np.arange(10)
        metadata = [{"name": f"item_{i}"} for i in range(10)]
        store.add_embeddings(embs, metadata=metadata, labels=labels)

        with tempfile.TemporaryDirectory() as tmpdir:
            store.save(tmpdir)
            loaded = EmbeddingStore()
            loaded.load(tmpdir)

            assert len(loaded) == 10
            np.testing.assert_allclose(loaded.embeddings, embs)
            np.testing.assert_array_equal(loaded.labels, labels)
            assert loaded.metadata[0]["name"] == "item_0"


# ---------------------------------------------------------------------------
# FAISSIndex tests
# ---------------------------------------------------------------------------


class TestFAISSIndex:
    def test_add_and_search(self):
        dim = 64
        n = 100
        index = FAISSIndex(dim=dim, metric="cosine")

        vectors = np.random.randn(n, dim).astype(np.float32)
        index.add(vectors)

        query = vectors[0:1]
        distances, indices = index.search(query, top_k=5)
        assert indices.shape == (1, 5)
        assert indices[0, 0] == 0  # Closest to itself

    def test_l2_metric(self):
        dim = 32
        index = FAISSIndex(dim=dim, metric="l2")
        vectors = np.random.randn(50, dim).astype(np.float32)
        index.add(vectors)

        distances, indices = index.search(vectors[:2], top_k=3)
        assert distances.shape == (2, 3)


# ---------------------------------------------------------------------------
# SimilaritySearch tests
# ---------------------------------------------------------------------------


class TestSimilaritySearch:
    def setup_method(self):
        self.dim = 32
        self.n = 50
        self.search = SimilaritySearch(dim=self.dim, metric="cosine")

        embeddings = np.random.randn(self.n, self.dim).astype(np.float32)
        metadata = [{"class": f"cls_{i % 5}"} for i in range(self.n)]
        labels = np.arange(self.n)
        self.search.build_index(embeddings, metadata, labels)

    def test_search_returns_results(self):
        query = np.random.randn(1, self.dim).astype(np.float32)
        results = self.search.search(query, top_k=5)
        assert len(results) == 5
        assert all(hasattr(r, "score") for r in results)

    def test_search_scores_sorted(self):
        query = np.random.randn(1, self.dim).astype(np.float32)
        results = self.search.search(query, top_k=10)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_batch_search(self):
        queries = np.random.randn(3, self.dim).astype(np.float32)
        batch_results = self.search.search_batch(queries, top_k=5)
        assert len(batch_results) == 3
        assert all(len(r) == 5 for r in batch_results)

    def test_save_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.search.save(tmpdir)
            loaded = SimilaritySearch(dim=self.dim, metric="cosine")
            loaded.load(tmpdir)

            query = np.random.randn(1, self.dim).astype(np.float32)
            r1 = self.search.search(query, top_k=3)
            r2 = loaded.search(query, top_k=3)
            assert [r.index for r in r1] == [r.index for r in r2]
