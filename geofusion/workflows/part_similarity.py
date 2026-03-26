"""Part similarity retrieval workflow.

'Given a CAD-derived geometry, retrieve similar parts or historically
validated components.' — supports reuse and engineering efficiency.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from geofusion.retrieval.search import SearchResult, SimilaritySearch

logger = logging.getLogger(__name__)


@dataclass
class SimilarityReport:
    """Report from part similarity analysis."""

    query_id: str
    query_category: str | None
    top_matches: list[SearchResult]
    cluster_id: int | None = None
    confidence: float = 0.0
    reuse_candidates: list[dict] = field(default_factory=list)


class PartSimilarityWorkflow:
    """Engineering workflow for finding similar parts.

    Given a query part (point cloud or embedding), finds historically
    validated components that are geometrically similar.

    Use cases:
    - Part reuse during design
    - Finding reference components for new designs
    - Standardization analysis (detecting near-duplicates)
    """

    def __init__(
        self,
        model: nn.Module,
        search_engine: SimilaritySearch,
        device: str = "cuda",
        similarity_threshold: float = 0.8,
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.search_engine = search_engine
        self.similarity_threshold = similarity_threshold

    @torch.no_grad()
    def find_similar(
        self,
        query_points: torch.Tensor,
        top_k: int = 10,
        query_id: str = "unknown",
    ) -> SimilarityReport:
        """Find parts similar to a query geometry.

        Args:
            query_points: (N, C) or (1, N, C) query point cloud
            top_k: Number of results to return
            query_id: Identifier for the query part

        Returns:
            SimilarityReport with ranked matches
        """
        if query_points.ndim == 2:
            query_points = query_points.unsqueeze(0)
        query_points = query_points.to(self.device)

        # Encode
        if hasattr(self.model, "encode_geometry"):
            emb = self.model.encode_geometry(query_points)
        elif hasattr(self.model, "encoder"):
            emb = self.model.encoder(query_points)
        else:
            output = self.model(query_points)
            emb = output[1] if isinstance(output, tuple) else output

        query_emb = emb.cpu().numpy()

        # Search
        results = self.search_engine.search(query_emb, top_k)

        # Filter by threshold
        reuse_candidates = []
        for r in results:
            if r.score >= self.similarity_threshold:
                reuse_candidates.append(
                    {
                        "index": r.index,
                        "score": r.score,
                        "metadata": r.metadata,
                        "recommendation": "REUSE" if r.score > 0.95 else "REVIEW",
                    }
                )

        return SimilarityReport(
            query_id=query_id,
            query_category=None,
            top_matches=results,
            confidence=results[0].score if results else 0.0,
            reuse_candidates=reuse_candidates,
        )

    def find_near_duplicates(
        self,
        embeddings: np.ndarray,
        metadata: list[dict],
        threshold: float = 0.95,
    ) -> list[tuple[int, int, float]]:
        """Find near-duplicate parts in a collection for standardization.

        Args:
            embeddings: (N, D) all part embeddings
            metadata: Per-part metadata
            threshold: Similarity threshold for duplicate detection

        Returns:
            List of (idx_i, idx_j, similarity) tuples
        """
        from sklearn.metrics.pairwise import cosine_similarity

        sim_matrix = cosine_similarity(embeddings)
        duplicates = []

        n = sim_matrix.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                if sim_matrix[i, j] >= threshold:
                    duplicates.append((i, j, float(sim_matrix[i, j])))

        logger.info(f"Found {len(duplicates)} near-duplicate pairs (threshold={threshold})")
        return duplicates

    def cluster_parts(
        self,
        embeddings: np.ndarray,
        n_clusters: int = 20,
    ) -> np.ndarray:
        """Cluster parts by geometric similarity.

        Args:
            embeddings: (N, D) part embeddings
            n_clusters: Number of clusters

        Returns:
            cluster_labels: (N,) cluster assignments
        """
        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        logger.info(f"Clustered {len(labels)} parts into {n_clusters} groups")
        return labels
