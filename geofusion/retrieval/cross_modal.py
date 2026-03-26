"""Cross-modal retrieval: geometry <-> text <-> metadata."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from geofusion.retrieval.search import SearchResult, SimilaritySearch

logger = logging.getLogger(__name__)


class CrossModalRetriever:
    """Cross-modal retrieval system for engineering search.

    Supports:
    - Text-to-Shape: Find 3D shapes matching a text description
    - Shape-to-Text: Find text descriptions matching a shape
    - Shape-to-Shape: Find similar shapes
    - Text-to-Text: Find similar descriptions
    """

    def __init__(
        self,
        model: nn.Module,
        embed_dim: int = 256,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = device

        self.geo_search = SimilaritySearch(dim=embed_dim)
        self.text_search = SimilaritySearch(dim=embed_dim)

        self.geo_embeddings: np.ndarray | None = None
        self.text_embeddings: np.ndarray | None = None
        self.geo_metadata: list[dict] = []
        self.text_metadata: list[dict] = []

    @torch.no_grad()
    def build_index(
        self,
        dataloader: Any,
        include_text: bool = True,
    ) -> None:
        """Build retrieval indices from a dataset.

        Args:
            dataloader: DataLoader yielding batches with 'points' and optionally 'text'
            include_text: Whether to also build text index
        """
        geo_embs = []
        text_embs = []
        geo_metas = []
        text_metas = []

        for batch in dataloader:
            points = batch["points"].to(self.device)

            # Geometry embeddings
            if hasattr(self.model, "encode_geometry"):
                geo_emb = self.model.encode_geometry(points)
            else:
                geo_emb = self.model(points)
                if isinstance(geo_emb, dict):
                    geo_emb = geo_emb["geometry_embedding"]

            geo_embs.append(geo_emb.cpu().numpy())

            # Per-sample metadata
            bs = points.shape[0]
            for i in range(bs):
                meta = {}
                for key in ["class_name", "category", "filename", "model_id"]:
                    if key in batch:
                        val = batch[key]
                        meta[key] = (
                            val[i]
                            if isinstance(val, (list, tuple))
                            else val[i].item()
                            if hasattr(val[i], "item")
                            else str(val[i])
                        )
                geo_metas.append(meta)

            # Text embeddings
            if include_text and "text" in batch:
                texts = batch["text"]
                if hasattr(self.model, "encode_text"):
                    text_emb = self.model.encode_text(texts)
                    text_embs.append(text_emb.cpu().numpy())
                    for i in range(bs):
                        text_metas.append({"text": texts[i], **geo_metas[-bs + i]})

        self.geo_embeddings = np.concatenate(geo_embs, axis=0)
        self.geo_metadata = geo_metas
        self.geo_search.build_index(self.geo_embeddings, geo_metas)

        if text_embs:
            self.text_embeddings = np.concatenate(text_embs, axis=0)
            self.text_metadata = text_metas
            self.text_search.build_index(self.text_embeddings, text_metas)

        logger.info(
            f"Built cross-modal index: {len(geo_metas)} geometry entries"
            + (f", {len(text_metas)} text entries" if text_embs else "")
        )

    @torch.no_grad()
    def text_to_shape(
        self,
        query_text: str,
        top_k: int = 10,
    ) -> list[SearchResult]:
        """Retrieve shapes matching a text description.

        Args:
            query_text: Natural language shape description
            top_k: Number of results

        Returns:
            List of SearchResult with shape metadata
        """
        if not hasattr(self.model, "encode_text"):
            raise RuntimeError("Model does not support text encoding")

        # Encode query text
        text_emb = self.model.encode_text([query_text])
        query = text_emb.cpu().numpy()

        # Search in geometry embedding space via the aligner projection
        if hasattr(self.model, "aligner") and self.model.aligner is not None:
            # Project text to shared space
            from torch.nn.functional import normalize

            text_proj = normalize(self.model.aligner.text_proj(text_emb), dim=-1).cpu().numpy()

            # Build temporary projected index and search
            geo_proj = (
                normalize(
                    self.model.aligner.geometry_proj(
                        torch.from_numpy(self.geo_embeddings).to(self.device)
                    ),
                    dim=-1,
                )
                .cpu()
                .numpy()
            )

            # Cosine similarity
            similarities = (text_proj @ geo_proj.T).squeeze(0)
            top_indices = np.argsort(similarities)[::-1][:top_k]

            results = []
            for idx in top_indices:
                results.append(
                    SearchResult(
                        index=int(idx),
                        distance=float(1 - similarities[idx]),
                        score=float(similarities[idx]),
                        metadata=self.geo_metadata[idx],
                    )
                )
            return results

        # Fallback: direct embedding search
        return self.geo_search.search(query, top_k)

    @torch.no_grad()
    def shape_to_shape(
        self,
        query_points: torch.Tensor,
        top_k: int = 10,
    ) -> list[SearchResult]:
        """Find similar shapes to a query point cloud.

        Args:
            query_points: (1, N, C) or (N, C) query point cloud
            top_k: Number of results

        Returns:
            List of SearchResult with shape metadata
        """
        if query_points.ndim == 2:
            query_points = query_points.unsqueeze(0)

        query_points = query_points.to(self.device)

        if hasattr(self.model, "encode_geometry"):
            emb = self.model.encode_geometry(query_points)
        else:
            emb = self.model(query_points)
            if isinstance(emb, dict):
                emb = emb["geometry_embedding"]

        query = emb.cpu().numpy()
        return self.geo_search.search(query, top_k)

    @torch.no_grad()
    def shape_to_text(
        self,
        query_points: torch.Tensor,
        top_k: int = 10,
    ) -> list[SearchResult]:
        """Find text descriptions matching a query shape.

        Args:
            query_points: (1, N, C) or (N, C) query point cloud
            top_k: Number of results

        Returns:
            List of SearchResult with text metadata
        """
        if self.text_embeddings is None:
            raise RuntimeError("No text index built. Build index with include_text=True")

        if query_points.ndim == 2:
            query_points = query_points.unsqueeze(0)

        query_points = query_points.to(self.device)

        if hasattr(self.model, "encode_geometry"):
            emb = self.model.encode_geometry(query_points)
        else:
            emb = self.model(query_points)
            if isinstance(emb, dict):
                emb = emb["geometry_embedding"]

        query = emb.cpu().numpy()
        return self.text_search.search(query, top_k)

    def get_statistics(self) -> dict[str, Any]:
        """Get summary statistics of the retrieval index."""
        stats = {
            "num_geometry_entries": len(self.geo_metadata),
            "geometry_embed_dim": self.geo_embeddings.shape[1]
            if self.geo_embeddings is not None
            else 0,
        }
        if self.text_embeddings is not None:
            stats["num_text_entries"] = len(self.text_metadata)
            stats["text_embed_dim"] = self.text_embeddings.shape[1]

        # Category distribution
        categories = {}
        for meta in self.geo_metadata:
            cat = meta.get("category", meta.get("class_name", "unknown"))
            categories[cat] = categories.get(cat, 0) + 1
        stats["category_distribution"] = categories

        return stats
