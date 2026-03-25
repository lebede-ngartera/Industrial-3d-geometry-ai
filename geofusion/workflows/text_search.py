"""Text-to-geometry search workflow.

'Given a text description like "lightweight bracket with curved support arm,"
retrieve candidate 3D designs.' — multimodal foundation model capability.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from geofusion.retrieval.cross_modal import CrossModalRetriever
from geofusion.retrieval.search import SearchResult

logger = logging.getLogger(__name__)


@dataclass
class TextSearchResult:
    """Result from text-to-geometry search."""
    query: str
    results: list[SearchResult]
    num_results: int
    best_match_score: float
    interpretation: str = ""


class TextToGeometrySearch:
    """Engineering workflow for text-based 3D shape search.

    Allows engineers to find relevant CAD parts using natural language
    descriptions, supporting early-stage design exploration.

    Example queries:
    - "lightweight bracket with curved support arm"
    - "cylindrical housing with mounting flange"
    - "thin-walled aerodynamic panel"
    """

    def __init__(
        self,
        retriever: CrossModalRetriever,
    ):
        self.retriever = retriever

    def search(
        self,
        query: str,
        top_k: int = 10,
        min_score: float = 0.0,
    ) -> TextSearchResult:
        """Search for 3D shapes matching a text description.

        Args:
            query: Natural language shape description
            top_k: Maximum number of results
            min_score: Minimum similarity score to include

        Returns:
            TextSearchResult with ranked matches
        """
        results = self.retriever.text_to_shape(query, top_k=top_k)

        # Filter by minimum score
        filtered = [r for r in results if r.score >= min_score]

        interpretation = self._interpret_results(query, filtered)

        return TextSearchResult(
            query=query,
            results=filtered,
            num_results=len(filtered),
            best_match_score=filtered[0].score if filtered else 0.0,
            interpretation=interpretation,
        )

    def batch_search(
        self,
        queries: list[str],
        top_k: int = 10,
    ) -> list[TextSearchResult]:
        """Search multiple queries.

        Args:
            queries: List of text descriptions
            top_k: Results per query

        Returns:
            List of TextSearchResult
        """
        return [self.search(q, top_k) for q in queries]

    def _interpret_results(
        self, query: str, results: list[SearchResult]
    ) -> str:
        """Generate human-readable interpretation of search results."""
        if not results:
            return f"No matching shapes found for: '{query}'"

        best = results[0]
        categories = set()
        for r in results[:5]:
            if r.metadata:
                cat = r.metadata.get("category", r.metadata.get("class_name"))
                if cat:
                    categories.add(cat)

        cat_str = ", ".join(sorted(categories)) if categories else "various"

        if best.score > 0.9:
            quality = "High confidence match"
        elif best.score > 0.7:
            quality = "Moderate confidence match"
        else:
            quality = "Low confidence match — refine query"

        return (
            f"{quality}. Found {len(results)} candidates. "
            f"Top categories: {cat_str}. "
            f"Best similarity score: {best.score:.3f}."
        )
