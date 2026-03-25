"""Evaluation metrics for geometry understanding and retrieval."""

from __future__ import annotations

import numpy as np
import torch


def compute_accuracy(
    logits: torch.Tensor, targets: torch.Tensor, topk: tuple[int, ...] = (1, 5)
) -> dict[str, float]:
    """Compute top-k classification accuracy.

    Args:
        logits: (B, C) prediction logits
        targets: (B,) ground truth labels
        topk: Tuple of k values for top-k accuracy

    Returns:
        Dictionary of accuracy values keyed by "top1", "top5", etc.
    """
    maxk = max(topk)
    batch_size = targets.size(0)

    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))

    results = {}
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        results[f"top{k}"] = (correct_k / batch_size).item()

    return results


def compute_retrieval_metrics(
    query_embeddings: torch.Tensor,
    gallery_embeddings: torch.Tensor,
    query_labels: torch.Tensor,
    gallery_labels: torch.Tensor,
    top_k: list[int] | None = None,
) -> dict[str, float]:
    """Compute retrieval metrics: Recall@K, Precision@K, mAP.

    Args:
        query_embeddings: (Q, D) query embeddings
        gallery_embeddings: (G, D) gallery embeddings
        query_labels: (Q,) query labels
        gallery_labels: (G,) gallery labels
        top_k: List of k values to evaluate

    Returns:
        Dictionary of metric values
    """
    top_k = top_k or [1, 5, 10]

    # Normalize
    query_norm = torch.nn.functional.normalize(query_embeddings, dim=-1)
    gallery_norm = torch.nn.functional.normalize(gallery_embeddings, dim=-1)

    # Similarity matrix
    sim = torch.matmul(query_norm, gallery_norm.T)

    # Sort by similarity (descending)
    _, indices = sim.sort(dim=1, descending=True)

    metrics = {}

    # Recall@K and Precision@K
    for k in top_k:
        top_k_indices = indices[:, :k]
        top_k_labels = gallery_labels[top_k_indices]
        matches = (top_k_labels == query_labels.unsqueeze(1)).float()

        recall_at_k = (matches.sum(dim=1) > 0).float().mean().item()
        precision_at_k = matches.mean(dim=1).mean().item()

        metrics[f"recall@{k}"] = recall_at_k
        metrics[f"precision@{k}"] = precision_at_k

    # Mean Average Precision
    metrics["mAP"] = compute_map(sim, query_labels, gallery_labels)

    return metrics


def compute_map(
    similarity: torch.Tensor,
    query_labels: torch.Tensor,
    gallery_labels: torch.Tensor,
) -> float:
    """Compute Mean Average Precision (mAP).

    Args:
        similarity: (Q, G) similarity matrix
        query_labels: (Q,) query labels
        gallery_labels: (G,) gallery labels

    Returns:
        mAP score
    """
    _, indices = similarity.sort(dim=1, descending=True)
    aps = []

    for i in range(similarity.shape[0]):
        sorted_labels = gallery_labels[indices[i]]
        matches = (sorted_labels == query_labels[i]).float()

        if matches.sum() == 0:
            aps.append(0.0)
            continue

        # Cumulative precision
        cum_matches = matches.cumsum(dim=0)
        positions = torch.arange(1, len(matches) + 1, device=matches.device).float()
        precisions = cum_matches / positions
        ap = (precisions * matches).sum() / matches.sum()
        aps.append(ap.item())

    return np.mean(aps)


def compute_cross_modal_metrics(
    geo_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    labels: torch.Tensor,
    top_k: list[int] | None = None,
) -> dict[str, float]:
    """Compute cross-modal retrieval metrics (geometry <-> text).

    Args:
        geo_embeddings: (N, D) geometry embeddings
        text_embeddings: (N, D) text embeddings
        labels: (N,) shared labels
        top_k: List of k values

    Returns:
        Dictionary with g2t (geometry-to-text) and t2g metrics
    """
    top_k = top_k or [1, 5, 10]

    # Geometry -> Text retrieval
    g2t_metrics = compute_retrieval_metrics(
        geo_embeddings, text_embeddings, labels, labels, top_k
    )
    g2t_metrics = {f"g2t_{k}": v for k, v in g2t_metrics.items()}

    # Text -> Geometry retrieval
    t2g_metrics = compute_retrieval_metrics(
        text_embeddings, geo_embeddings, labels, labels, top_k
    )
    t2g_metrics = {f"t2g_{k}": v for k, v in t2g_metrics.items()}

    metrics = {}
    metrics.update(g2t_metrics)
    metrics.update(t2g_metrics)
    return metrics
