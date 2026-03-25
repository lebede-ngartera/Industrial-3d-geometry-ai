"""Loss functions for geometry understanding and multimodal alignment."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """Normalized Temperature-scaled Cross-Entropy Loss (NT-Xent).

    Used for contrastive learning between geometry and text embeddings.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self, z_i: torch.Tensor, z_j: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            z_i: (B, D) embeddings from modality 1
            z_j: (B, D) embeddings from modality 2

        Returns:
            loss: scalar contrastive loss
        """
        z_i = F.normalize(z_i, dim=-1)
        z_j = F.normalize(z_j, dim=-1)

        B = z_i.shape[0]
        labels = torch.arange(B, device=z_i.device)

        sim_i2j = torch.matmul(z_i, z_j.T) / self.temperature
        sim_j2i = sim_i2j.T

        loss_i2j = F.cross_entropy(sim_i2j, labels)
        loss_j2i = F.cross_entropy(sim_j2i, labels)

        return (loss_i2j + loss_j2i) / 2


class TripletLoss(nn.Module):
    """Triplet loss with semi-hard negative mining."""

    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        dist_pos = F.pairwise_distance(anchor, positive)
        dist_neg = F.pairwise_distance(anchor, negative)
        loss = F.relu(dist_pos - dist_neg + self.margin)
        return loss.mean()


class ClassificationLoss(nn.Module):
    """Cross-entropy classification loss with optional label smoothing."""

    def __init__(self, num_classes: int = 40, label_smoothing: float = 0.1):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        return self.loss_fn(logits, targets)


class MultiTaskLoss(nn.Module):
    """Weighted combination of multiple loss functions.

    Supports learnable task weights via uncertainty weighting
    (Kendall et al., "Multi-Task Learning Using Uncertainty").
    """

    def __init__(
        self,
        num_tasks: int = 2,
        learnable_weights: bool = True,
    ):
        super().__init__()
        self.learnable_weights = learnable_weights
        if learnable_weights:
            # Log variance parameters (initialized to 0 = equal weighting)
            self.log_vars = nn.Parameter(torch.zeros(num_tasks))
        else:
            self.register_buffer("weights", torch.ones(num_tasks) / num_tasks)

    def forward(self, losses: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            losses: List of individual task losses

        Returns:
            Combined weighted loss
        """
        if self.learnable_weights:
            total = 0.0
            for i, loss in enumerate(losses):
                precision = torch.exp(-self.log_vars[i])
                total = total + precision * loss + self.log_vars[i]
            return total
        else:
            return sum(w * l for w, l in zip(self.weights, losses))
