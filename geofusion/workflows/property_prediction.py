"""Geometry-to-property prediction workflow.

'Predict approximate physical or manufacturing-related attributes
from shape embeddings.' — connects to simulation and engineering workflows.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class PropertyPrediction:
    """Predicted properties for a shape."""

    part_id: str
    predictions: dict[str, float]
    uncertainties: dict[str, float]
    confidence: float


class PropertyPredictionHead(nn.Module):
    """Multi-property prediction head.

    Predicts multiple physical/manufacturing properties from geometry
    embeddings with uncertainty estimation.
    """

    def __init__(
        self,
        input_dim: int = 256,
        property_names: list[str] | None = None,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.property_names = property_names or [
            "mass",
            "volume",
            "surface_area",
            "max_stress",
            "manufacturability_score",
        ]

        # Shared backbone
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # Per-property heads (predict mean and log-variance for uncertainty)
        self.heads = nn.ModuleDict()
        for name in self.property_names:
            self.heads[name] = nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 2),  # [mean, log_var]
            )

    def forward(self, embeddings: torch.Tensor) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """Predict properties with uncertainty.

        Args:
            embeddings: (B, input_dim) geometry embeddings

        Returns:
            Dictionary of property_name -> (mean, log_var) tensors
        """
        shared_features = self.shared(embeddings)

        predictions = {}
        for name, head in self.heads.items():
            out = head(shared_features)
            mean = out[:, 0]
            log_var = out[:, 1]
            predictions[name] = (mean, log_var)

        return predictions


class PropertyPredictor:
    """Engineering workflow for predicting physical properties from geometry.

    Uses geometry embeddings to predict:
    - Approximate mass, volume, surface area
    - Stress/strain indicators
    - Manufacturability scores
    - Material compatibility estimates

    Includes uncertainty quantification for reliable engineering decisions.
    """

    def __init__(
        self,
        geometry_encoder: nn.Module,
        prediction_head: PropertyPredictionHead,
        device: str = "cuda",
    ):
        self.encoder = geometry_encoder.to(device)
        self.predictor = prediction_head.to(device)
        self.device = device

    @torch.no_grad()
    def predict(
        self,
        points: torch.Tensor,
        part_id: str = "unknown",
    ) -> PropertyPrediction:
        """Predict properties for a single part.

        Args:
            points: (N, C) or (1, N, C) point cloud
            part_id: Part identifier

        Returns:
            PropertyPrediction with values and uncertainties
        """
        if points.ndim == 2:
            points = points.unsqueeze(0)
        points = points.to(self.device)

        self.encoder.eval()
        self.predictor.eval()

        # Encode geometry
        if hasattr(self.encoder, "encode_geometry"):
            emb = self.encoder.encode_geometry(points)
        else:
            emb = self.encoder(points)
            if isinstance(emb, tuple):
                emb = emb[1]

        # Predict properties
        property_outputs = self.predictor(emb)

        predictions = {}
        uncertainties = {}
        for name, (mean, log_var) in property_outputs.items():
            predictions[name] = mean.item()
            uncertainties[name] = torch.exp(0.5 * log_var).item()

        # Confidence based on average uncertainty
        avg_uncertainty = sum(uncertainties.values()) / len(uncertainties)
        confidence = max(0.0, 1.0 - avg_uncertainty)

        return PropertyPrediction(
            part_id=part_id,
            predictions=predictions,
            uncertainties=uncertainties,
            confidence=confidence,
        )

    @torch.no_grad()
    def batch_predict(
        self,
        dataloader: Any,
    ) -> list[PropertyPrediction]:
        """Predict properties for a batch of parts."""
        results = []

        for batch in dataloader:
            points = batch["points"].to(self.device)

            if hasattr(self.encoder, "encode_geometry"):
                emb = self.encoder.encode_geometry(points)
            else:
                emb = self.encoder(points)
                if isinstance(emb, tuple):
                    emb = emb[1]

            property_outputs = self.predictor(emb)

            for i in range(points.shape[0]):
                predictions = {}
                uncertainties = {}
                for name, (mean, log_var) in property_outputs.items():
                    predictions[name] = mean[i].item()
                    uncertainties[name] = torch.exp(0.5 * log_var[i]).item()

                avg_unc = sum(uncertainties.values()) / len(uncertainties)
                part_id = batch.get("filename", [""] * points.shape[0])
                pid = part_id[i] if isinstance(part_id, (list, tuple)) else str(part_id)

                results.append(
                    PropertyPrediction(
                        part_id=pid,
                        predictions=predictions,
                        uncertainties=uncertainties,
                        confidence=max(0.0, 1.0 - avg_unc),
                    )
                )

        return results

    @staticmethod
    def gaussian_nll_loss(
        predictions: dict[str, tuple[torch.Tensor, torch.Tensor]],
        targets: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Gaussian negative log-likelihood loss for uncertainty-aware training.

        Args:
            predictions: Dict of property -> (mean, log_var) tensors
            targets: Dict of property -> target value tensors

        Returns:
            Total loss
        """
        total_loss = torch.tensor(0.0)
        for name in predictions:
            if name in targets:
                mean, log_var = predictions[name]
                target = targets[name]
                # NLL = 0.5 * (log_var + (target - mean)^2 / var)
                var = torch.exp(log_var)
                loss = 0.5 * (log_var + (target - mean) ** 2 / var)
                total_loss = total_loss + loss.mean()
        return total_loss
