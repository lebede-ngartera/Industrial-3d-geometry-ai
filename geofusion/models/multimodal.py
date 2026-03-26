"""Multimodal alignment and the main GeoFusion model.

Implements CLIP-style contrastive learning to align geometry, text,
and metadata embeddings in a shared latent space.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """MLP projection head for mapping embeddings to shared space."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultimodalAligner(nn.Module):
    """CLIP-style contrastive alignment for geometry and text.

    Aligns geometry embeddings with text embeddings using
    normalized temperature-scaled cross-entropy (NT-Xent) loss.
    """

    def __init__(
        self,
        geometry_dim: int = 256,
        text_dim: int = 256,
        shared_dim: int = 256,
        hidden_dim: int = 512,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(temperature).log(), requires_grad=True)
        self.shared_dim = shared_dim

        self.geometry_proj = ProjectionHead(geometry_dim, hidden_dim, shared_dim)
        self.text_proj = ProjectionHead(text_dim, hidden_dim, shared_dim)

    def forward(
        self,
        geometry_emb: torch.Tensor,
        text_emb: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute contrastive alignment loss.

        Args:
            geometry_emb: (B, geometry_dim) geometry embeddings
            text_emb: (B, text_dim) text embeddings

        Returns:
            Dictionary with loss, logits, and projected embeddings
        """
        # Project to shared space
        geo_proj = F.normalize(self.geometry_proj(geometry_emb), dim=-1)
        txt_proj = F.normalize(self.text_proj(text_emb), dim=-1)

        # Compute similarity matrix
        temperature = self.temperature.exp().clamp(min=0.01, max=100.0)
        logits = torch.matmul(geo_proj, txt_proj.T) / temperature

        # Symmetric contrastive loss
        B = logits.shape[0]
        labels = torch.arange(B, device=logits.device)
        loss_g2t = F.cross_entropy(logits, labels)
        loss_t2g = F.cross_entropy(logits.T, labels)
        loss = (loss_g2t + loss_t2g) / 2

        return {
            "loss": loss,
            "logits_g2t": logits,
            "logits_t2g": logits.T,
            "geometry_proj": geo_proj,
            "text_proj": txt_proj,
            "temperature": temperature,
        }

    def compute_similarity(
        self,
        geometry_emb: torch.Tensor,
        text_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cosine similarity between geometry and text embeddings."""
        geo_proj = F.normalize(self.geometry_proj(geometry_emb), dim=-1)
        txt_proj = F.normalize(self.text_proj(text_emb), dim=-1)
        return torch.matmul(geo_proj, txt_proj.T)


class MetadataEncoder(nn.Module):
    """Encoder for structured engineering metadata.

    Handles mixed-type metadata: categorical features (material, process),
    continuous features (mass, volume, tolerances), etc.
    """

    def __init__(
        self,
        num_continuous: int = 10,
        num_categorical: int = 5,
        category_sizes: list[int] | None = None,
        categorical_embed_dim: int = 16,
        output_dim: int = 256,
    ):
        super().__init__()
        category_sizes = category_sizes or [20] * num_categorical

        self.continuous_bn = nn.BatchNorm1d(num_continuous)
        self.continuous_fc = nn.Linear(num_continuous, 128)

        self.categorical_embeddings = nn.ModuleList(
            [nn.Embedding(size, categorical_embed_dim) for size in category_sizes]
        )
        cat_total_dim = categorical_embed_dim * len(category_sizes)

        self.fc = nn.Sequential(
            nn.Linear(128 + cat_total_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, output_dim),
            nn.BatchNorm1d(output_dim),
        )

    def forward(
        self,
        continuous: torch.Tensor,
        categorical: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            continuous: (B, num_continuous) continuous features
            categorical: (B, num_categorical) integer category indices

        Returns:
            embeddings: (B, output_dim)
        """
        cont = F.relu(self.continuous_fc(self.continuous_bn(continuous)))

        cat_embeds = []
        for i, emb_layer in enumerate(self.categorical_embeddings):
            cat_embeds.append(emb_layer(categorical[:, i]))
        cat = torch.cat(cat_embeds, dim=-1)

        x = torch.cat([cont, cat], dim=-1)
        return self.fc(x)


class GeoFusionModel(nn.Module):
    """Main GeoFusion multimodal model.

    Integrates geometry encoding, text encoding, metadata encoding,
    and contrastive alignment into a single model.
    """

    def __init__(
        self,
        geometry_encoder: nn.Module,
        text_encoder: nn.Module | None = None,
        metadata_encoder: nn.Module | None = None,
        embed_dim: int = 256,
        temperature: float = 0.07,
        num_classes: int | None = None,
    ):
        super().__init__()
        self.geometry_encoder = geometry_encoder
        self.text_encoder = text_encoder
        self.metadata_encoder = metadata_encoder
        self.embed_dim = embed_dim

        # Multimodal alignment
        if text_encoder is not None:
            self.aligner = MultimodalAligner(
                geometry_dim=embed_dim,
                text_dim=embed_dim,
                shared_dim=embed_dim,
                temperature=temperature,
            )
        else:
            self.aligner = None

        # Optional classification head
        if num_classes is not None:
            self.classifier = nn.Sequential(
                nn.Linear(embed_dim, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, num_classes),
            )
        else:
            self.classifier = None

    def encode_geometry(self, points: torch.Tensor) -> torch.Tensor:
        """Encode 3D point cloud to embedding."""
        return self.geometry_encoder(points)

    def encode_text(self, texts: list[str]) -> torch.Tensor:
        """Encode text descriptions to embedding."""
        if self.text_encoder is None:
            raise RuntimeError("No text encoder configured")
        return self.text_encoder(texts)

    def forward(
        self,
        points: torch.Tensor,
        texts: list[str] | None = None,
        metadata_continuous: torch.Tensor | None = None,
        metadata_categorical: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Full forward pass.

        Args:
            points: (B, N, C) point clouds
            texts: Optional list of text descriptions
            metadata_continuous: Optional (B, D_cont) continuous metadata
            metadata_categorical: Optional (B, D_cat) categorical metadata

        Returns:
            Dictionary with embeddings, losses, and logits
        """
        results = {}

        # Geometry encoding
        geo_emb = self.encode_geometry(points)
        results["geometry_embedding"] = geo_emb

        # Classification
        if self.classifier is not None:
            logits = self.classifier(geo_emb)
            results["logits"] = logits

        # Multimodal alignment
        if texts is not None and self.aligner is not None:
            text_emb = self.encode_text(texts)
            results["text_embedding"] = text_emb
            align_out = self.aligner(geo_emb, text_emb)
            results.update(align_out)

        # Metadata encoding
        if self.metadata_encoder is not None and metadata_continuous is not None:
            meta_emb = self.metadata_encoder(metadata_continuous, metadata_categorical)
            results["metadata_embedding"] = meta_emb

        return results
