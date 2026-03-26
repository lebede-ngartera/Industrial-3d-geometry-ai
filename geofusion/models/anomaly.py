"""Anomaly detection for 3D geometry.

Detects unusual geometric patterns that may indicate manufacturability
or quality risks. Supports reconstruction-based and density-based methods.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class PointCloudAutoencoder(nn.Module):
    """Point cloud autoencoder for reconstruction-based anomaly detection.

    Learns to reconstruct normal geometries; anomalous shapes will have
    higher reconstruction error.
    """

    def __init__(
        self,
        num_points: int = 2048,
        latent_dim: int = 64,
        point_dim: int = 3,
    ):
        super().__init__()
        self.num_points = num_points
        self.latent_dim = latent_dim

        # Encoder: points -> latent
        self.encoder = nn.Sequential(
            nn.Conv1d(point_dim, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )
        self.fc_encode = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )

        # Decoder: latent -> points
        self.fc_decode = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, point_dim, 1),
        )

    def encode(self, points: torch.Tensor) -> torch.Tensor:
        """Encode point cloud to latent representation.

        Args:
            points: (B, N, 3) point cloud

        Returns:
            z: (B, latent_dim) latent code
        """
        x = points.permute(0, 2, 1)  # (B, 3, N)
        x = self.encoder(x)  # (B, 512, N)
        x = x.max(dim=-1)[0]  # (B, 512)
        z = self.fc_encode(x)  # (B, latent_dim)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent code to point cloud.

        Args:
            z: (B, latent_dim) latent code

        Returns:
            points: (B, N, 3) reconstructed point cloud
        """
        x = self.fc_decode(z)  # (B, 512)
        x = x.unsqueeze(-1).repeat(1, 1, self.num_points)  # (B, 512, N)
        x = self.decoder(x)  # (B, 3, N)
        return x.permute(0, 2, 1)  # (B, N, 3)

    def forward(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(points)
        reconstructed = self.decode(z)
        return reconstructed, z


def chamfer_distance(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute Chamfer distance between two point clouds.

    Args:
        pred: (B, N, 3) predicted points
        target: (B, M, 3) target points

    Returns:
        distance: (B,) chamfer distance per sample
    """
    # pred -> target
    diff = pred.unsqueeze(2) - target.unsqueeze(1)  # (B, N, M, 3)
    dist_matrix = (diff**2).sum(-1)  # (B, N, M)

    min_dist_pred = dist_matrix.min(dim=2)[0].mean(dim=1)  # (B,)
    min_dist_target = dist_matrix.min(dim=1)[0].mean(dim=1)  # (B,)

    return min_dist_pred + min_dist_target


class GeometryAnomalyDetector(nn.Module):
    """Anomaly detection system for 3D engineering geometry.

    Combines reconstruction-based anomaly scoring with learned
    normality representations. Higher anomaly scores indicate
    shapes that deviate from the training distribution.

    Methods:
    - reconstruction: Chamfer distance of autoencoder reconstruction
    - density: Distance to nearest normal embedding cluster
    - ensemble: Combination of both methods
    """

    def __init__(
        self,
        num_points: int = 2048,
        latent_dim: int = 64,
        method: str = "reconstruction",
        threshold_percentile: float = 95.0,
    ):
        super().__init__()
        self.method = method
        self.threshold_percentile = threshold_percentile
        self.threshold: float | None = None

        self.autoencoder = PointCloudAutoencoder(
            num_points=num_points,
            latent_dim=latent_dim,
        )

        # For density-based detection, store normal embeddings
        self.register_buffer(
            "normal_embeddings",
            torch.zeros(0, latent_dim),
        )

    def fit_threshold(self, dataloader: object) -> float:
        """Compute anomaly threshold from normal training data.

        Args:
            dataloader: DataLoader of normal (non-anomalous) samples

        Returns:
            threshold: Anomaly score threshold
        """
        self.eval()
        scores = []
        with torch.no_grad():
            for batch in dataloader:
                points = batch["points"]
                if next(self.parameters()).is_cuda:
                    points = points.cuda()
                score = self.anomaly_score(points)
                scores.append(score.cpu())

        all_scores = torch.cat(scores).numpy()
        self.threshold = float(np.percentile(all_scores, self.threshold_percentile))
        return self.threshold

    def store_normal_embeddings(self, dataloader: object) -> None:
        """Store embeddings of normal samples for density-based detection."""
        self.eval()
        embeddings = []
        with torch.no_grad():
            for batch in dataloader:
                points = batch["points"]
                if next(self.parameters()).is_cuda:
                    points = points.cuda()
                z = self.autoencoder.encode(points[:, :, :3])
                embeddings.append(z.cpu())

        self.normal_embeddings = torch.cat(embeddings, dim=0)

    def anomaly_score(self, points: torch.Tensor) -> torch.Tensor:
        """Compute anomaly score for input point clouds.

        Args:
            points: (B, N, 3+) input point clouds

        Returns:
            scores: (B,) anomaly scores (higher = more anomalous)
        """
        xyz = points[:, :, :3]

        if self.method == "reconstruction":
            reconstructed, z = self.autoencoder(xyz)
            scores = chamfer_distance(reconstructed, xyz)

        elif self.method == "density":
            z = self.autoencoder.encode(xyz)
            if self.normal_embeddings.shape[0] == 0:
                raise RuntimeError("Call store_normal_embeddings() first for density method")
            normal = self.normal_embeddings.to(z.device)
            # Distance to nearest normal embedding
            dists = torch.cdist(z, normal)  # (B, num_normal)
            scores = dists.min(dim=1)[0]  # (B,)

        elif self.method == "ensemble":
            reconstructed, z = self.autoencoder(xyz)
            recon_score = chamfer_distance(reconstructed, xyz)

            if self.normal_embeddings.shape[0] > 0:
                normal = self.normal_embeddings.to(z.device)
                dists = torch.cdist(z, normal)
                density_score = dists.min(dim=1)[0]
                # Normalize and combine
                scores = 0.5 * recon_score + 0.5 * density_score
            else:
                scores = recon_score
        else:
            raise ValueError(f"Unknown method: {self.method}")

        return scores

    def detect(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Detect anomalies in input point clouds.

        Args:
            points: (B, N, 3+) input point clouds

        Returns:
            is_anomaly: (B,) boolean mask
            scores: (B,) anomaly scores
        """
        scores = self.anomaly_score(points)

        if self.threshold is None:
            raise RuntimeError("Call fit_threshold() first")

        is_anomaly = scores > self.threshold
        return is_anomaly, scores

    def forward(self, points: torch.Tensor) -> dict[str, torch.Tensor]:
        """Training forward pass."""
        xyz = points[:, :, :3]
        reconstructed, z = self.autoencoder(xyz)
        recon_loss = chamfer_distance(reconstructed, xyz).mean()

        return {
            "loss": recon_loss,
            "reconstructed": reconstructed,
            "latent": z,
            "chamfer_distance": recon_loss,
        }
