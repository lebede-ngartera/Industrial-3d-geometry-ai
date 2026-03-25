"""PointNet++ encoder for 3D point cloud geometry understanding.

Implements the hierarchical point set learning architecture from
Qi et al., "PointNet++: Deep Hierarchical Feature Learning on
Point Sets in a Metric Space" (NeurIPS 2017).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Compute pairwise squared Euclidean distance.

    Args:
        src: (B, N, C) source points
        dst: (B, M, C) target points

    Returns:
        dist: (B, N, M) squared distances
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).unsqueeze(-1)
    dist += torch.sum(dst ** 2, -1).unsqueeze(-2)
    return dist


def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """Farthest point sampling.

    Args:
        xyz: (B, N, 3) input points
        npoint: number of points to sample

    Returns:
        centroids: (B, npoint) sampled point indices
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device)
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]

    return centroids


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Index points by given indices.

    Args:
        points: (B, N, C) input points
        idx: (B, S) or (B, S, K) indices

    Returns:
        new_points: indexed points
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = (
        torch.arange(B, dtype=torch.long, device=device)
        .view(view_shape)
        .repeat(repeat_shape)
    )
    new_points = points[batch_indices, idx, :]
    return new_points


def query_ball_point(
    radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor
) -> torch.Tensor:
    """Ball query — find all points within a radius.

    Args:
        radius: local region radius
        nsample: max number of neighbors
        xyz: (B, N, 3) all points
        new_xyz: (B, S, 3) query points

    Returns:
        group_idx: (B, S, nsample) grouped point indices
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = (
        torch.arange(N, dtype=torch.long, device=device)
        .view(1, 1, N)
        .repeat(B, S, 1)
    )
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    # Fill with first point if not enough neighbors
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat(1, 1, nsample)
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


class SetAbstractionLayer(nn.Module):
    """PointNet++ Set Abstraction layer.

    Combines sampling, grouping, and PointNet-style feature learning.
    """

    def __init__(
        self,
        npoint: int,
        radius: float,
        nsample: int,
        in_channel: int,
        mlp: list[int],
        group_all: bool = False,
    ):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(
        self, xyz: torch.Tensor, points: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            xyz: (B, N, 3) input coordinates
            points: (B, N, D) input features (or None)

        Returns:
            new_xyz: (B, S, 3) sampled coordinates
            new_points: (B, S, D') abstracted features
        """
        if self.group_all:
            new_xyz, new_points = self._sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = self._sample_and_group(xyz, points)

        # new_points: (B, S, K, D+3)  ->  (B, D+3, K, S) for Conv2d
        new_points = new_points.permute(0, 3, 2, 1)

        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points = F.relu(bn(conv(new_points)))

        # Max pool over neighbors: (B, D', 1, S) -> (B, S, D')
        new_points = torch.max(new_points, 2)[0].permute(0, 2, 1)

        return new_xyz, new_points

    def _sample_and_group(
        self, xyz: torch.Tensor, points: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, N, C = xyz.shape
        S = self.npoint

        fps_idx = farthest_point_sample(xyz, S)
        new_xyz = index_points(xyz, fps_idx)

        idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)
        grouped_xyz = index_points(xyz, idx)
        grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

        if points is not None:
            grouped_points = index_points(points, idx)
            new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
        else:
            new_points = grouped_xyz_norm

        return new_xyz, new_points

    def _sample_and_group_all(
        self, xyz: torch.Tensor, points: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, N, C = xyz.shape
        new_xyz = torch.zeros(B, 1, C, device=xyz.device)
        grouped_xyz = xyz.view(B, 1, N, C)

        if points is not None:
            new_points = torch.cat(
                [grouped_xyz, points.view(B, 1, N, -1)], dim=-1
            )
        else:
            new_points = grouped_xyz

        return new_xyz, new_points


class PointNet2Encoder(nn.Module):
    """PointNet++ encoder for extracting geometry embeddings.

    Hierarchical point set feature learning with multi-scale grouping.
    Produces a fixed-size embedding vector from variable-size point clouds.
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 256,
        use_normals: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.use_normals = use_normals
        self.embed_dim = embed_dim

        in_ch = 3 + 3 if use_normals else 3  # xyz + normals

        # SA layers with increasing abstraction
        self.sa1 = SetAbstractionLayer(
            npoint=512, radius=0.2, nsample=32,
            in_channel=in_ch, mlp=[64, 64, 128],
        )
        self.sa2 = SetAbstractionLayer(
            npoint=128, radius=0.4, nsample=64,
            in_channel=128 + 3, mlp=[128, 128, 256],
        )
        self.sa3 = SetAbstractionLayer(
            npoint=None, radius=None, nsample=None,
            in_channel=256 + 3, mlp=[256, 512, 1024],
            group_all=True,
        )

        # Projection head
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(512, embed_dim)
        self.bn2 = nn.BatchNorm1d(embed_dim)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """Extract geometry embedding from point cloud.

        Args:
            points: (B, N, C) point cloud with C >= 3 channels

        Returns:
            embeddings: (B, embed_dim) geometry embeddings
        """
        B, N, C = points.shape
        xyz = points[:, :, :3]

        if self.use_normals and C >= 6:
            normals = points[:, :, 3:6]
        else:
            normals = None

        # Hierarchical feature extraction
        l1_xyz, l1_points = self.sa1(xyz, normals)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        # Global feature
        x = l3_points.view(B, -1)

        # Project to embedding space
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.bn2(self.fc2(x))

        return x


class PointNet2Classifier(nn.Module):
    """PointNet++ classifier for shape classification.

    Wraps the encoder with a classification head.
    """

    def __init__(
        self,
        num_classes: int = 40,
        in_channels: int = 3,
        embed_dim: int = 256,
        use_normals: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.encoder = PointNet2Encoder(
            in_channels=in_channels,
            embed_dim=embed_dim,
            use_normals=use_normals,
            dropout=dropout,
        )
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(
        self, points: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            points: (B, N, C) input point cloud

        Returns:
            logits: (B, num_classes) classification logits
            embeddings: (B, embed_dim) geometry embeddings
        """
        embeddings = self.encoder(points)
        logits = self.classifier(embeddings)
        return logits, embeddings
