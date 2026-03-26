"""Graph Neural Network encoders for 3D geometry.

Implements DGCNN (Dynamic Graph CNN) and a flexible GNN encoder
using PyTorch Geometric for processing point clouds as graphs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import (
        EdgeConv,
        GATConv,
        SAGEConv,
        global_max_pool,
        global_mean_pool,
        knn_graph,
    )

    HAS_PYG = True
except ImportError:
    HAS_PYG = False


def knn_graph_batch(x: torch.Tensor, k: int = 20) -> torch.Tensor:
    """Build k-NN graph for batched point clouds (pure PyTorch fallback).

    Args:
        x: (B, N, 3) point coordinates
        k: number of nearest neighbors

    Returns:
        edge_index: (2, B*N*k) edge index tensor
    """
    B, N, C = x.shape
    # Pairwise distances within each batch
    inner = -2 * torch.matmul(x, x.transpose(2, 1))
    xx = torch.sum(x**2, dim=2, keepdim=True)
    pairwise_dist = -xx - inner - xx.transpose(2, 1)

    # Top-k neighbors
    _, idx = pairwise_dist.topk(k=k, dim=-1)  # (B, N, k)

    # Build edge_index in COO format
    device = x.device
    batch_offset = torch.arange(B, device=device).view(-1, 1, 1) * N  # (B, 1, 1)
    idx = idx + batch_offset  # global indices

    src = torch.arange(N, device=device).view(1, -1, 1).repeat(B, 1, k) + batch_offset

    edge_index = torch.stack([src.reshape(-1), idx.reshape(-1)], dim=0)
    return edge_index


class EdgeConvBlock(nn.Module):
    """Edge convolution block (from DGCNN) — pure PyTorch implementation.

    Computes edge features as h(x_i, x_j - x_i) where h is an MLP.
    """

    def __init__(self, in_channels: int, out_channels: int, k: int = 20):
        super().__init__()
        self.k = k
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, N) point features

        Returns:
            out: (B, C_out, N) updated features
        """
        B, C, N = x.shape

        # k-NN graph
        x_t = x.permute(0, 2, 1)  # (B, N, C)
        inner = -2 * torch.matmul(x_t, x_t.transpose(2, 1))
        xx = torch.sum(x_t**2, dim=2, keepdim=True)
        pairwise_dist = -xx - inner - xx.transpose(2, 1)
        idx = pairwise_dist.topk(k=self.k, dim=-1)[1]  # (B, N, k)

        # Gather neighbors
        idx_expanded = idx.unsqueeze(1).expand(-1, C, -1, -1)  # (B, C, N, k)
        x_expanded = x.unsqueeze(-1).expand(-1, -1, -1, self.k)  # (B, C, N, k)
        neighbors = torch.gather(
            x.unsqueeze(2).expand(-1, -1, N, -1),
            dim=3,
            index=idx_expanded,
        )  # (B, C, N, k)

        # Edge features: [x_i, x_j - x_i]
        edge_features = torch.cat([x_expanded, neighbors - x_expanded], dim=1)
        # (B, 2C, N, k)

        out = self.conv(edge_features)  # (B, C_out, N, k)
        out = out.max(dim=-1)[0]  # (B, C_out, N)

        return out


class DGCNNEncoder(nn.Module):
    """Dynamic Graph CNN encoder for point cloud geometry.

    Wang et al., "Dynamic Graph CNN for Learning on Point Clouds" (TOG 2019).
    Builds dynamic k-NN graphs in feature space at each layer.
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 256,
        k: int = 20,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.edge_conv1 = EdgeConvBlock(in_channels, 64, k)
        self.edge_conv2 = EdgeConvBlock(64, 64, k)
        self.edge_conv3 = EdgeConvBlock(64, 128, k)
        self.edge_conv4 = EdgeConvBlock(128, 256, k)

        self.conv5 = nn.Sequential(
            nn.Conv1d(512, 1024, 1, bias=False),  # 64+64+128+256=512
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
        )

        self.fc = nn.Sequential(
            nn.Linear(2048, 512),  # 1024*2 (max+avg pool)
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(512, embed_dim),
            nn.BatchNorm1d(embed_dim),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Args:
            points: (B, N, C) point cloud

        Returns:
            embeddings: (B, embed_dim) geometry embeddings
        """
        x = points[:, :, :3].permute(0, 2, 1)  # (B, 3, N)

        x1 = self.edge_conv1(x)  # (B, 64, N)
        x2 = self.edge_conv2(x1)  # (B, 64, N)
        x3 = self.edge_conv3(x2)  # (B, 128, N)
        x4 = self.edge_conv4(x3)  # (B, 256, N)

        x = torch.cat([x1, x2, x3, x4], dim=1)  # (B, 512, N)
        x = self.conv5(x)  # (B, 1024, N)

        # Global feature: max + average pooling
        x_max = x.max(dim=-1)[0]  # (B, 1024)
        x_avg = x.mean(dim=-1)  # (B, 1024)
        x = torch.cat([x_max, x_avg], dim=1)  # (B, 2048)

        x = self.fc(x)  # (B, embed_dim)
        return x


class GNNEncoder(nn.Module):
    """Flexible GNN encoder using PyTorch Geometric.

    Supports multiple graph convolution types: EdgeConv, GAT, GraphSAGE.
    Requires torch_geometric to be installed.
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 256,
        hidden_channels: list[int] | None = None,
        conv_type: str = "EdgeConv",
        k_neighbors: int = 20,
        dropout: float = 0.3,
        global_pool: str = "mean",
    ):
        super().__init__()
        if not HAS_PYG:
            raise ImportError(
                "torch_geometric required for GNNEncoder. Install with: pip install torch-geometric"
            )

        self.embed_dim = embed_dim
        self.k_neighbors = k_neighbors
        hidden_channels = hidden_channels or [64, 128, 256, 512]

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        prev_ch = in_channels
        for h_ch in hidden_channels:
            if conv_type == "EdgeConv":
                mlp = nn.Sequential(
                    nn.Linear(prev_ch * 2, h_ch),
                    nn.BatchNorm1d(h_ch),
                    nn.ReLU(),
                )
                conv = EdgeConv(mlp, aggr="max")
            elif conv_type == "GATConv":
                conv = GATConv(prev_ch, h_ch, heads=1)
            elif conv_type == "SAGEConv":
                conv = SAGEConv(prev_ch, h_ch)
            else:
                raise ValueError(f"Unknown conv_type: {conv_type}")

            self.convs.append(conv)
            self.bns.append(nn.BatchNorm1d(h_ch))
            prev_ch = h_ch

        self.pool = global_mean_pool if global_pool == "mean" else global_max_pool

        self.fc = nn.Sequential(
            nn.Linear(hidden_channels[-1], 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, embed_dim),
            nn.BatchNorm1d(embed_dim),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Args:
            points: (B, N, C) point cloud

        Returns:
            embeddings: (B, embed_dim) geometry embeddings
        """
        B, N, C = points.shape
        device = points.device

        # Reshape to flat node features
        x = points[:, :, :3].reshape(B * N, 3)

        # Create batch assignment
        batch = torch.arange(B, device=device).unsqueeze(1).repeat(1, N).reshape(-1)

        # Build k-NN graph
        edge_index = knn_graph(x, k=self.k_neighbors, batch=batch, loop=False)

        # GNN layers
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            # Rebuild graph in feature space (dynamic graph)
            edge_index = knn_graph(x, k=self.k_neighbors, batch=batch, loop=False)

        # Global pooling
        x = self.pool(x, batch)  # (B, hidden[-1])

        # Project to embedding space
        x = self.fc(x)  # (B, embed_dim)
        return x
