"""Text encoder for engineering descriptions.

Uses a pretrained transformer model (e.g., sentence-transformers)
to encode textual engineering descriptions into the shared embedding space.
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    from transformers import AutoModel, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


class TextEncoder(nn.Module):
    """Transformer-based text encoder for engineering descriptions.

    Encodes text strings into fixed-size embeddings aligned with
    the geometry embedding space via a learned projection head.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        embed_dim: int = 256,
        freeze_backbone: bool = True,
        projection_dim: int | None = None,
    ):
        super().__init__()
        if not HAS_TRANSFORMERS:
            raise ImportError(
                "transformers required for TextEncoder. "
                "Install with: pip install transformers"
            )

        self.embed_dim = embed_dim
        self.model_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Get backbone output dimension
        backbone_dim = self.backbone.config.hidden_size

        # Projection head to align with geometry embedding space
        proj_dim = projection_dim or embed_dim
        self.projection = nn.Sequential(
            nn.Linear(backbone_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
        )

    def _mean_pooling(
        self, model_output: dict, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Mean pooling over token embeddings, respecting attention mask."""
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def encode_text(self, texts: list[str]) -> torch.Tensor:
        """Tokenize and encode a list of text strings.

        Args:
            texts: List of engineering description strings.

        Returns:
            embeddings: (B, embed_dim) text embeddings.
        """
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )

        # Move to same device as model
        device = next(self.projection.parameters()).device
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.set_grad_enabled(self.training and any(
            p.requires_grad for p in self.backbone.parameters()
        )):
            model_output = self.backbone(**encoded)

        pooled = self._mean_pooling(model_output, encoded["attention_mask"])
        embeddings = self.projection(pooled)
        return embeddings

    def forward(self, texts: list[str]) -> torch.Tensor:
        """Forward pass — encode text to embeddings."""
        return self.encode_text(texts)


class SimpleTextEncoder(nn.Module):
    """Lightweight text encoder without pretrained transformers.

    Uses a trainable embedding + 1D CNN for faster prototyping when
    transformers are not available or when training from scratch.
    """

    def __init__(
        self,
        vocab_size: int = 30000,
        embed_dim: int = 256,
        max_length: int = 128,
        num_filters: int = 256,
        kernel_sizes: list[int] | None = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_length = max_length
        kernel_sizes = kernel_sizes or [3, 4, 5]

        self.embedding = nn.Embedding(vocab_size, 128, padding_idx=0)
        self.position = nn.Embedding(max_length, 128)

        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(128, num_filters, k, padding=k // 2),
                nn.BatchNorm1d(num_filters),
                nn.ReLU(),
            )
            for k in kernel_sizes
        ])

        self.fc = nn.Sequential(
            nn.Linear(num_filters * len(kernel_sizes), 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, embed_dim),
            nn.BatchNorm1d(embed_dim),
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: (B, L) integer token IDs

        Returns:
            embeddings: (B, embed_dim)
        """
        B, L = token_ids.shape
        positions = torch.arange(L, device=token_ids.device).unsqueeze(0)
        x = self.embedding(token_ids) + self.position(positions)
        x = x.permute(0, 2, 1)  # (B, 128, L)

        conv_outs = []
        for conv in self.convs:
            c = conv(x)
            c = c.max(dim=-1)[0]  # Global max pool
            conv_outs.append(c)

        x = torch.cat(conv_outs, dim=1)
        x = self.fc(x)
        return x
