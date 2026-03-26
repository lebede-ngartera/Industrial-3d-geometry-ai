"""Embedding computation and storage for retrieval."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class EmbeddingStore:
    """Store and manage computed embeddings for retrieval.

    Stores embeddings along with metadata (labels, filenames, etc.)
    for efficient retrieval operations.
    """

    def __init__(self):
        self.embeddings: np.ndarray | None = None
        self.metadata: list[dict[str, Any]] = []
        self.labels: np.ndarray | None = None

    @torch.no_grad()
    def build_from_model(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: str = "cuda",
        modality: str = "geometry",
    ) -> None:
        """Compute embeddings for all samples using a model.

        Args:
            model: Trained model with encode_geometry/encode_text method
            dataloader: DataLoader over the dataset
            device: Compute device
            modality: "geometry" or "text"
        """
        model.eval()
        all_embeddings = []
        all_labels = []
        all_metadata = []

        for batch in dataloader:
            if modality == "geometry":
                points = batch["points"].to(device)
                if hasattr(model, "encode_geometry"):
                    emb = model.encode_geometry(points)
                elif hasattr(model, "encoder"):
                    emb = model.encoder(points)
                else:
                    emb = model(points)
                    if isinstance(emb, tuple):
                        emb = emb[1]  # (logits, embeddings)
                    elif isinstance(emb, dict):
                        emb = emb["geometry_embedding"]
            elif modality == "text":
                texts = batch["text"]
                if hasattr(model, "encode_text"):
                    emb = model.encode_text(texts)
                else:
                    emb = model(texts)
            else:
                raise ValueError(f"Unknown modality: {modality}")

            all_embeddings.append(emb.cpu().numpy())

            if "label" in batch:
                all_labels.append(batch["label"].numpy())

            # Store per-sample metadata
            batch_size = emb.shape[0]
            for i in range(batch_size):
                meta = {}
                if "class_name" in batch:
                    meta["class_name"] = batch["class_name"][i]
                if "category" in batch:
                    meta["category"] = batch["category"][i]
                if "filename" in batch:
                    meta["filename"] = batch["filename"][i]
                if "model_id" in batch:
                    meta["model_id"] = batch["model_id"][i]
                all_metadata.append(meta)

        self.embeddings = np.concatenate(all_embeddings, axis=0)
        self.metadata = all_metadata

        if all_labels:
            self.labels = np.concatenate(all_labels, axis=0)

        logger.info(
            f"Built embedding store: {self.embeddings.shape[0]} samples, "
            f"{self.embeddings.shape[1]}-dim"
        )

    def add_embeddings(
        self,
        embeddings: np.ndarray,
        metadata: list[dict] | None = None,
        labels: np.ndarray | None = None,
    ) -> None:
        """Add pre-computed embeddings to the store."""
        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.concatenate([self.embeddings, embeddings], axis=0)

        if metadata:
            self.metadata.extend(metadata)
        if labels is not None:
            if self.labels is None:
                self.labels = labels
            else:
                self.labels = np.concatenate([self.labels, labels])

    def save(self, path: str) -> None:
        """Save embedding store to disk."""
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        np.save(save_dir / "embeddings.npy", self.embeddings)
        if self.labels is not None:
            np.save(save_dir / "labels.npy", self.labels)

        with open(save_dir / "metadata.json", "w") as f:
            json.dump(self.metadata, f, indent=2)

        logger.info(f"Saved embedding store to {save_dir}")

    def load(self, path: str) -> None:
        """Load embedding store from disk."""
        load_dir = Path(path)

        self.embeddings = np.load(load_dir / "embeddings.npy")

        labels_path = load_dir / "labels.npy"
        if labels_path.exists():
            self.labels = np.load(labels_path)

        metadata_path = load_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                self.metadata = json.load(f)

        logger.info(f"Loaded embedding store: {self.embeddings.shape[0]} samples")

    def __len__(self) -> int:
        return 0 if self.embeddings is None else self.embeddings.shape[0]
