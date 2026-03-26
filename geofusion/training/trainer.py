"""Training loop for GeoFusion models.

Supports single-task geometry training, multimodal alignment training,
and multi-task combined training with logging and checkpointing.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader

from geofusion.training.losses import ClassificationLoss, NTXentLoss
from geofusion.training.metrics import compute_accuracy

logger = logging.getLogger(__name__)


class Trainer:
    """Training orchestrator for GeoFusion models.

    Handles training loops, validation, checkpointing, logging,
    and early stopping.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict[str, Any],
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # Training config
        train_cfg = config.get("training", {})
        self.epochs = train_cfg.get("epochs", 200)
        self.lr = train_cfg.get("learning_rate", 0.001)
        self.weight_decay = train_cfg.get("weight_decay", 0.0001)
        self.gradient_clip = train_cfg.get("gradient_clip", 1.0)
        self.warmup_epochs = train_cfg.get("warmup_epochs", 10)
        self.save_every = train_cfg.get("save_every", 10)
        self.patience = train_cfg.get("early_stopping_patience", 20)

        # Output
        self.output_dir = Path(config.get("project", {}).get("output_dir", "outputs"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # Scheduler
        scheduler_type = train_cfg.get("scheduler", "cosine")
        if scheduler_type == "cosine":
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=1e-6)
        elif scheduler_type == "plateau":
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", patience=5, factor=0.5)
        else:
            self.scheduler = None

        # Loss functions
        self.cls_loss = ClassificationLoss(
            num_classes=config.get("data", {}).get("num_classes", 40)
        )
        self.contrastive_loss = NTXentLoss(
            temperature=config.get("multimodal", {}).get("temperature", 0.07)
        )

        # State
        self.current_epoch = 0
        self.best_val_metric = float("inf")
        self.patience_counter = 0
        self.history: list[dict] = []

        # Wandb integration
        self.use_wandb = config.get("logging", {}).get("use_wandb", False)
        self.wandb_run = None
        if self.use_wandb:
            try:
                import wandb

                self.wandb_run = wandb.init(
                    project=config["logging"].get("wandb_project", "geofusion"),
                    config=config,
                )
            except ImportError:
                logger.warning("wandb not installed, skipping experiment tracking")
                self.use_wandb = False

    def train(self) -> dict[str, Any]:
        """Run full training loop.

        Returns:
            Training history dictionary
        """
        logger.info(f"Starting training for {self.epochs} epochs")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(self.epochs):
            self.current_epoch = epoch
            epoch_start = time.time()

            # Training
            train_metrics = self._train_epoch()

            # Validation
            val_metrics = self._validate_epoch()

            # Scheduling
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_metrics.get("val_loss", 0))
            elif self.scheduler is not None:
                self.scheduler.step()

            epoch_time = time.time() - epoch_start

            # Log
            metrics = {
                "epoch": epoch,
                "lr": self.optimizer.param_groups[0]["lr"],
                "epoch_time": epoch_time,
                **train_metrics,
                **val_metrics,
            }
            self.history.append(metrics)

            if epoch % max(1, self.config.get("logging", {}).get("log_every", 10)) == 0:
                logger.info(
                    f"Epoch {epoch}/{self.epochs} | "
                    f"Train Loss: {train_metrics.get('train_loss', 0):.4f} | "
                    f"Val Loss: {val_metrics.get('val_loss', 0):.4f} | "
                    f"Val Acc: {val_metrics.get('val_top1', 0):.4f} | "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.6f} | "
                    f"Time: {epoch_time:.1f}s"
                )

            if self.use_wandb and self.wandb_run:
                import wandb

                wandb.log(metrics)

            # Checkpointing
            val_loss = val_metrics.get("val_loss", float("inf"))
            if val_loss < self.best_val_metric:
                self.best_val_metric = val_loss
                self.patience_counter = 0
                self._save_checkpoint("best_model.pt")
            else:
                self.patience_counter += 1

            if epoch % self.save_every == 0:
                self._save_checkpoint(f"checkpoint_epoch_{epoch}.pt")

            # Early stopping
            if self.patience_counter >= self.patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        self._save_checkpoint("final_model.pt")

        if self.use_wandb and self.wandb_run:
            self.wandb_run.finish()

        return {"history": self.history, "best_val_metric": self.best_val_metric}

    def _train_epoch(self) -> dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch in self.train_loader:
            points = batch["points"].to(self.device)
            labels = batch.get("label")
            texts = batch.get("text")

            self.optimizer.zero_grad()

            # Forward
            output = self.model(
                points=points,
                texts=texts if isinstance(texts, list) else None,
            )

            # Compute loss
            loss = torch.tensor(0.0, device=self.device)

            if "logits" in output and labels is not None:
                labels = labels.to(self.device)
                cls_loss = self.cls_loss(output["logits"], labels)
                loss = loss + cls_loss

                preds = output["logits"].argmax(dim=1)
                total_correct += (preds == labels).sum().item()

            if "loss" in output:
                loss = loss + output["loss"]

            loss.backward()

            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)

            self.optimizer.step()

            total_loss += loss.item() * points.size(0)
            total_samples += points.size(0)

        avg_loss = total_loss / max(total_samples, 1)
        avg_acc = total_correct / max(total_samples, 1)

        return {"train_loss": avg_loss, "train_acc": avg_acc}

    @torch.no_grad()
    def _validate_epoch(self) -> dict[str, float]:
        """Run validation."""
        self.model.eval()
        total_loss = 0.0
        all_logits = []
        all_labels = []

        for batch in self.val_loader:
            points = batch["points"].to(self.device)
            labels = batch.get("label")
            texts = batch.get("text")

            output = self.model(
                points=points,
                texts=texts if isinstance(texts, list) else None,
            )

            loss = torch.tensor(0.0, device=self.device)

            if "logits" in output and labels is not None:
                labels = labels.to(self.device)
                cls_loss = self.cls_loss(output["logits"], labels)
                loss = loss + cls_loss
                all_logits.append(output["logits"].cpu())
                all_labels.append(labels.cpu())

            if "loss" in output:
                loss = loss + output["loss"]

            total_loss += loss.item() * points.size(0)

        num_samples = sum(b["points"].size(0) for b in [])  # placeholder
        num_samples = max(sum(l.size(0) for l in all_labels), 1) if all_labels else 1
        avg_loss = total_loss / num_samples

        metrics = {"val_loss": avg_loss}

        if all_logits:
            logits = torch.cat(all_logits)
            labels = torch.cat(all_labels)
            acc = compute_accuracy(logits, labels, topk=(1, 5))
            metrics.update({f"val_{k}": v for k, v in acc.items()})

        return metrics

    def _save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint."""
        path = self.output_dir / filename
        torch.save(
            {
                "epoch": self.current_epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_val_metric": self.best_val_metric,
                "config": self.config,
            },
            path,
        )
        logger.info(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint.get("epoch", 0)
        self.best_val_metric = checkpoint.get("best_val_metric", float("inf"))
        logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
