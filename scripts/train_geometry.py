"""Train geometry encoder (PointNet++ or DGCNN) on ModelNet40."""

import argparse
import logging

import torch
import yaml
from torch.utils.data import DataLoader, random_split

from geofusion.data.datasets import ModelNet40Dataset
from geofusion.data.transforms import (
    Compose,
    FarthestPointSample,
    NormalizePointCloud,
    RandomJitter,
    RandomRotate,
    RandomScale,
)
from geofusion.models.pointnet2 import PointNet2Classifier
from geofusion.models.gnn_encoder import DGCNNEncoder
from geofusion.training.trainer import Trainer

logger = logging.getLogger(__name__)


def build_model(config: dict) -> torch.nn.Module:
    """Build geometry model from config."""
    backbone = config.get("geometry_encoder", {}).get("backbone", "pointnet2")
    embed_dim = config.get("geometry_encoder", {}).get("embed_dim", 256)
    num_classes = 40
    dropout = config.get("geometry_encoder", {}).get("dropout", 0.3)
    use_normals = config.get("data", {}).get("use_normals", True)

    if backbone == "pointnet2":
        model = PointNet2Classifier(
            num_classes=num_classes,
            embed_dim=embed_dim,
            use_normals=use_normals,
            dropout=dropout,
        )
    elif backbone == "dgcnn":
        encoder = DGCNNEncoder(
            in_channels=3,
            embed_dim=embed_dim,
            k=config.get("geometry_encoder", {}).get("k_neighbors", 20),
            dropout=dropout,
        )
        classifier = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(128, num_classes),
        )
        model = torch.nn.ModuleDict({"encoder": encoder, "classifier": classifier})
    else:
        raise ValueError(f"Unknown backbone: {backbone}")

    return model


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Train geometry encoder")
    parser.add_argument("--config", default="configs/pointnet2.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Overrides
    if args.data_root:
        config.setdefault("data", {})["data_root"] = args.data_root
    if args.device:
        config.setdefault("project", {})["device"] = args.device

    device = config.get("project", {}).get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        logger.info("CUDA not available, using CPU")

    # Seed
    seed = config.get("project", {}).get("seed", 42)
    torch.manual_seed(seed)

    # Data
    data_cfg = config.get("data", {})
    num_points = data_cfg.get("num_points", 2048)
    use_normals = data_cfg.get("use_normals", True)

    train_transform = Compose([
        FarthestPointSample(num_points),
        NormalizePointCloud(),
        RandomRotate(axis="y"),
        RandomJitter(sigma=0.01),
        RandomScale(lo=0.8, hi=1.25),
    ])
    val_transform = Compose([
        FarthestPointSample(num_points),
        NormalizePointCloud(),
    ])

    data_root = data_cfg.get("data_root", "data/raw/modelnet40_normal_resampled")

    train_dataset = ModelNet40Dataset(
        data_root=data_root, split="train",
        num_points=num_points, use_normals=use_normals,
        transform=train_transform,
    )
    test_dataset = ModelNet40Dataset(
        data_root=data_root, split="test",
        num_points=num_points, use_normals=use_normals,
        transform=val_transform,
    )

    # Split train into train/val
    val_size = int(len(train_dataset) * 0.1)
    train_size = len(train_dataset) - val_size
    train_subset, val_subset = random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    batch_size = data_cfg.get("batch_size", 32)
    num_workers = data_cfg.get("num_workers", 4)

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    # Model
    model = build_model(config)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")

    # Train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
    )

    results = trainer.train()
    logger.info(f"Training complete. Best val metric: {results['best_val_metric']:.4f}")


if __name__ == "__main__":
    main()
