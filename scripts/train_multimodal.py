"""Train multimodal alignment model (geometry + text)."""

import argparse
import logging

import torch
import yaml
from torch.utils.data import DataLoader, random_split

from geofusion.data.datasets import ShapeNetDataset
from geofusion.data.text_metadata import TextMetadataGenerator
from geofusion.data.transforms import Compose, FarthestPointSample, NormalizePointCloud
from geofusion.models.multimodal import GeoFusionModel
from geofusion.models.pointnet2 import PointNet2Encoder
from geofusion.models.text_encoder import TextEncoder
from geofusion.training.trainer import Trainer

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Train multimodal alignment")
    parser.add_argument("--config", default="configs/multimodal.yaml")
    parser.add_argument("--geo-checkpoint", default=None, help="Pretrained geometry encoder")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = config.get("project", {}).get("device", "cuda")
    if args.device:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    seed = config.get("project", {}).get("seed", 42)
    torch.manual_seed(seed)

    # Data
    data_cfg = config.get("data", {})
    num_points = data_cfg.get("num_points", 2048)

    text_gen = TextMetadataGenerator(seed=seed)

    transform = Compose([
        FarthestPointSample(num_points),
        NormalizePointCloud(),
    ])

    dataset = ShapeNetDataset(
        data_root=data_cfg.get("data_root", "data/raw/shapenetcore_partanno_segmentation_benchmark_v0"),
        split="train",
        num_points=num_points,
        transform=transform,
        include_text=True,
        text_generator=text_gen,
    )

    # Split
    val_size = int(len(dataset) * 0.1)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    batch_size = data_cfg.get("batch_size", 64)

    def collate_fn(batch):
        points = torch.stack([b["points"] for b in batch])
        labels = torch.stack([b["label"] for b in batch])
        texts = [b["text"] for b in batch]
        categories = [b["category"] for b in batch]
        return {
            "points": points,
            "label": labels,
            "text": texts,
            "category": categories,
        }

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, collate_fn=collate_fn, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, collate_fn=collate_fn,
    )

    # Build model
    geo_cfg = config.get("geometry_encoder", {})
    text_cfg = config.get("text_encoder", {})
    mm_cfg = config.get("multimodal", {})

    embed_dim = geo_cfg.get("embed_dim", 256)

    geo_encoder = PointNet2Encoder(
        embed_dim=embed_dim,
        use_normals=False,  # ShapeNet typically xyz only
        dropout=geo_cfg.get("dropout", 0.3),
    )

    # Load pretrained geometry encoder if available
    if args.geo_checkpoint:
        checkpoint = torch.load(args.geo_checkpoint, map_location="cpu", weights_only=False)
        state = checkpoint.get("model_state_dict", checkpoint)
        # Extract encoder weights
        encoder_state = {
            k.replace("encoder.", ""): v
            for k, v in state.items()
            if k.startswith("encoder.")
        }
        if encoder_state:
            geo_encoder.load_state_dict(encoder_state, strict=False)
            logger.info("Loaded pretrained geometry encoder")

    text_encoder = TextEncoder(
        model_name=text_cfg.get("backbone", "sentence-transformers/all-MiniLM-L6-v2"),
        embed_dim=embed_dim,
        freeze_backbone=text_cfg.get("freeze_backbone", True),
    )

    num_categories = len(ShapeNetDataset.CATEGORY_MAP)

    model = GeoFusionModel(
        geometry_encoder=geo_encoder,
        text_encoder=text_encoder,
        embed_dim=embed_dim,
        temperature=mm_cfg.get("temperature", 0.07),
        num_classes=num_categories,
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {n_params:,}")

    # Train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
    )

    results = trainer.train()
    logger.info(f"Multimodal training complete. Best: {results['best_val_metric']:.4f}")


if __name__ == "__main__":
    main()
