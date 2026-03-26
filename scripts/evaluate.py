"""Evaluate trained model with comprehensive metrics."""

import argparse
import json
import logging
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from geofusion.data.datasets import ModelNet40Dataset
from geofusion.data.transforms import Compose, FarthestPointSample, NormalizePointCloud
from geofusion.models.pointnet2 import PointNet2Classifier
from geofusion.training.metrics import compute_accuracy, compute_retrieval_metrics

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Evaluate GeoFusion model")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", default="outputs/eval_results.json")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = args.device or config.get("project", {}).get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # Load model
    embed_dim = config.get("geometry_encoder", {}).get("embed_dim", 256)
    model = PointNet2Classifier(num_classes=40, embed_dim=embed_dim)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Load test data
    num_points = config.get("data", {}).get("num_points", 2048)
    transform = Compose([FarthestPointSample(num_points), NormalizePointCloud()])

    dataset = ModelNet40Dataset(
        data_root=config.get("data", {}).get("data_root", "data/raw/modelnet40_normal_resampled"),
        split="test",
        num_points=num_points,
        transform=transform,
    )
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

    # Evaluate
    all_logits = []
    all_labels = []
    all_embeddings = []

    with torch.no_grad():
        for batch in dataloader:
            points = batch["points"].to(device)
            labels = batch["label"]

            logits, embeddings = model(points)
            all_logits.append(logits.cpu())
            all_labels.append(labels)
            all_embeddings.append(embeddings.cpu())

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    embeddings = torch.cat(all_embeddings)

    # Classification metrics
    acc = compute_accuracy(logits, labels, topk=(1, 5))
    logger.info(f"Top-1 Accuracy: {acc['top1']:.4f}")
    logger.info(f"Top-5 Accuracy: {acc['top5']:.4f}")

    # Retrieval metrics
    ret_metrics = compute_retrieval_metrics(
        embeddings, embeddings, labels, labels, top_k=[1, 5, 10]
    )
    for k, v in ret_metrics.items():
        logger.info(f"{k}: {v:.4f}")

    # Per-class accuracy
    num_classes = logits.shape[1]
    per_class_acc = {}
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() > 0:
            class_acc = (logits[mask].argmax(dim=1) == c).float().mean().item()
            class_name = dataset.classes[c]
            per_class_acc[class_name] = class_acc

    # Save results
    results = {
        "classification": acc,
        "retrieval": ret_metrics,
        "per_class_accuracy": per_class_acc,
        "num_samples": len(dataset),
        "checkpoint": args.checkpoint,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
