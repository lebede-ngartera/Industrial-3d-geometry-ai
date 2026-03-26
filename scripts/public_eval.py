"""Run a small public-safe evaluation for GeoFusion AI.

This script intentionally uses a lightweight synthetic subset so the public
repository can report honest, reproducible metrics without exposing private
datasets or internal benchmark assets.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from geofusion.models.anomaly import GeometryAnomalyDetector
from geofusion.models.pointnet2 import PointNet2Classifier
from geofusion.training.metrics import (
    compute_accuracy,
    compute_cross_modal_metrics,
    compute_retrieval_metrics,
)
from geofusion.workflows.anomaly_detection import AnomalyDetectionWorkflow


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_sphere(n: int, rng: np.random.Generator) -> np.ndarray:
    pts = rng.normal(size=(n, 3)).astype(np.float32)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True) + 1e-8
    return pts


def make_cube(n: int, rng: np.random.Generator) -> np.ndarray:
    pts = rng.uniform(-1, 1, size=(n, 3)).astype(np.float32)
    face = rng.integers(0, 3, size=n)
    sign = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=n)
    pts[np.arange(n), face] = sign
    return pts


def make_cylinder(n: int, rng: np.random.Generator) -> np.ndarray:
    theta = rng.uniform(0, 2 * np.pi, size=n).astype(np.float32)
    h = rng.uniform(-1, 1, size=n).astype(np.float32)
    return np.stack([np.cos(theta), np.sin(theta), h], axis=1)


def make_cone(n: int, rng: np.random.Generator) -> np.ndarray:
    t = rng.uniform(0, 1, size=n).astype(np.float32)
    theta = rng.uniform(0, 2 * np.pi, size=n).astype(np.float32)
    return np.stack([t * np.cos(theta), t * np.sin(theta), 1 - t], axis=1)


def make_torus(n: int, rng: np.random.Generator, major: float = 1.0, minor: float = 0.3) -> np.ndarray:
    theta = rng.uniform(0, 2 * np.pi, size=n).astype(np.float32)
    phi = rng.uniform(0, 2 * np.pi, size=n).astype(np.float32)
    x = (major + minor * np.cos(phi)) * np.cos(theta)
    y = (major + minor * np.cos(phi)) * np.sin(theta)
    z = minor * np.sin(phi)
    return np.stack([x, y, z], axis=1)


SHAPES = {
    "Sphere": make_sphere,
    "Cube": make_cube,
    "Cylinder": make_cylinder,
    "Cone": make_cone,
    "Torus": make_torus,
}


TEXT_TEMPLATES = {
    "Sphere": "round symmetric part with continuous surface",
    "Cube": "boxy part with flat faces and sharp transitions",
    "Cylinder": "cylindrical part with circular profile and straight body",
    "Cone": "tapered conical part narrowing to one end",
    "Torus": "ring-shaped part with hollow circular center",
}


class SyntheticShapeDataset(Dataset):
    def __init__(self, items: list[dict[str, object]]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, object]:
        item = self.items[index]
        return {
            "points": torch.from_numpy(item["points"]),
            "label": torch.tensor(item["label"], dtype=torch.long),
            "text": item["text"],
            "filename": item["filename"],
            "class_name": item["class_name"],
        }


def build_public_subset(samples_per_class: int, num_points: int, seed: int) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rng = np.random.default_rng(seed)
    train_items: list[dict[str, object]] = []
    test_items: list[dict[str, object]] = []

    for label, (class_name, generator) in enumerate(SHAPES.items()):
        for sample_index in range(samples_per_class):
            points = generator(num_points, rng).astype(np.float32)
            points += rng.normal(0.0, 0.02, size=points.shape).astype(np.float32)
            item = {
                "points": points,
                "label": label,
                "text": TEXT_TEMPLATES[class_name],
                "filename": f"{class_name.lower()}_{sample_index:03d}.npy",
                "class_name": class_name,
            }

            if sample_index < max(2, int(samples_per_class * 0.8)):
                train_items.append(item)
            else:
                test_items.append(item)

    return train_items, test_items


def train_classifier(
    model: PointNet2Classifier,
    dataloader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()

    for _ in range(epochs):
        for batch in dataloader:
            points = batch["points"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits, _ = model(points)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()


def build_vocabulary(texts: list[str]) -> dict[str, int]:
    vocabulary: dict[str, int] = {}
    for text in texts:
        for token in re.findall(r"[a-z]+", text.lower()):
            if token not in vocabulary:
                vocabulary[token] = len(vocabulary)
    return vocabulary


def encode_texts(texts: list[str], vocabulary: dict[str, int]) -> torch.Tensor:
    features = np.zeros((len(texts), len(vocabulary)), dtype=np.float32)
    for row_index, text in enumerate(texts):
        for token in re.findall(r"[a-z]+", text.lower()):
            if token in vocabulary:
                features[row_index, vocabulary[token]] += 1.0
    return torch.from_numpy(features)


def fit_text_projection(
    train_text_features: torch.Tensor,
    train_geo_embeddings: torch.Tensor,
    ridge: float = 1e-3,
) -> torch.Tensor:
    x = train_text_features.float()
    y = train_geo_embeddings.float()
    xtx = x.T @ x
    identity = torch.eye(xtx.shape[0], dtype=xtx.dtype)
    projection = torch.linalg.solve(xtx + ridge * identity, x.T @ y)
    return projection


def project_text_features(text_features: torch.Tensor, projection: torch.Tensor) -> torch.Tensor:
    return text_features.float() @ projection


@torch.no_grad()
def collect_logits_and_embeddings(
    model: PointNet2Classifier,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()
    all_logits = []
    all_embeddings = []
    all_labels = []

    for batch in dataloader:
        points = batch["points"].to(device)
        labels = batch["label"]
        logits, embeddings = model(points)
        all_logits.append(logits.cpu())
        all_embeddings.append(embeddings.cpu())
        all_labels.append(labels.cpu())

    return (
        torch.cat(all_logits, dim=0),
        torch.cat(all_embeddings, dim=0),
        torch.cat(all_labels, dim=0),
    )


def train_anomaly_detector(
    detector: GeometryAnomalyDetector,
    dataloader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
) -> None:
    optimizer = torch.optim.Adam(detector.parameters(), lr=learning_rate)
    detector.train()

    for _ in range(epochs):
        for batch in dataloader:
            points = batch["points"].to(device)
            optimizer.zero_grad()
            loss = detector(points)["loss"]
            loss.backward()
            optimizer.step()


def make_loader(items: list[dict[str, object]], batch_size: int, shuffle: bool = False) -> DataLoader:
    return DataLoader(SyntheticShapeDataset(items), batch_size=batch_size, shuffle=shuffle)


def write_markdown(results: dict[str, object], output_path: Path) -> None:
    lines = [
        "# Public Metrics",
        "",
        "These metrics are generated from the reproducible synthetic public subset used by `scripts/public_eval.py`.",
        "",
        "## Evaluation Setup",
        "",
        "- Script: `python scripts/public_eval.py`",
        "- Seed: `42`",
        "- Geometry classes: `Sphere`, `Cube`, `Cylinder`, `Cone`, `Torus`",
        "- Samples per class: `10`",
        "- Public split generated by script: `8 train / 2 test` per class",
        "- Public evaluation size: `40 train / 10 test`",
        "- Points per sample: `512`",
        "- PointNet++ training budget: `3 epochs`",
        "- Anomaly detector calibration budget: `4 epochs`",
        "",
        "These numbers are public sanity metrics for a reproducible synthetic subset. They validate the executable workflow surface and shared-representation plumbing; they are not a claim of industrial CAD benchmark saturation.",
        "",
        "| Area | Metric | Value | Notes |",
        "| ---- | ------ | ----- | ----- |",
        f"| Classification | Top-1 accuracy | {results['classification']['top1']:.4f} | Synthetic public subset |",
        f"| Classification | Top-5 accuracy | {results['classification']['top5']:.4f} | Synthetic public subset |",
        f"| Retrieval | Recall@1 | {results['retrieval']['recall@1']:.4f} | Test queries vs train gallery |",
        f"| Retrieval | Recall@5 | {results['retrieval']['recall@5']:.4f} | Test queries vs train gallery |",
        f"| Retrieval | Recall@10 | {results['retrieval']['recall@10']:.4f} | Test queries vs train gallery |",
        f"| Retrieval | Precision@5 | {results['retrieval']['precision@5']:.4f} | Test queries vs train gallery |",
        f"| Retrieval | mAP | {results['retrieval']['mAP']:.4f} | Test queries vs train gallery |",
        f"| Cross-modal | G2T Recall@5 | {results['cross_modal']['g2t_recall@5']:.4f} | Test geometry vs projected text |",
        f"| Cross-modal | T2G Recall@5 | {results['cross_modal']['t2g_recall@5']:.4f} | Test text vs geometry |",
        f"| Anomaly workflow | Warning threshold | {results['anomaly']['warning_threshold']:.6f} | Calibrated on normal subset |",
        f"| Anomaly workflow | Critical threshold | {results['anomaly']['critical_threshold']:.6f} | Calibrated on normal subset |",
        f"| Anomaly workflow | Warning or higher count | {results['anomaly']['num_flagged_warning_or_higher']} | Mixed public subset |",
        f"| Anomaly workflow | Critical count | {results['anomaly']['num_flagged_critical']} | Mixed public subset |",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run public-safe evaluation for GeoFusion AI")
    parser.add_argument("--samples-per-class", type=int, default=10)
    parser.add_argument("--num-points", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--anomaly-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", type=Path, default=Path("results/public_metrics.json"))
    parser.add_argument("--output-markdown", type=Path, default=Path("results/public_metrics.md"))
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_items, test_items = build_public_subset(
        samples_per_class=args.samples_per_class,
        num_points=args.num_points,
        seed=args.seed,
    )

    train_loader = make_loader(train_items, batch_size=args.batch_size, shuffle=True)
    train_eval_loader = make_loader(train_items, batch_size=args.batch_size, shuffle=False)
    test_loader = make_loader(test_items, batch_size=args.batch_size, shuffle=False)

    classifier = PointNet2Classifier(
        num_classes=len(SHAPES),
        in_channels=3,
        embed_dim=64,
        use_normals=False,
        dropout=0.2,
    ).to(device)

    train_classifier(
        model=classifier,
        dataloader=train_loader,
        device=device,
        epochs=args.epochs,
        learning_rate=1e-3,
    )

    test_logits, test_embeddings, test_labels = collect_logits_and_embeddings(classifier, test_loader, device)
    _, train_embeddings, train_labels = collect_logits_and_embeddings(classifier, train_eval_loader, device)

    train_texts = [item["text"] for item in train_items]
    test_texts = [item["text"] for item in test_items]
    vocabulary = build_vocabulary(train_texts + test_texts)
    train_text_features = encode_texts(train_texts, vocabulary)
    test_text_features = encode_texts(test_texts, vocabulary)
    text_projection = fit_text_projection(train_text_features, train_embeddings)
    test_text_embeddings = project_text_features(test_text_features, text_projection)

    classification = compute_accuracy(test_logits, test_labels, topk=(1, 5))
    retrieval = compute_retrieval_metrics(
        query_embeddings=test_embeddings,
        gallery_embeddings=train_embeddings,
        query_labels=test_labels,
        gallery_labels=train_labels,
        top_k=[1, 5, 10],
    )
    cross_modal = compute_cross_modal_metrics(
        geo_embeddings=test_embeddings,
        text_embeddings=test_text_embeddings,
        labels=test_labels,
        top_k=[1, 5, 10],
    )

    normal_train_items = [item for item in train_items if item["class_name"] == "Sphere"]
    normal_val_items = [item for item in test_items if item["class_name"] == "Sphere"]
    mixed_items = normal_val_items + [item for item in test_items if item["class_name"] in {"Cube", "Cone"}]

    anomaly_detector = GeometryAnomalyDetector(
        num_points=args.num_points,
        latent_dim=32,
        method="reconstruction",
    ).to(device)

    train_anomaly_detector(
        detector=anomaly_detector,
        dataloader=make_loader(normal_train_items, batch_size=min(4, args.batch_size), shuffle=True),
        device=device,
        epochs=args.anomaly_epochs,
        learning_rate=1e-3,
    )

    anomaly_workflow = AnomalyDetectionWorkflow(
        detector=anomaly_detector,
        device=str(device),
        warning_percentile=90.0,
        critical_percentile=99.0,
    )
    thresholds = anomaly_workflow.calibrate(make_loader(normal_val_items, batch_size=min(4, args.batch_size), shuffle=False))
    reports = anomaly_workflow.batch_analyze(make_loader(mixed_items, batch_size=min(4, args.batch_size), shuffle=False))

    results = {
        "dataset": {
            "name": "public_subset",
            "source_type": "synthetic_public_safe",
            "num_train_samples": len(train_items),
            "num_test_samples": len(test_items),
            "num_classes": len(SHAPES),
            "num_points": args.num_points,
        },
        "classification": {
            "top1": classification["top1"],
            "top5": classification["top5"],
        },
        "retrieval": {
            "recall@1": retrieval["recall@1"],
            "recall@5": retrieval["recall@5"],
            "recall@10": retrieval["recall@10"],
            "precision@5": retrieval["precision@5"],
            "mAP": retrieval["mAP"],
        },
        "cross_modal": {
            "g2t_recall@1": cross_modal["g2t_recall@1"],
            "g2t_recall@5": cross_modal["g2t_recall@5"],
            "g2t_recall@10": cross_modal["g2t_recall@10"],
            "g2t_mAP": cross_modal["g2t_mAP"],
            "t2g_recall@1": cross_modal["t2g_recall@1"],
            "t2g_recall@5": cross_modal["t2g_recall@5"],
            "t2g_recall@10": cross_modal["t2g_recall@10"],
            "t2g_mAP": cross_modal["t2g_mAP"],
            "text_embedding_note": "Linear text-projection baseline fitted on synthetic paired text templates",
        },
        "anomaly": {
            "warning_threshold": thresholds["warning_threshold"],
            "critical_threshold": thresholds["critical_threshold"],
            "mean_normal_score": thresholds["mean_normal_score"],
            "std_normal_score": thresholds["std_normal_score"],
            "num_flagged_warning_or_higher": sum(r.risk_level in {"warning", "critical"} for r in reports),
            "num_flagged_critical": sum(r.risk_level == "critical" for r in reports),
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_markdown(results, args.output_markdown)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()