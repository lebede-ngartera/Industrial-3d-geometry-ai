"""Interactive demo showcasing GeoFusion capabilities."""

import argparse
import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


def demo_shape_classification(model, dataset, device):
    """Demo: classify random shapes from the dataset."""
    print("\n" + "=" * 60)
    print("DEMO: Shape Classification")
    print("=" * 60)

    indices = np.random.choice(len(dataset), size=min(5, len(dataset)), replace=False)

    model.eval()
    with torch.no_grad():
        for idx in indices:
            sample = dataset[int(idx)]
            points = sample["points"].unsqueeze(0).to(device)
            true_label = sample["class_name"]

            logits, emb = model(points)
            pred_idx = logits.argmax(dim=1).item()
            pred_label = dataset.classes[pred_idx]
            confidence = torch.softmax(logits, dim=1).max().item()

            status = "✓" if pred_label == true_label else "✗"
            print(f"  {status} True: {true_label:15s} | Predicted: {pred_label:15s} | Confidence: {confidence:.3f}")


def demo_similarity_search(model, dataset, search_engine, device):
    """Demo: find similar parts."""
    print("\n" + "=" * 60)
    print("DEMO: Part Similarity Search")
    print("=" * 60)

    idx = np.random.randint(len(dataset))
    sample = dataset[idx]
    query_points = sample["points"].unsqueeze(0).to(device)
    query_class = sample["class_name"]

    model.eval()
    with torch.no_grad():
        _, query_emb = model(query_points)

    results = search_engine.search(query_emb.cpu().numpy(), top_k=5)

    print(f"\n  Query: {query_class} (index {idx})")
    print(f"  Top-5 similar parts:")
    for i, r in enumerate(results):
        meta = r.metadata or {}
        cls = meta.get("class_name", "?")
        print(f"    {i+1}. {cls:15s} | Score: {r.score:.4f} | Index: {r.index}")


def demo_text_search(model, retriever):
    """Demo: text-to-shape retrieval."""
    print("\n" + "=" * 60)
    print("DEMO: Text-to-Shape Search")
    print("=" * 60)

    queries = [
        "A compact aerodynamic body with swept wings",
        "Ergonomic seating structure with armrests",
        "Cylindrical vessel with handle attachment",
        "Flat horizontal surface supported by four legs",
    ]

    for query in queries:
        print(f"\n  Query: '{query}'")
        try:
            results = retriever.text_to_shape(query, top_k=3)
            for i, r in enumerate(results):
                meta = r.metadata or {}
                cls = meta.get("class_name", meta.get("category", "?"))
                print(f"    {i+1}. {cls:15s} | Score: {r.score:.4f}")
        except Exception as e:
            print(f"    (text search unavailable: {e})")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="GeoFusion Interactive Demo")
    parser.add_argument("--checkpoint", default="outputs/best_model.pt")
    parser.add_argument("--index", default="outputs/retrieval_index")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = args.device or config.get("project", {}).get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    print("=" * 60)
    print("  GeoFusion AI — Industrial Multimodal AI Demo")
    print("=" * 60)
    print(f"  Device: {device}")

    # Load model
    from geofusion.models.pointnet2 import PointNet2Classifier

    embed_dim = config.get("geometry_encoder", {}).get("embed_dim", 256)
    model = PointNet2Classifier(num_classes=40, embed_dim=embed_dim)

    try:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"  Loaded checkpoint: {args.checkpoint}")
    except FileNotFoundError:
        print("  No checkpoint found — running with random weights (demo only)")

    model = model.to(device)

    # Load dataset
    from geofusion.data.datasets import ModelNet40Dataset
    from geofusion.data.transforms import Compose, FarthestPointSample, NormalizePointCloud

    num_points = config.get("data", {}).get("num_points", 2048)
    transform = Compose([FarthestPointSample(num_points), NormalizePointCloud()])

    data_root = config.get("data", {}).get("data_root", "data/raw/modelnet40_normal_resampled")

    try:
        dataset = ModelNet40Dataset(data_root=data_root, split="test", transform=transform)
        print(f"  Dataset: {len(dataset)} samples")
    except Exception:
        print("  Dataset not found — run `python scripts/download_data.py` first")
        return

    # Demo 1: Classification
    demo_shape_classification(model, dataset, device)

    # Demo 2: Similarity search
    try:
        from geofusion.retrieval.search import SimilaritySearch
        search = SimilaritySearch(dim=embed_dim, metric="cosine")
        search.load(args.index)
        demo_similarity_search(model, dataset, search, device)
    except Exception:
        print("\n  Similarity search index not found — run build_index.py first")

    print("\n" + "=" * 60)
    print("  Demo complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
