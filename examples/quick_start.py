"""
GeoFusion AI Quick Start Example

Demonstrates the core pipeline: generate synthetic shapes, encode with PointNet++,
build a FAISS similarity index, and run a retrieval query. Runs entirely on CPU
with no external data downloads.
"""

import numpy as np
import torch
from geofusion.models.pointnet2 import PointNet2Encoder
from geofusion.retrieval.search import SimilaritySearch


def generate_sphere(n_points=2048):
    phi = np.random.uniform(0, 2 * np.pi, n_points)
    cos_theta = np.random.uniform(-1, 1, n_points)
    theta = np.arccos(cos_theta)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.stack([x, y, z], axis=1).astype(np.float32)


def generate_cube(n_points=2048):
    points = np.random.uniform(-1, 1, (n_points, 3)).astype(np.float32)
    return points


def generate_cylinder(n_points=2048):
    theta = np.random.uniform(0, 2 * np.pi, n_points)
    z = np.random.uniform(-1, 1, n_points)
    x = np.cos(theta)
    y = np.sin(theta)
    return np.stack([x, y, z], axis=1).astype(np.float32)


def main():
    print("GeoFusion AI Quick Start")
    print("=" * 50)

    # Generate synthetic data
    shapes = {
        "sphere": [generate_sphere() for _ in range(10)],
        "cube": [generate_cube() for _ in range(10)],
        "cylinder": [generate_cylinder() for _ in range(10)],
    }

    print(f"Generated {sum(len(v) for v in shapes.values())} synthetic shapes")

    # Initialize encoder
    encoder = PointNet2Encoder(input_channels=3, embed_dim=256)
    encoder.eval()
    print(f"PointNet++ encoder: {sum(p.numel() for p in encoder.parameters()):,} parameters")

    # Compute embeddings
    all_embeddings = []
    all_labels = []
    all_metadata = []

    with torch.no_grad():
        for label, shape_list in shapes.items():
            for i, pts in enumerate(shape_list):
                tensor = torch.from_numpy(pts).unsqueeze(0)
                emb = encoder(tensor)
                all_embeddings.append(emb.squeeze(0).numpy())
                all_labels.append(label)
                all_metadata.append({"category": label, "index": i})

    embeddings = np.stack(all_embeddings)
    print(f"Computed {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")

    # Build FAISS index
    search = SimilaritySearch(dim=256, index_type="Flat", metric="cosine")
    search.build_index(embeddings, all_metadata, all_labels)
    print("FAISS index built")

    # Query: find shapes similar to the first sphere
    query = embeddings[0:1]
    results = search.search(query, top_k=5)

    print("\nQuery: sphere[0]")
    print("Top 5 similar shapes:")
    for r in results:
        print(f"  {r.metadata['category']}[{r.metadata['index']}]  score={r.score:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
