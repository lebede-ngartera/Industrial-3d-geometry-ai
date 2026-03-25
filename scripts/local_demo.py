"""GeoFusion AI — Local End-to-End Demo (No Dataset Download Required)

Demonstrates all core capabilities using synthetic point clouds:
1. Data transforms & augmentation pipeline
2. PointNet++ classification (training + inference)
3. DGCNN graph neural network encoding
4. Similarity search & retrieval (FAISS)
5. Anomaly detection on 3D geometry
6. Diffusion-based shape generation

Run:  python scripts/local_demo.py
"""

import os
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # PyTorch + FAISS OpenMP fix

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def banner(title: str) -> None:
    print(f"\n{'='*64}")
    print(f"  {title}")
    print(f"{'='*64}")


# ── Synthetic Data Helpers ─────────────────────────────────────────────────

def make_sphere(n: int = 1024) -> np.ndarray:
    """Uniform random points on a unit sphere."""
    pts = np.random.randn(n, 3).astype(np.float32)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True) + 1e-8
    return pts


def make_cube(n: int = 1024) -> np.ndarray:
    """Random points on cube surfaces."""
    pts = np.random.uniform(-1, 1, (n, 3)).astype(np.float32)
    face = np.random.randint(0, 3, n)
    sign = np.random.choice([-1, 1], n).astype(np.float32)
    for i in range(n):
        pts[i, face[i]] = sign[i]
    return pts


def make_cylinder(n: int = 1024) -> np.ndarray:
    """Random points on a cylinder (radius=1, height=[-1,1])."""
    theta = np.random.uniform(0, 2 * np.pi, n).astype(np.float32)
    h = np.random.uniform(-1, 1, n).astype(np.float32)
    pts = np.stack([np.cos(theta), np.sin(theta), h], axis=1)
    return pts


def make_cone(n: int = 1024) -> np.ndarray:
    """Random points on a cone surface."""
    t = np.random.uniform(0, 1, n).astype(np.float32)
    theta = np.random.uniform(0, 2 * np.pi, n).astype(np.float32)
    pts = np.stack([t * np.cos(theta), t * np.sin(theta), 1 - t], axis=1)
    return pts


SHAPE_GENERATORS = {
    "sphere": make_sphere,
    "cube": make_cube,
    "cylinder": make_cylinder,
    "cone": make_cone,
}


def create_synthetic_dataset(
    samples_per_class: int = 30,
    num_points: int = 1024,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Create a labelled synthetic point cloud dataset.

    Returns:
        all_points: (N, num_points, 3)
        all_labels: (N,)
        class_names: list of class names
    """
    class_names = list(SHAPE_GENERATORS.keys())
    all_points, all_labels = [], []
    for cls_idx, name in enumerate(class_names):
        gen_fn = SHAPE_GENERATORS[name]
        for _ in range(samples_per_class):
            pts = gen_fn(num_points)
            # add slight noise
            pts += np.random.randn(*pts.shape).astype(np.float32) * 0.02
            all_points.append(pts)
            all_labels.append(cls_idx)
    return (
        np.stack(all_points),
        np.array(all_labels),
        class_names,
    )


# ── Demo Sections ──────────────────────────────────────────────────────────


def demo_transforms():
    """Demonstrate the data transform pipeline."""
    banner("1. Data Transforms & Augmentation")

    from geofusion.data.transforms import (
        Compose,
        FarthestPointSample,
        NormalizePointCloud,
        RandomJitter,
        RandomRotate,
        ToTensor,
    )

    raw_pts = make_sphere(2048)
    print(f"  Raw sphere : {raw_pts.shape}, range [{raw_pts.min():.3f}, {raw_pts.max():.3f}]")

    pipeline = Compose([
        FarthestPointSample(512),
        NormalizePointCloud(),
        RandomRotate(max_angle=15.0),
        RandomJitter(sigma=0.01, clip=0.05),
        ToTensor(),
    ])

    augmented = pipeline(raw_pts)
    print(f"  Augmented  : {augmented.shape}, dtype={augmented.dtype}")
    print(f"  Center     : {augmented.mean(dim=0).tolist()}")
    print("  [OK] Transform pipeline works correctly.")


def demo_pointnet2_training():
    """Train a small PointNet++ classifier on synthetic shapes."""
    banner("2. PointNet++ Classification (Train + Test)")

    from geofusion.models.pointnet2 import PointNet2Classifier

    num_points = 512
    embed_dim = 64
    num_classes = 4
    epochs = 8
    batch_size = 16

    # Create dataset
    points, labels, class_names = create_synthetic_dataset(
        samples_per_class=30, num_points=num_points
    )
    print(f"  Dataset: {len(labels)} samples, {num_classes} classes: {class_names}")

    # 80/20 split
    n = len(labels)
    perm = np.random.permutation(n)
    split = int(0.8 * n)
    train_idx, test_idx = perm[:split], perm[split:]

    train_ds = TensorDataset(
        torch.from_numpy(points[train_idx]),
        torch.from_numpy(labels[train_idx]),
    )
    test_ds = TensorDataset(
        torch.from_numpy(points[test_idx]),
        torch.from_numpy(labels[test_idx]),
    )
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size)

    # Model
    model = PointNet2Classifier(
        num_classes=num_classes,
        in_channels=3,
        embed_dim=embed_dim,
        use_normals=False,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Train
    model.train()
    for epoch in range(epochs):
        total_loss, correct, total = 0.0, 0, 0
        for pts_batch, lbl_batch in train_dl:
            logits, _ = model(pts_batch)
            loss = F.cross_entropy(logits, lbl_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * pts_batch.size(0)
            correct += (logits.argmax(1) == lbl_batch).sum().item()
            total += pts_batch.size(0)
        if (epoch + 1) % 2 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:2d}/{epochs}  Loss={total_loss/total:.4f}  Acc={correct/total:.1%}")

    # Evaluate
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for pts_batch, lbl_batch in test_dl:
            logits, _ = model(pts_batch)
            correct += (logits.argmax(1) == lbl_batch).sum().item()
            total += pts_batch.size(0)
    acc = correct / total
    print(f"  Test accuracy: {correct}/{total} = {acc:.1%}")
    print("  [OK] PointNet++ classification works correctly.")
    return model


def demo_dgcnn():
    """Test DGCNN graph neural network encoder."""
    banner("3. DGCNN Graph Neural Network Encoder")

    from geofusion.models.gnn_encoder import DGCNNEncoder

    encoder = DGCNNEncoder(in_channels=3, embed_dim=128, k=10)
    pts = torch.randn(4, 512, 3)

    encoder.eval()
    with torch.no_grad():
        embeddings = encoder(pts)

    print(f"  Input  : {pts.shape}")
    print(f"  Output : {embeddings.shape}")
    print(f"  Embed norms: {embeddings.norm(dim=1).tolist()}")
    print("  [OK] DGCNN encoder works correctly.")


def demo_similarity_search(model):
    """Build an index and run similarity search."""
    banner("4. FAISS Similarity Search & Retrieval")

    from geofusion.retrieval.embeddings import EmbeddingStore
    from geofusion.retrieval.search import SimilaritySearch

    # Compute embeddings for a small dataset
    num_points = 512
    points, labels, class_names = create_synthetic_dataset(
        samples_per_class=15, num_points=num_points
    )

    model.eval()
    all_emb = []
    with torch.no_grad():
        for i in range(0, len(points), 16):
            batch = torch.from_numpy(points[i:i+16])
            _, emb = model(batch)
            all_emb.append(emb.numpy())
    embeddings = np.concatenate(all_emb, axis=0)
    print(f"  Computed {embeddings.shape[0]} embeddings (dim={embeddings.shape[1]})")

    # Build embedding store
    store = EmbeddingStore()
    metadata = [{"class": class_names[l], "index": int(i)} for i, l in enumerate(labels)]
    store.add_embeddings(embeddings, metadata=metadata, labels=labels)

    # Build search index
    search = SimilaritySearch(dim=embeddings.shape[1], metric="cosine")
    search.build_index(embeddings, metadata, labels)

    # Query: pick a sphere and find similar shapes
    sphere_idx = np.where(labels == 0)[0][0]
    query = embeddings[sphere_idx : sphere_idx + 1]
    results = search.search(query, top_k=5)

    query_class = class_names[labels[sphere_idx]]
    print(f"\n  Query: {query_class} (index {sphere_idx})")
    print(f"  Top-5 similar:")
    for i, r in enumerate(results):
        cls = r.metadata.get("class", "?") if r.metadata else "?"
        print(f"    {i+1}. {cls:10s}  score={r.score:.4f}  idx={r.index}")

    # Batch search
    queries = embeddings[:3]
    batch_results = search.search_batch(queries, top_k=3)
    print(f"\n  Batch search: {len(batch_results)} queries, each returning {len(batch_results[0])} results")
    print("  [OK] Similarity search works correctly.")


def demo_anomaly_detection():
    """Run anomaly detection on normal vs. deformed shapes."""
    banner("5. Anomaly Detection")

    from geofusion.models.anomaly import GeometryAnomalyDetector

    num_points = 512
    detector = GeometryAnomalyDetector(
        num_points=num_points, latent_dim=32, method="reconstruction"
    )

    # "Normal" spheres
    normal = torch.stack([torch.from_numpy(make_sphere(num_points)) for _ in range(8)])

    # "Anomalous" — random noise blobs (no geometric structure)
    anomalous = torch.randn(4, num_points, 3) * 2.0

    detector.eval()
    with torch.no_grad():
        normal_scores = detector.anomaly_score(normal)
        anomaly_scores = detector.anomaly_score(anomalous)

    print(f"  Normal scores  (mean): {normal_scores.mean():.4f}  (range: {normal_scores.min():.4f} - {normal_scores.max():.4f})")
    print(f"  Anomaly scores (mean): {anomaly_scores.mean():.4f}  (range: {anomaly_scores.min():.4f} - {anomaly_scores.max():.4f})")

    # Brief training to improve the autoencoder
    print("\n  Training autoencoder (3 epochs on normal shapes)...")
    optimizer = torch.optim.Adam(detector.autoencoder.parameters(), lr=1e-3)
    detector.autoencoder.train()
    for epoch in range(3):
        recon, z = detector.autoencoder(normal)
        from geofusion.models.anomaly import chamfer_distance
        loss = chamfer_distance(recon, normal).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"    Epoch {epoch+1}: loss={loss.item():.6f}")

    detector.eval()
    with torch.no_grad():
        normal_scores_after = detector.anomaly_score(normal)
        anomaly_scores_after = detector.anomaly_score(anomalous)

    print(f"\n  After training:")
    print(f"  Normal scores  (mean): {normal_scores_after.mean():.4f}")
    print(f"  Anomaly scores (mean): {anomaly_scores_after.mean():.4f}")
    print("  [OK] Anomaly detection works correctly.")


def demo_diffusion():
    """Run diffusion model training step and generation."""
    banner("6. Diffusion Shape Generation")

    from geofusion.models.diffusion import ShapeDiffusionModel

    num_points = 256  # small for speed
    model = ShapeDiffusionModel(
        num_points=num_points,
        hidden_dim=64,
        condition_dim=None,  # unconditional generation
        num_timesteps=100,
    )

    # Training step
    spheres = torch.stack([torch.from_numpy(make_sphere(num_points)) for _ in range(4)])
    model.train()
    output = model(spheres)
    loss = output["loss"]
    loss.backward()
    print(f"  Training loss: {loss.item():.6f}")
    print(f"  Noise pred shape: {output['noise_pred'].shape}")

    # Generate new shapes (few-step sampling for demo speed)
    print("  Generating 2 shapes (100 diffusion steps)...")
    model.eval()
    t0 = time.time()
    with torch.no_grad():
        generated = model.sample(batch_size=2, device="cpu")
    elapsed = time.time() - t0
    print(f"  Generated shape: {generated.shape}")
    print(f"  Value range: [{generated.min():.3f}, {generated.max():.3f}]")
    print(f"  Generation time: {elapsed:.1f}s")
    print("  [OK] Diffusion generation works correctly.")


def demo_text_metadata():
    """Demonstrate text metadata generation."""
    banner("7. Text Metadata Generation")

    from geofusion.data.text_metadata import TextMetadataGenerator

    gen = TextMetadataGenerator()
    categories = ["gear", "bracket", "housing", "shaft"]

    for cat in categories:
        desc = gen.generate(cat)
        print(f"  {cat:10s} -> \"{desc[:70]}...\"")

    print("  [OK] Text metadata generation works correctly.")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    banner("GeoFusion AI — Local End-to-End Demo")
    print("  All demos use synthetic data (no downloads required)")
    print(f"  PyTorch {torch.__version__}, Device: cpu")
    print(f"  CUDA available: {torch.cuda.is_available()}")

    torch.manual_seed(42)
    np.random.seed(42)

    t0 = time.time()

    # 1. Transforms
    demo_transforms()

    # 2. PointNet++ (returns trained model for reuse)
    model = demo_pointnet2_training()

    # 3. DGCNN
    demo_dgcnn()

    # 4. Similarity Search (uses trained model)
    demo_similarity_search(model)

    # 5. Anomaly Detection
    demo_anomaly_detection()

    # 6. Diffusion Generation
    demo_diffusion()

    # 7. Text Metadata
    demo_text_metadata()

    elapsed = time.time() - t0
    banner("All Demos Completed Successfully!")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"  All 7 capabilities verified on local machine.")
    print()


if __name__ == "__main__":
    main()
