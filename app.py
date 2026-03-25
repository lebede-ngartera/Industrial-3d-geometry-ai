"""GeoFusion AI — Interactive Streamlit Dashboard

Run:  streamlit run app.py
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import numpy as np
import torch
import torch.nn.functional as F
import streamlit as st
import plotly.graph_objects as go


# ── Page Config ────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="GeoFusion AI",
    page_icon="🔧",
    layout="wide",
)

# ── Synthetic Shape Generators ─────────────────────────────────────────────

def make_sphere(n: int = 1024) -> np.ndarray:
    pts = np.random.randn(n, 3).astype(np.float32)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True) + 1e-8
    return pts

def make_cube(n: int = 1024) -> np.ndarray:
    pts = np.random.uniform(-1, 1, (n, 3)).astype(np.float32)
    face = np.random.randint(0, 3, n)
    sign = np.random.choice([-1, 1], n).astype(np.float32)
    for i in range(n):
        pts[i, face[i]] = sign[i]
    return pts

def make_cylinder(n: int = 1024) -> np.ndarray:
    theta = np.random.uniform(0, 2 * np.pi, n).astype(np.float32)
    h = np.random.uniform(-1, 1, n).astype(np.float32)
    return np.stack([np.cos(theta), np.sin(theta), h], axis=1)

def make_cone(n: int = 1024) -> np.ndarray:
    t = np.random.uniform(0, 1, n).astype(np.float32)
    theta = np.random.uniform(0, 2 * np.pi, n).astype(np.float32)
    return np.stack([t * np.cos(theta), t * np.sin(theta), 1 - t], axis=1)

def make_torus(n: int = 1024, R: float = 1.0, r: float = 0.3) -> np.ndarray:
    theta = np.random.uniform(0, 2 * np.pi, n).astype(np.float32)
    phi = np.random.uniform(0, 2 * np.pi, n).astype(np.float32)
    x = (R + r * np.cos(phi)) * np.cos(theta)
    y = (R + r * np.cos(phi)) * np.sin(theta)
    z = r * np.sin(phi)
    return np.stack([x, y, z], axis=1)

SHAPES = {
    "Sphere": make_sphere,
    "Cube": make_cube,
    "Cylinder": make_cylinder,
    "Cone": make_cone,
    "Torus": make_torus,
}


def plot_point_cloud(pts: np.ndarray, title: str = "", color: str | np.ndarray | None = None, size: int = 2) -> go.Figure:
    """Create a 3D scatter plot of a point cloud."""
    marker_kwargs = dict(size=size, opacity=0.8)
    if color is not None:
        if isinstance(color, np.ndarray):
            marker_kwargs["color"] = color
            marker_kwargs["colorscale"] = "Viridis"
        else:
            marker_kwargs["color"] = color
    else:
        marker_kwargs["color"] = pts[:, 2]
        marker_kwargs["colorscale"] = "Viridis"

    fig = go.Figure(data=[go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
        mode="markers",
        marker=marker_kwargs,
    )])
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        height=450,
    )
    return fig


# ── Cached Model Loaders ──────────────────────────────────────────────────

@st.cache_resource
def load_pointnet2(num_classes: int, embed_dim: int):
    from geofusion.models.pointnet2 import PointNet2Classifier
    return PointNet2Classifier(
        num_classes=num_classes, in_channels=3,
        embed_dim=embed_dim, use_normals=False,
    )

@st.cache_resource
def load_dgcnn(embed_dim: int, k: int):
    from geofusion.models.gnn_encoder import DGCNNEncoder
    return DGCNNEncoder(in_channels=3, embed_dim=embed_dim, k=k)

@st.cache_resource
def load_anomaly_detector(num_points: int, latent_dim: int):
    from geofusion.models.anomaly import GeometryAnomalyDetector
    return GeometryAnomalyDetector(
        num_points=num_points, latent_dim=latent_dim, method="reconstruction",
    )

@st.cache_resource
def load_diffusion(num_points: int, hidden_dim: int, num_timesteps: int):
    from geofusion.models.diffusion import ShapeDiffusionModel
    return ShapeDiffusionModel(
        num_points=num_points, hidden_dim=hidden_dim,
        condition_dim=None, num_timesteps=num_timesteps,
    )


# ── Sidebar ────────────────────────────────────────────────────────────────

st.sidebar.title("GeoFusion AI")
st.sidebar.markdown("**Industrial 3D Part Intelligence**")

page = st.sidebar.radio("Module", [
    "Overview",
    "Data Transforms",
    "PointNet++ Classification",
    "DGCNN Encoder",
    "Similarity Search",
    "Anomaly Detection",
    "Shape Generation",
])

st.sidebar.markdown("---")
st.sidebar.caption(f"PyTorch {torch.__version__}")
st.sidebar.caption(f"CUDA: {'Yes' if torch.cuda.is_available() else 'No (CPU)'}")


# ── Pages ──────────────────────────────────────────────────────────────────

if page == "Overview":
    st.title("GeoFusion AI — Interactive Dashboard")
    st.markdown("""
    This dashboard demonstrates all core capabilities of the **GeoFusion AI** system
    for 3D CAD/engineering part intelligence using Graph Neural Networks
    and multimodal AI.

    **Select a module** from the sidebar to explore:

    | Module | Description |
    |--------|-------------|
    | **Data Transforms** | Point cloud augmentation pipeline |
    | **PointNet++ Classification** | Train & test 3D shape classifier |
    | **DGCNN Encoder** | Graph neural network feature extraction |
    | **Similarity Search** | FAISS-powered part retrieval |
    | **Anomaly Detection** | Detect unusual geometry |
    | **Shape Generation** | Diffusion-based 3D point cloud generation |

    All demos use **synthetic point clouds** — no dataset download needed.
    """)

    # Show sample shapes
    st.subheader("Sample Synthetic Shapes")
    cols = st.columns(len(SHAPES))
    for col, (name, gen_fn) in zip(cols, SHAPES.items()):
        pts = gen_fn(512)
        col.plotly_chart(plot_point_cloud(pts, title=name, size=1), use_container_width=True)


elif page == "Data Transforms":
    st.title("Data Transforms & Augmentation")

    from geofusion.data.transforms import (
        Compose, FarthestPointSample, NormalizePointCloud,
        RandomRotate, RandomJitter, RandomScale, RandomFlip, ToTensor,
    )

    col1, col2 = st.columns(2)
    with col1:
        shape = st.selectbox("Input shape", list(SHAPES.keys()))
        num_points_raw = st.slider("Raw points", 256, 4096, 2048)
    with col2:
        sample_to = st.slider("Downsample to", 64, 2048, 512)
        jitter_sigma = st.slider("Jitter sigma", 0.0, 0.05, 0.01, step=0.005)
        rotate_angle = st.slider("Max rotation (deg)", 0, 180, 15)

    raw = SHAPES[shape](num_points_raw)

    pipeline = Compose([
        FarthestPointSample(sample_to),
        NormalizePointCloud(),
        RandomRotate(max_angle=float(rotate_angle)),
        RandomJitter(sigma=jitter_sigma, clip=jitter_sigma * 5),
        ToTensor(),
    ])

    augmented = pipeline(raw).numpy()

    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(plot_point_cloud(raw, f"Raw {shape} ({num_points_raw} pts)"), use_container_width=True)
    with col_b:
        st.plotly_chart(plot_point_cloud(augmented, f"Augmented ({sample_to} pts)"), use_container_width=True)

    st.info(f"Pipeline: FPS({sample_to}) → Normalize → Rotate(±{rotate_angle}°) → Jitter(σ={jitter_sigma})")


elif page == "PointNet++ Classification":
    st.title("PointNet++ Shape Classification")

    col1, col2, col3 = st.columns(3)
    with col1:
        samples_per_class = st.slider("Samples per class", 10, 80, 30)
        num_points = st.selectbox("Points per sample", [256, 512, 1024], index=1)
    with col2:
        embed_dim = st.selectbox("Embedding dim", [32, 64, 128], index=1)
        epochs = st.slider("Training epochs", 2, 20, 8)
    with col3:
        lr = st.select_slider("Learning rate", [1e-4, 5e-4, 1e-3, 5e-3], value=1e-3)
        batch_size = st.selectbox("Batch size", [8, 16, 32], index=1)

    class_names = list(SHAPES.keys())
    num_classes = len(class_names)

    if st.button("Train & Evaluate", type="primary"):
        # Create dataset
        all_pts, all_labels = [], []
        for idx, name in enumerate(class_names):
            for _ in range(samples_per_class):
                pts = SHAPES[name](num_points) + np.random.randn(num_points, 3).astype(np.float32) * 0.02
                all_pts.append(pts)
                all_labels.append(idx)
        all_pts = np.stack(all_pts)
        all_labels = np.array(all_labels)

        perm = np.random.permutation(len(all_labels))
        split = int(0.8 * len(all_labels))
        train_idx, test_idx = perm[:split], perm[split:]

        from torch.utils.data import DataLoader, TensorDataset
        train_dl = DataLoader(
            TensorDataset(torch.from_numpy(all_pts[train_idx]), torch.from_numpy(all_labels[train_idx])),
            batch_size=batch_size, shuffle=True,
        )
        test_dl = DataLoader(
            TensorDataset(torch.from_numpy(all_pts[test_idx]), torch.from_numpy(all_labels[test_idx])),
            batch_size=batch_size,
        )

        model = load_pointnet2(num_classes, embed_dim)
        # Reset weights for fresh training
        for m in model.modules():
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        progress = st.progress(0.0)
        metrics_area = st.empty()
        train_losses, train_accs = [], []

        model.train()
        for epoch in range(epochs):
            total_loss, correct, total = 0.0, 0, 0
            for pts_b, lbl_b in train_dl:
                logits, _ = model(pts_b)
                loss = F.cross_entropy(logits, lbl_b)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * pts_b.size(0)
                correct += (logits.argmax(1) == lbl_b).sum().item()
                total += pts_b.size(0)

            epoch_loss = total_loss / total
            epoch_acc = correct / total
            train_losses.append(epoch_loss)
            train_accs.append(epoch_acc)
            progress.progress((epoch + 1) / epochs)
            metrics_area.text(f"Epoch {epoch+1}/{epochs}  Loss={epoch_loss:.4f}  Acc={epoch_acc:.1%}")

        # Evaluate
        model.eval()
        correct, total = 0, 0
        per_class_correct = np.zeros(num_classes)
        per_class_total = np.zeros(num_classes)
        with torch.no_grad():
            for pts_b, lbl_b in test_dl:
                logits, _ = model(pts_b)
                preds = logits.argmax(1)
                correct += (preds == lbl_b).sum().item()
                total += pts_b.size(0)
                for c in range(num_classes):
                    mask = lbl_b == c
                    per_class_correct[c] += (preds[mask] == c).sum().item()
                    per_class_total[c] += mask.sum().item()

        test_acc = correct / total
        st.success(f"Test Accuracy: **{test_acc:.1%}** ({correct}/{total})")

        # Charts
        col_a, col_b = st.columns(2)
        with col_a:
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=train_losses, mode="lines+markers", name="Loss"))
            fig.update_layout(title="Training Loss", xaxis_title="Epoch", yaxis_title="Loss", height=300)
            st.plotly_chart(fig, use_container_width=True)
        with col_b:
            per_class_acc = per_class_correct / np.maximum(per_class_total, 1)
            fig = go.Figure(data=[go.Bar(x=class_names, y=per_class_acc, marker_color="steelblue")])
            fig.update_layout(title="Per-Class Test Accuracy", yaxis_title="Accuracy", height=300, yaxis_range=[0, 1.05])
            st.plotly_chart(fig, use_container_width=True)

        # Store model in session state for reuse
        st.session_state["trained_model"] = model
        st.session_state["embed_dim"] = embed_dim


elif page == "DGCNN Encoder":
    st.title("DGCNN Graph Neural Network Encoder")

    col1, col2 = st.columns(2)
    with col1:
        k = st.slider("k-NN neighbors", 5, 40, 20)
        dgcnn_dim = st.selectbox("Embedding dim", [64, 128, 256], index=1)
    with col2:
        shape = st.selectbox("Shape to encode", list(SHAPES.keys()))
        n_pts = st.slider("Points", 128, 1024, 512)

    if st.button("Encode Shape", type="primary"):
        encoder = load_dgcnn(dgcnn_dim, k)
        pts = SHAPES[shape](n_pts)
        pts_tensor = torch.from_numpy(pts).unsqueeze(0)

        encoder.eval()
        with torch.no_grad():
            emb = encoder(pts_tensor)

        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.plotly_chart(plot_point_cloud(pts, f"{shape} ({n_pts} pts)"), use_container_width=True)
        with col_b:
            st.subheader("Embedding Vector")
            emb_np = emb[0].numpy()
            fig = go.Figure(data=[go.Bar(y=emb_np, marker_color="teal")])
            fig.update_layout(title=f"Embedding (dim={dgcnn_dim})", height=350,
                              xaxis_title="Dimension", yaxis_title="Value")
            st.plotly_chart(fig, use_container_width=True)
            st.metric("L2 Norm", f"{np.linalg.norm(emb_np):.4f}")

        # Encode all shapes and show similarity matrix
        st.subheader("Inter-Shape Similarity Matrix")
        all_embs = {}
        encoder.eval()
        with torch.no_grad():
            for sname, gen_fn in SHAPES.items():
                sp = gen_fn(n_pts)
                e = encoder(torch.from_numpy(sp).unsqueeze(0))
                all_embs[sname] = e[0].numpy()

        names = list(all_embs.keys())
        mat = np.zeros((len(names), len(names)))
        for i, n1 in enumerate(names):
            for j, n2 in enumerate(names):
                v1 = all_embs[n1] / (np.linalg.norm(all_embs[n1]) + 1e-8)
                v2 = all_embs[n2] / (np.linalg.norm(all_embs[n2]) + 1e-8)
                mat[i, j] = np.dot(v1, v2)

        fig = go.Figure(data=go.Heatmap(z=mat, x=names, y=names, colorscale="RdBu", zmin=-1, zmax=1,
                                         text=np.round(mat, 3), texttemplate="%{text}"))
        fig.update_layout(height=400, title="Cosine Similarity (random weights)")
        st.plotly_chart(fig, use_container_width=True)


elif page == "Similarity Search":
    st.title("FAISS Similarity Search & Retrieval")

    from geofusion.retrieval.search import SimilaritySearch
    from geofusion.models.gnn_encoder import DGCNNEncoder

    col1, col2 = st.columns(2)
    with col1:
        n_samples = st.slider("Samples per class", 10, 40, 20)
        n_pts = st.selectbox("Points per sample", [256, 512], index=1)
    with col2:
        query_shape = st.selectbox("Query shape", list(SHAPES.keys()))
        top_k = st.slider("Top-K results", 3, 15, 5)

    if st.button("Build Index & Search", type="primary"):
        with st.spinner("Computing embeddings..."):
            encoder = load_dgcnn(128, 20)
            encoder.eval()

            all_pts, all_labels, metadata = [], [], []
            class_names = list(SHAPES.keys())
            for idx, name in enumerate(class_names):
                for s in range(n_samples):
                    pts = SHAPES[name](n_pts) + np.random.randn(n_pts, 3).astype(np.float32) * 0.02
                    all_pts.append(pts)
                    all_labels.append(idx)
                    metadata.append({"class": name, "sample": s})
            all_pts = np.stack(all_pts)
            all_labels = np.array(all_labels)

            embs = []
            with torch.no_grad():
                for i in range(0, len(all_pts), 16):
                    batch = torch.from_numpy(all_pts[i:i+16])
                    embs.append(encoder(batch).numpy())
            embeddings = np.concatenate(embs, axis=0)

        with st.spinner("Building FAISS index..."):
            search = SimilaritySearch(dim=embeddings.shape[1], metric="cosine")
            search.build_index(embeddings, metadata, all_labels)

        # Query
        query_pts = SHAPES[query_shape](n_pts)
        query_tensor = torch.from_numpy(query_pts).unsqueeze(0)
        with torch.no_grad():
            query_emb = encoder(query_tensor).numpy()

        results = search.search(query_emb, top_k=top_k)

        st.subheader(f"Query: {query_shape}")
        st.plotly_chart(plot_point_cloud(query_pts, f"Query: {query_shape}", color="red"), use_container_width=True)

        st.subheader(f"Top-{top_k} Results")
        result_cols = st.columns(min(top_k, 5))
        for i, r in enumerate(results[:5]):
            cls = r.metadata.get("class", "?") if r.metadata else "?"
            with result_cols[i]:
                st.metric(f"#{i+1}: {cls}", f"{r.score:.4f}")
                st.plotly_chart(
                    plot_point_cloud(all_pts[r.index], cls, size=1),
                    use_container_width=True,
                )

        # Class distribution in results
        result_classes = [r.metadata.get("class", "?") for r in results]
        from collections import Counter
        counts = Counter(result_classes)
        st.subheader("Result Class Distribution")
        fig = go.Figure(data=[go.Pie(labels=list(counts.keys()), values=list(counts.values()))])
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)


elif page == "Anomaly Detection":
    st.title("Anomaly Detection on 3D Geometry")

    col1, col2 = st.columns(2)
    with col1:
        normal_shape = st.selectbox("Normal shape", list(SHAPES.keys()), index=0)
        n_normal = st.slider("Normal samples", 4, 20, 8)
    with col2:
        anomaly_type = st.selectbox("Anomaly type", ["Random noise", "Scaled distortion", "Partial deletion"])
        n_anomaly = st.slider("Anomaly samples", 2, 10, 4)

    num_pts = 512

    if st.button("Run Anomaly Detection", type="primary"):
        detector = load_anomaly_detector(num_pts, 32)

        # Normal samples
        normal = torch.stack([torch.from_numpy(SHAPES[normal_shape](num_pts)) for _ in range(n_normal)])

        # Anomalous samples
        if anomaly_type == "Random noise":
            anomalous = torch.randn(n_anomaly, num_pts, 3) * 2.0
        elif anomaly_type == "Scaled distortion":
            anomalous = torch.stack([
                torch.from_numpy(SHAPES[normal_shape](num_pts)) * torch.tensor([3.0, 0.1, 3.0])
                for _ in range(n_anomaly)
            ])
        else:  # Partial deletion
            anomalous = []
            for _ in range(n_anomaly):
                pts = SHAPES[normal_shape](num_pts)
                mask = pts[:, 0] > 0  # keep only half
                kept = pts[mask]
                if len(kept) < num_pts:
                    pad = kept[np.random.choice(len(kept), num_pts - len(kept))]
                    kept = np.concatenate([kept, pad], axis=0)
                anomalous.append(torch.from_numpy(kept[:num_pts]))
            anomalous = torch.stack(anomalous)

        detector.eval()
        with torch.no_grad():
            normal_scores = detector.anomaly_score(normal).numpy()
            anomaly_scores = detector.anomaly_score(anomalous).numpy()

        # Train briefly
        with st.spinner("Training autoencoder (5 epochs)..."):
            from geofusion.models.anomaly import chamfer_distance
            optimizer = torch.optim.Adam(detector.autoencoder.parameters(), lr=1e-3)
            detector.autoencoder.train()
            train_losses = []
            for epoch in range(5):
                recon, z = detector.autoencoder(normal)
                loss = chamfer_distance(recon, normal).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

        detector.eval()
        with torch.no_grad():
            normal_scores_after = detector.anomaly_score(normal).numpy()
            anomaly_scores_after = detector.anomaly_score(anomalous).numpy()

        # Display
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader(f"Normal: {normal_shape}")
            st.plotly_chart(plot_point_cloud(normal[0].numpy(), f"Normal {normal_shape}", color="green"), use_container_width=True)
        with col_b:
            st.subheader(f"Anomaly: {anomaly_type}")
            st.plotly_chart(plot_point_cloud(anomalous[0].numpy(), f"Anomalous", color="red"), use_container_width=True)

        st.subheader("Anomaly Scores (after training)")
        fig = go.Figure()
        fig.add_trace(go.Box(y=normal_scores_after, name="Normal", marker_color="green"))
        fig.add_trace(go.Box(y=anomaly_scores_after, name="Anomalous", marker_color="red"))
        fig.update_layout(yaxis_title="Anomaly Score", height=350, title="Score Distribution")
        st.plotly_chart(fig, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Normal (mean)", f"{normal_scores_after.mean():.4f}")
        c2.metric("Anomaly (mean)", f"{anomaly_scores_after.mean():.4f}")
        separation = anomaly_scores_after.mean() / max(normal_scores_after.mean(), 1e-8)
        c3.metric("Score Ratio", f"{separation:.1f}x")


elif page == "Shape Generation":
    st.title("Diffusion-Based Shape Generation")

    col1, col2 = st.columns(2)
    with col1:
        gen_pts = st.selectbox("Points to generate", [128, 256, 512], index=1)
        hidden = st.selectbox("Hidden dim", [32, 64, 128], index=1)
    with col2:
        n_steps = st.selectbox("Diffusion steps", [50, 100, 200], index=1)
        n_shapes = st.slider("Number of shapes", 1, 4, 2)

    if st.button("Generate Shapes", type="primary"):
        model = load_diffusion(gen_pts, hidden, n_steps)

        # Quick training step on spheres
        with st.spinner("Quick training (1 step on synthetic data)..."):
            model.train()
            train_data = torch.stack([torch.from_numpy(make_sphere(gen_pts)) for _ in range(4)])
            out = model(train_data)
            out["loss"].backward()
            st.info(f"Training loss: {out['loss'].item():.4f}")

        with st.spinner(f"Generating {n_shapes} shapes ({n_steps} diffusion steps)..."):
            model.eval()
            t0 = time.time()
            with torch.no_grad():
                generated = model.sample(batch_size=n_shapes, device="cpu")
            elapsed = time.time() - t0

        st.success(f"Generated {n_shapes} shapes in {elapsed:.1f}s")

        cols = st.columns(n_shapes)
        for i in range(n_shapes):
            pts = generated[i].numpy()
            with cols[i]:
                st.plotly_chart(
                    plot_point_cloud(pts, f"Generated #{i+1}", size=2),
                    use_container_width=True,
                )
                st.caption(f"Range: [{pts.min():.2f}, {pts.max():.2f}]")
