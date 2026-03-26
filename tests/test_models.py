"""Tests for model architectures — forward passes, output shapes."""

import torch

from geofusion.models.anomaly import GeometryAnomalyDetector, PointCloudAutoencoder
from geofusion.models.diffusion import ShapeDiffusionModel
from geofusion.models.gnn_encoder import DGCNNEncoder
from geofusion.models.multimodal import GeoFusionModel, MultimodalAligner
from geofusion.models.pointnet2 import PointNet2Classifier, PointNet2Encoder
from geofusion.models.text_encoder import SimpleTextEncoder

BATCH = 4
NUM_POINTS = 256
EMBED_DIM = 64


# ---------------------------------------------------------------------------
# PointNet++ tests
# ---------------------------------------------------------------------------


class TestPointNet2:
    def test_encoder_output_shape(self):
        model = PointNet2Encoder(embed_dim=EMBED_DIM, use_normals=False)
        x = torch.randn(BATCH, NUM_POINTS, 3)
        emb = model(x)
        assert emb.shape == (BATCH, EMBED_DIM)

    def test_encoder_with_normals(self):
        model = PointNet2Encoder(embed_dim=EMBED_DIM, use_normals=True)
        x = torch.randn(BATCH, NUM_POINTS, 6)
        emb = model(x)
        assert emb.shape == (BATCH, EMBED_DIM)

    def test_classifier(self):
        model = PointNet2Classifier(num_classes=10, embed_dim=EMBED_DIM, use_normals=False)
        x = torch.randn(BATCH, NUM_POINTS, 3)
        logits, emb = model(x)
        assert logits.shape == (BATCH, 10)
        assert emb.shape == (BATCH, EMBED_DIM)

    def test_backward(self):
        model = PointNet2Classifier(num_classes=5, embed_dim=EMBED_DIM, use_normals=False)
        x = torch.randn(BATCH, NUM_POINTS, 3)
        logits, _ = model(x)
        loss = logits.sum()
        loss.backward()
        for p in model.parameters():
            if p.requires_grad:
                assert p.grad is not None


# ---------------------------------------------------------------------------
# DGCNN tests
# ---------------------------------------------------------------------------


class TestDGCNN:
    def test_output_shape(self):
        model = DGCNNEncoder(in_channels=3, embed_dim=EMBED_DIM, k=10)
        x = torch.randn(BATCH, NUM_POINTS, 3)
        emb = model(x)
        assert emb.shape == (BATCH, EMBED_DIM)

    def test_different_k(self):
        model = DGCNNEncoder(in_channels=3, embed_dim=EMBED_DIM, k=5)
        x = torch.randn(BATCH, 64, 3)
        emb = model(x)
        assert emb.shape == (BATCH, EMBED_DIM)


# ---------------------------------------------------------------------------
# Text encoder tests
# ---------------------------------------------------------------------------


class TestSimpleTextEncoder:
    def test_output_shape(self):
        model = SimpleTextEncoder(vocab_size=1000, embed_dim=EMBED_DIM)
        tokens = torch.randint(0, 1000, (BATCH, 32))
        out = model(tokens)
        assert out.shape == (BATCH, EMBED_DIM)


# ---------------------------------------------------------------------------
# Multimodal alignment tests
# ---------------------------------------------------------------------------


class TestMultimodalAligner:
    def test_loss_not_nan(self):
        aligner = MultimodalAligner(
            geometry_dim=EMBED_DIM, text_dim=EMBED_DIM, shared_dim=EMBED_DIM
        )
        geo_emb = torch.randn(BATCH, EMBED_DIM)
        txt_emb = torch.randn(BATCH, EMBED_DIM)
        out = aligner(geo_emb, txt_emb)
        loss = out["loss"]
        assert not torch.isnan(loss)
        assert loss.item() > 0

    def test_temperature_learnable(self):
        aligner = MultimodalAligner(geometry_dim=EMBED_DIM, text_dim=EMBED_DIM, temperature=0.1)
        assert aligner.temperature.requires_grad


class TestGeoFusionModel:
    def test_forward_without_text(self):
        geo_enc = PointNet2Encoder(embed_dim=EMBED_DIM, use_normals=False)
        txt_enc = SimpleTextEncoder(vocab_size=500, embed_dim=EMBED_DIM)
        model = GeoFusionModel(
            geometry_encoder=geo_enc,
            text_encoder=txt_enc,
            embed_dim=EMBED_DIM,
            num_classes=10,
        )
        points = torch.randn(BATCH, NUM_POINTS, 3)
        out = model(points=points)
        assert "geometry_embedding" in out
        assert out["geometry_embedding"].shape == (BATCH, EMBED_DIM)


# ---------------------------------------------------------------------------
# Anomaly detection tests
# ---------------------------------------------------------------------------


class TestPointCloudAutoencoder:
    def test_reconstruction_shape(self):
        ae = PointCloudAutoencoder(num_points=NUM_POINTS, latent_dim=EMBED_DIM)
        x = torch.randn(BATCH, NUM_POINTS, 3)
        recon, emb = ae(x)
        assert recon.shape == (BATCH, NUM_POINTS, 3)
        assert emb.shape == (BATCH, EMBED_DIM)


class TestGeometryAnomalyDetector:
    def test_score(self):
        detector = GeometryAnomalyDetector(
            num_points=NUM_POINTS, latent_dim=EMBED_DIM, method="reconstruction"
        )
        detector.eval()

        # Score a batch
        test_batch = torch.randn(BATCH, NUM_POINTS, 3)
        with torch.no_grad():
            scores = detector.anomaly_score(test_batch)
        assert scores.shape == (BATCH,)
        assert not torch.isnan(scores).any()


# ---------------------------------------------------------------------------
# Diffusion tests
# ---------------------------------------------------------------------------


class TestShapeDiffusion:
    def test_training_loss(self):
        model = ShapeDiffusionModel(
            num_points=NUM_POINTS,
            hidden_dim=EMBED_DIM,
            num_timesteps=10,
            condition_dim=None,
        )
        x = torch.randn(BATCH, NUM_POINTS, 3)
        out = model(x)
        loss = out["loss"]
        assert not torch.isnan(loss)
        assert loss.item() > 0

    def test_sample_shape(self):
        model = ShapeDiffusionModel(
            num_points=32,
            hidden_dim=EMBED_DIM,
            num_timesteps=5,
            condition_dim=None,
        )
        samples = model.sample(batch_size=2)
        assert samples.shape == (2, 32, 3)
