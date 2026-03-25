"""Tests for data pipeline — datasets, transforms, text metadata."""

import numpy as np
import pytest
import torch

from geofusion.data.transforms import (
    Compose,
    FarthestPointSample,
    NormalizePointCloud,
    RandomFlip,
    RandomJitter,
    RandomRotate,
    RandomScale,
    ToTensor,
)
from geofusion.data.text_metadata import TextMetadataGenerator


# ---------------------------------------------------------------------------
# Transform tests
# ---------------------------------------------------------------------------


class TestNormalizePointCloud:
    def test_center_and_scale(self):
        points = np.random.randn(100, 3).astype(np.float32) + 10.0
        normed = NormalizePointCloud()(points)
        assert np.abs(normed.mean(axis=0)).max() < 1e-5, "Not centred"
        assert np.max(np.linalg.norm(normed, axis=1)) <= 1.0 + 1e-5

    def test_preserves_shape(self):
        points = np.random.randn(256, 3).astype(np.float32)
        assert NormalizePointCloud()(points).shape == (256, 3)


class TestFarthestPointSample:
    def test_correct_count(self):
        points = np.random.randn(2048, 3).astype(np.float32)
        sampled = FarthestPointSample(512)(points)
        assert sampled.shape == (512, 3)

    def test_upsample_repeats(self):
        points = np.random.randn(10, 3).astype(np.float32)
        sampled = FarthestPointSample(20)(points)
        assert sampled.shape == (20, 3)


class TestRandomRotate:
    def test_output_shape(self):
        points = np.random.randn(100, 3).astype(np.float32)
        rotated = RandomRotate(axis="y")(points)
        assert rotated.shape == points.shape

    def test_preserves_norms(self):
        points = np.random.randn(50, 3).astype(np.float32)
        norms_before = np.linalg.norm(points, axis=1)
        norms_after = np.linalg.norm(RandomRotate()(points), axis=1)
        np.testing.assert_allclose(norms_before, norms_after, atol=1e-5)


class TestRandomJitter:
    def test_bounded(self):
        points = np.zeros((100, 3), dtype=np.float32)
        jittered = RandomJitter(sigma=0.01, clip=0.05)(points)
        assert np.abs(jittered).max() <= 0.05 + 1e-6


class TestRandomScale:
    def test_range(self):
        points = np.ones((50, 3), dtype=np.float32)
        for _ in range(20):
            scaled = RandomScale(lo=0.8, hi=1.2)(points)
            ratio = scaled[0, 0]
            assert 0.8 - 1e-5 <= ratio <= 1.2 + 1e-5


class TestRandomFlip:
    def test_axis(self):
        points = np.array([[1, 2, 3]], dtype=np.float32)
        np.random.seed(0)
        flipped = RandomFlip(axis=0, prob=1.0)(points.copy())
        assert flipped[0, 0] == -1.0


class TestCompose:
    def test_pipeline(self):
        pipeline = Compose([NormalizePointCloud(), FarthestPointSample(64)])
        points = np.random.randn(256, 3).astype(np.float32)
        out = pipeline(points)
        assert out.shape == (64, 3)


class TestToTensor:
    def test_numpy_to_tensor(self):
        arr = np.random.randn(10, 3).astype(np.float32)
        t = ToTensor()(arr)
        assert isinstance(t, torch.Tensor)
        assert t.shape == (10, 3)


# ---------------------------------------------------------------------------
# TextMetadataGenerator tests
# ---------------------------------------------------------------------------


class TestTextMetadataGenerator:
    def setup_method(self):
        self.gen = TextMetadataGenerator(seed=42)

    def test_generates_string(self):
        text = self.gen.generate("airplane")
        assert isinstance(text, str)
        assert len(text) > 10

    def test_deterministic_with_seed(self):
        g1 = TextMetadataGenerator(seed=7)
        g2 = TextMetadataGenerator(seed=7)
        assert g1.generate("chair") == g2.generate("chair")

    def test_batch_generate(self):
        categories = ["chair", "table", "car"]
        texts = [self.gen.generate(cat) for cat in categories]
        assert len(texts) == 3
        assert all(isinstance(t, str) for t in texts)
