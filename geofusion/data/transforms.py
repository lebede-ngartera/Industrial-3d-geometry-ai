"""Point cloud transforms for data augmentation and preprocessing."""

from __future__ import annotations

import numpy as np
import torch


class Compose:
    """Compose multiple transforms sequentially."""

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, points: np.ndarray) -> np.ndarray:
        for t in self.transforms:
            points = t(points)
        return points


class NormalizePointCloud:
    """Center and scale point cloud to unit sphere."""

    def __call__(self, points: np.ndarray) -> np.ndarray:
        centroid = points[:, :3].mean(axis=0)
        points[:, :3] -= centroid
        max_dist = np.max(np.linalg.norm(points[:, :3], axis=1))
        if max_dist > 0:
            points[:, :3] /= max_dist
        return points


class FarthestPointSample:
    """Farthest point sampling to downsample point clouds."""

    def __init__(self, num_points: int):
        self.num_points = num_points

    def __call__(self, points: np.ndarray) -> np.ndarray:
        n = points.shape[0]
        if n <= self.num_points:
            # Pad by repeating points
            indices = np.arange(n)
            pad = np.random.choice(n, self.num_points - n, replace=True)
            indices = np.concatenate([indices, pad])
            return points[indices]

        centroids = np.zeros(self.num_points, dtype=np.int64)
        distances = np.full(n, np.inf)
        farthest = np.random.randint(0, n)

        for i in range(self.num_points):
            centroids[i] = farthest
            centroid_point = points[farthest, :3]
            dist = np.sum((points[:, :3] - centroid_point) ** 2, axis=1)
            distances = np.minimum(distances, dist)
            farthest = np.argmax(distances)

        return points[centroids]


class RandomRotate:
    """Random rotation around specified axis."""

    def __init__(self, axis: str = "y", max_angle: float = np.pi):
        self.axis = axis
        self.max_angle = max_angle

    def __call__(self, points: np.ndarray) -> np.ndarray:
        angle = np.random.uniform(-self.max_angle, self.max_angle)
        cos_a, sin_a = np.cos(angle), np.sin(angle)

        if self.axis == "x":
            rot = np.array([[1, 0, 0], [0, cos_a, -sin_a], [0, sin_a, cos_a]])
        elif self.axis == "y":
            rot = np.array([[cos_a, 0, sin_a], [0, 1, 0], [-sin_a, 0, cos_a]])
        else:  # z
            rot = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])

        points = points.copy()
        points[:, :3] = points[:, :3] @ rot.T
        if points.shape[1] > 3:
            # Rotate normals too
            points[:, 3:6] = points[:, 3:6] @ rot.T
        return points


class RandomJitter:
    """Add random Gaussian noise to point positions."""

    def __init__(self, sigma: float = 0.01, clip: float = 0.05):
        self.sigma = sigma
        self.clip = clip

    def __call__(self, points: np.ndarray) -> np.ndarray:
        points = points.copy()
        noise = np.clip(
            np.random.randn(points.shape[0], 3) * self.sigma,
            -self.clip,
            self.clip,
        )
        points[:, :3] += noise
        return points


class RandomScale:
    """Random uniform scaling of the point cloud."""

    def __init__(self, lo: float = 0.8, hi: float = 1.25):
        self.lo = lo
        self.hi = hi

    def __call__(self, points: np.ndarray) -> np.ndarray:
        scale = np.random.uniform(self.lo, self.hi)
        points = points.copy()
        points[:, :3] *= scale
        return points


class RandomFlip:
    """Random reflection along an axis."""

    def __init__(self, axis: int = 0, prob: float = 0.5):
        self.axis = axis
        self.prob = prob

    def __call__(self, points: np.ndarray) -> np.ndarray:
        if np.random.random() < self.prob:
            points = points.copy()
            points[:, self.axis] = -points[:, self.axis]
            if points.shape[1] > 3:
                points[:, 3 + self.axis] = -points[:, 3 + self.axis]
        return points


class ToTensor:
    """Convert numpy point cloud to PyTorch tensor."""

    def __call__(self, points: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(points).float()
