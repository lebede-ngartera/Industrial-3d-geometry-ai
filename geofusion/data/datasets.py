"""Dataset classes for 3D shape datasets."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from geofusion.data.transforms import (
    Compose,
    FarthestPointSample,
    NormalizePointCloud,
)

logger = logging.getLogger(__name__)

# ModelNet40 class names
MODELNET40_CLASSES = [
    "airplane", "bathtub", "bed", "bench", "bookshelf", "bottle", "bowl", "car",
    "chair", "cone", "cup", "curtain", "desk", "door", "dresser", "flower_pot",
    "glass_box", "guitar", "keyboard", "lamp", "laptop", "mantel", "monitor",
    "night_stand", "person", "piano", "plant", "radio", "range_hood", "sink",
    "sofa", "stairs", "stool", "table", "tent", "toilet", "tv_stand", "vase",
    "wardrobe", "xbox",
]


class ModelNet40Dataset(Dataset):
    """ModelNet40 point cloud dataset.

    Loads point clouds from the ModelNet40 normal-resampled dataset.
    Each sample has XYZ coordinates + surface normals (6 channels).
    """

    def __init__(
        self,
        data_root: str = "data/raw/modelnet40_normal_resampled",
        split: str = "train",
        num_points: int = 2048,
        use_normals: bool = True,
        transform: Any = None,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.num_points = num_points
        self.use_normals = use_normals
        self.transform = transform

        self.classes = sorted(MODELNET40_CLASSES)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.samples: list[tuple[Path, int]] = []
        self._load_file_list()

        if self.transform is None:
            self.transform = Compose([
                FarthestPointSample(num_points),
                NormalizePointCloud(),
            ])

    def _load_file_list(self) -> None:
        """Scan data directory for point cloud files."""
        for class_name in self.classes:
            class_dir = self.data_root / class_name / self.split
            if not class_dir.exists():
                # Fallback: try without split subdirectory
                class_dir = self.data_root / class_name
                if not class_dir.exists():
                    continue

            label = self.class_to_idx[class_name]
            for f in sorted(class_dir.glob("*.txt")):
                self.samples.append((f, label))

        logger.info(
            f"ModelNet40 [{self.split}]: {len(self.samples)} samples, "
            f"{len(self.classes)} classes"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        filepath, label = self.samples[idx]
        points = np.loadtxt(filepath, delimiter=",", dtype=np.float32)

        if not self.use_normals:
            points = points[:, :3]

        if self.transform:
            points = self.transform(points)

        return {
            "points": torch.from_numpy(points).float(),
            "label": torch.tensor(label, dtype=torch.long),
            "class_name": self.classes[label],
            "filename": filepath.stem,
        }


class ShapeNetDataset(Dataset):
    """ShapeNet part segmentation dataset.

    Loads ShapeNet models with part annotations and optional text metadata.
    """

    # ShapeNet category mapping (synset ID -> human-readable name)
    CATEGORY_MAP = {
        "02691156": "airplane",
        "02773838": "bag",
        "02954340": "cap",
        "02958343": "car",
        "03001627": "chair",
        "03261776": "earphone",
        "03467517": "guitar",
        "03624134": "knife",
        "03636649": "lamp",
        "03642806": "laptop",
        "03790512": "motorbike",
        "03797390": "mug",
        "03948459": "pistol",
        "04099429": "rocket",
        "04225987": "skateboard",
        "04379243": "table",
    }

    def __init__(
        self,
        data_root: str = "data/raw/shapenetcore_partanno_segmentation_benchmark_v0",
        split: str = "train",
        num_points: int = 2048,
        categories: list[str] | None = None,
        transform: Any = None,
        include_text: bool = False,
        text_generator: Any = None,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.num_points = num_points
        self.include_text = include_text
        self.text_generator = text_generator
        self.transform = transform

        # Build category filter
        if categories:
            self.categories = {
                k: v for k, v in self.CATEGORY_MAP.items() if v in categories
            }
        else:
            self.categories = dict(self.CATEGORY_MAP)

        self.cat_to_idx = {
            name: i for i, name in enumerate(sorted(self.categories.values()))
        }

        self.samples: list[dict] = []
        self._load_file_list()

        if self.transform is None:
            self.transform = Compose([
                FarthestPointSample(num_points),
                NormalizePointCloud(),
            ])

    def _load_file_list(self) -> None:
        """Load file list from split files."""
        split_file = self.data_root / "train_test_split" / f"shuffled_{self.split}_file_list.json"

        if split_file.exists():
            import json
            with open(split_file) as f:
                file_list = json.load(f)
            for entry in file_list:
                # entry format: "shape_data/synsetId/modelId"
                parts = entry.strip().split("/")
                if len(parts) >= 3:
                    synset_id = parts[1]
                    model_id = parts[2]
                    if synset_id in self.categories:
                        pts_file = self.data_root / synset_id / f"{model_id}.pts"
                        seg_file = self.data_root / synset_id / f"{model_id}.seg"
                        if pts_file.exists():
                            self.samples.append({
                                "pts_file": pts_file,
                                "seg_file": seg_file if seg_file.exists() else None,
                                "category": self.categories[synset_id],
                                "synset_id": synset_id,
                                "model_id": model_id,
                            })
        else:
            # Fallback: scan directories
            for synset_id, cat_name in self.categories.items():
                cat_dir = self.data_root / synset_id
                if not cat_dir.exists():
                    continue
                for pts_file in sorted(cat_dir.glob("*.pts")):
                    model_id = pts_file.stem
                    seg_file = cat_dir / f"{model_id}.seg"
                    self.samples.append({
                        "pts_file": pts_file,
                        "seg_file": seg_file if seg_file.exists() else None,
                        "category": cat_name,
                        "synset_id": synset_id,
                        "model_id": model_id,
                    })

        logger.info(
            f"ShapeNet [{self.split}]: {len(self.samples)} samples, "
            f"{len(self.categories)} categories"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        points = np.loadtxt(sample["pts_file"], dtype=np.float32)

        # Load part segmentation labels if available
        seg_labels = None
        if sample["seg_file"] is not None:
            seg_labels = np.loadtxt(sample["seg_file"], dtype=np.int64)

        if self.transform:
            if seg_labels is not None:
                # Keep track of which points were sampled
                n_orig = points.shape[0]
                combined = np.column_stack([points, seg_labels.reshape(-1, 1)])
                combined = self.transform(combined)
                points = combined[:, :-1]
                seg_labels = combined[:, -1].astype(np.int64)
            else:
                points = self.transform(points)

        result = {
            "points": torch.from_numpy(points).float(),
            "label": torch.tensor(
                self.cat_to_idx[sample["category"]], dtype=torch.long
            ),
            "category": sample["category"],
            "model_id": sample["model_id"],
        }

        if seg_labels is not None:
            result["seg_labels"] = torch.from_numpy(seg_labels).long()

        if self.include_text and self.text_generator:
            text = self.text_generator.generate(
                category=sample["category"],
                points=points,
                model_id=sample["model_id"],
            )
            result["text"] = text

        return result


class PointCloudDataset(Dataset):
    """Generic point cloud dataset from numpy files or directories.

    Useful for custom CAD-derived point clouds.
    """

    def __init__(
        self,
        file_paths: list[str],
        labels: list[int] | None = None,
        num_points: int = 2048,
        transform: Any = None,
    ):
        self.file_paths = [Path(p) for p in file_paths]
        self.labels = labels
        self.num_points = num_points
        self.transform = transform or Compose([
            FarthestPointSample(num_points),
            NormalizePointCloud(),
        ])

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        filepath = self.file_paths[idx]

        if filepath.suffix == ".npy":
            points = np.load(filepath).astype(np.float32)
        elif filepath.suffix == ".npz":
            data = np.load(filepath)
            points = data["points"].astype(np.float32)
        elif filepath.suffix == ".txt":
            points = np.loadtxt(filepath, delimiter=",", dtype=np.float32)
        elif filepath.suffix == ".ply":
            import trimesh
            mesh = trimesh.load(str(filepath))
            points = np.array(mesh.vertices, dtype=np.float32)
        else:
            raise ValueError(f"Unsupported format: {filepath.suffix}")

        if self.transform:
            points = self.transform(points)

        result = {
            "points": torch.from_numpy(points).float(),
            "filename": filepath.stem,
        }
        if self.labels is not None:
            result["label"] = torch.tensor(self.labels[idx], dtype=torch.long)

        return result
