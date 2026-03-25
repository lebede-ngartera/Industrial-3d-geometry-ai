"""Dataset download utilities for ModelNet40, ShapeNet, and ABC datasets."""

import hashlib
import logging
import os
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

logger = logging.getLogger(__name__)

MODELNET40_URL = (
    "https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip"
)
SHAPENET_URL = "https://shapenet.cs.stanford.edu/ericyi/shapenetcore_partanno_segmentation_benchmark_v0.zip"


def _download_file(url: str, dest: Path, expected_hash: str | None = None) -> Path:
    """Download a file with optional hash verification."""
    if dest.exists():
        if expected_hash:
            sha256 = hashlib.sha256(dest.read_bytes()).hexdigest()
            if sha256 == expected_hash:
                logger.info(f"File already exists and hash matches: {dest}")
                return dest
            logger.warning(f"Hash mismatch for {dest}, re-downloading...")
        else:
            logger.info(f"File already exists: {dest}")
            return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading {url} -> {dest}")

    def _progress(count, block_size, total_size):
        pct = count * block_size * 100 / total_size if total_size > 0 else 0
        print(f"\rDownloading: {pct:.1f}%", end="", flush=True)

    urlretrieve(url, str(dest), reporthook=_progress)
    print()  # newline after progress
    return dest


def _extract_zip(zip_path: Path, dest_dir: Path) -> None:
    """Safely extract a zip file."""
    logger.info(f"Extracting {zip_path} -> {dest_dir}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            member_path = Path(dest_dir / member).resolve()
            if not str(member_path).startswith(str(dest_dir.resolve())):
                raise ValueError(f"Zip path traversal detected: {member}")
        zf.extractall(dest_dir)


def download_modelnet40(data_root: str = "data/raw") -> Path:
    """Download and extract ModelNet40 dataset.

    Args:
        data_root: Root directory for data storage.

    Returns:
        Path to extracted dataset directory.
    """
    root = Path(data_root)
    zip_path = root / "modelnet40_normal_resampled.zip"
    extract_dir = root / "modelnet40_normal_resampled"

    if extract_dir.exists() and any(extract_dir.iterdir()):
        logger.info(f"ModelNet40 already extracted at {extract_dir}")
        return extract_dir

    _download_file(MODELNET40_URL, zip_path)
    _extract_zip(zip_path, root)
    return extract_dir


def download_shapenet(data_root: str = "data/raw") -> Path:
    """Download and extract ShapeNet part segmentation dataset.

    Args:
        data_root: Root directory for data storage.

    Returns:
        Path to extracted dataset directory.
    """
    root = Path(data_root)
    zip_path = root / "shapenetcore_partanno.zip"
    extract_dir = root / "shapenetcore_partanno_segmentation_benchmark_v0"

    if extract_dir.exists() and any(extract_dir.iterdir()):
        logger.info(f"ShapeNet already extracted at {extract_dir}")
        return extract_dir

    _download_file(SHAPENET_URL, zip_path)
    _extract_zip(zip_path, root)
    return extract_dir


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Download 3D datasets")
    parser.add_argument(
        "--dataset",
        choices=["modelnet40", "shapenet", "all"],
        default="modelnet40",
    )
    parser.add_argument("--data-root", default="data/raw")
    args = parser.parse_args()

    if args.dataset in ("modelnet40", "all"):
        download_modelnet40(args.data_root)
    if args.dataset in ("shapenet", "all"):
        download_shapenet(args.data_root)
