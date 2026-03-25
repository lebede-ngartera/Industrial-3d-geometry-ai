"""Build FAISS retrieval index from trained model."""

import argparse
import logging

import torch
import yaml
from torch.utils.data import DataLoader

from geofusion.data.datasets import ModelNet40Dataset
from geofusion.data.transforms import Compose, FarthestPointSample, NormalizePointCloud
from geofusion.models.pointnet2 import PointNet2Classifier
from geofusion.retrieval.embeddings import EmbeddingStore
from geofusion.retrieval.search import SimilaritySearch

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Build retrieval index")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", default="outputs/retrieval_index")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = args.device or config.get("project", {}).get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # Load model
    embed_dim = config.get("geometry_encoder", {}).get("embed_dim", 256)
    model = PointNet2Classifier(num_classes=40, embed_dim=embed_dim)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Load dataset
    num_points = config.get("data", {}).get("num_points", 2048)
    transform = Compose([FarthestPointSample(num_points), NormalizePointCloud()])

    dataset = ModelNet40Dataset(
        data_root=config.get("data", {}).get("data_root", "data/raw/modelnet40_normal_resampled"),
        split="test",
        num_points=num_points,
        transform=transform,
    )

    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

    # Build embedding store
    store = EmbeddingStore()
    store.build_from_model(model, dataloader, device=device, modality="geometry")
    store.save(args.output)

    # Build FAISS search index
    search = SimilaritySearch(dim=embed_dim, metric="cosine")
    search.build_index(store.embeddings, store.metadata, store.labels)
    search.save(args.output)

    logger.info(f"Index built with {len(store)} entries at {args.output}")


if __name__ == "__main__":
    main()
