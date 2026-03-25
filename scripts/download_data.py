"""Download 3D datasets for GeoFusion training."""

import argparse
import logging

from geofusion.data.download import download_modelnet40, download_shapenet


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Download 3D shape datasets")
    parser.add_argument(
        "--dataset",
        choices=["modelnet40", "shapenet", "all"],
        default="modelnet40",
        help="Dataset to download",
    )
    parser.add_argument("--data-root", default="data/raw", help="Data storage directory")
    args = parser.parse_args()

    if args.dataset in ("modelnet40", "all"):
        path = download_modelnet40(args.data_root)
        print(f"ModelNet40 ready at: {path}")

    if args.dataset in ("shapenet", "all"):
        path = download_shapenet(args.data_root)
        print(f"ShapeNet ready at: {path}")


if __name__ == "__main__":
    main()
