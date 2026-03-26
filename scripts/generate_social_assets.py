from pathlib import Path

from PIL import Image, ImageOps

TARGET_SIZE = (1200, 627)
ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = ROOT / "docs" / "figures"
SOCIAL_DIR = FIGURES_DIR / "social"

ASSET_SPECS = {
    "demo_interface_snapshot.png": {
        "output": "geofusion-overview-linkedin.png",
        "centering": (0.5, 0.16),
    },
    "classification_training_snapshot.png": {
        "output": "geofusion-classification-linkedin.png",
        "centering": (0.5, 0.3),
    },
    "retrieval_query_vs_results.png": {
        "output": "geofusion-retrieval-linkedin.png",
        "centering": (0.5, 0.28),
    },
    "anomaly_normal_vs_anomalous.png": {
        "output": "geofusion-anomaly-linkedin.png",
        "centering": (0.5, 0.26),
    },
}


def render_social_crop(
    source_path: Path, output_path: Path, centering: tuple[float, float]
) -> None:
    with Image.open(source_path) as image:
        fitted = ImageOps.fit(
            image.convert("RGB"),
            TARGET_SIZE,
            method=Image.Resampling.LANCZOS,
            centering=centering,
        )
        fitted.save(output_path, format="PNG", optimize=True)


def main() -> None:
    SOCIAL_DIR.mkdir(parents=True, exist_ok=True)

    for source_name, spec in ASSET_SPECS.items():
        source_path = FIGURES_DIR / source_name
        output_path = SOCIAL_DIR / spec["output"]
        if not source_path.exists():
            raise FileNotFoundError(f"Missing source figure: {source_path}")
        render_social_crop(source_path, output_path, spec["centering"])
        print(f"wrote {output_path.relative_to(ROOT)}")

    preview_source = SOCIAL_DIR / ASSET_SPECS["demo_interface_snapshot.png"]["output"]
    preview_target = ROOT / "social-preview.png"
    with Image.open(preview_source) as image:
        image.save(preview_target, format="PNG", optimize=True)
    print(f"wrote {preview_target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
