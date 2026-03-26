# Public Benchmark Manifest

This file defines the public-safe benchmark used for the repository's reproducible evidence.

## Source Of Truth

- Script: `scripts/public_eval.py`
- Metrics output: `results/public_metrics.json`
- Markdown summary: `results/public_metrics.md`

## Default Command

```bash
python scripts/public_eval.py
```

## Default Evaluation Regime

- Seed: `42`
- Geometry classes: `Sphere`, `Cube`, `Cylinder`, `Cone`, `Torus`
- Samples per class: `10`
- Generated split: `8 train / 2 test` per class
- Total public subset: `40 train / 10 test`
- Points per sample: `512`
- Classifier: `PointNet2Classifier(embed_dim=64, dropout=0.2)`
- Classifier training epochs: `3`
- Anomaly detector: reconstruction-based `GeometryAnomalyDetector`
- Anomaly training epochs: `4`
- Batch size: `8`

## Interpretation Rules

1. These metrics validate the public repository's executable surface on a reproducible synthetic subset.
2. They support claims about workflow coherence, reproducibility, and module integration.
3. They do not substitute for private industrial CAD benchmarks, deployment studies, or domain-specific retrieval tables.
4. UI screenshots in the README are execution evidence for the public demo and should not be treated as the repository's headline benchmark source.

## Why This Manifest Exists

The repository contains both scripted evaluation artifacts and presentation-oriented screenshots. This manifest keeps those two evidence types separate so the public narrative stays precise.