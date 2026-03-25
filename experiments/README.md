# Experiments

This directory contains lightweight, reproducible entry points for key experiments,
along with conventions for documenting results.

The public repository intentionally includes only minimal experiment artifacts.
Detailed experiment history, full benchmark logs, and collaboration-specific
analyses are not included by default.

---

## Reproducible Entry Points

The following configurations can be used to reproduce core workflows:

- `configs/pointnet2.yaml`  
  Train a PointNet++ model for point-cloud-based geometry encoding.

- `configs/gnn.yaml`  
  Train graph-based models (e.g., DGCNN or GNNs on B-Rep-derived structures).

- `configs/multimodal.yaml`  
  Train a multimodal alignment model for geometry and text representations.

- `scripts/local_demo.py`  
  Run a synthetic end-to-end pipeline (data → model → retrieval) without external datasets.

---

## Experiment Documentation Guidelines

For each experiment, record the following:

1. Configuration file used  
2. Dataset description (including split and random seed)  
3. Hardware environment (CPU/GPU, memory)  
4. Key metrics (e.g., accuracy, retrieval performance, latency)  
5. Observed limitations or failure modes  
6. Checkpoint or artifact location (if applicable)

---

## Notes on Scope

This repository focuses on clarity, reproducibility, and architectural transparency.

It is not intended to expose full internal experimentation history or large-scale
benchmark logs, but instead to provide representative, reproducible examples of
the workflows and methods used.