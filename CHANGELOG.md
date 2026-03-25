# Changelog

All notable changes to GeoFusion AI are documented in this file.

## [0.1.0] - 2024-01-15

### Added
- PointNet++ encoder with three-level Set Abstraction hierarchy
- DGCNN encoder with dynamic k-NN graph construction
- GNN encoder supporting EdgeConv, GATConv, and SAGEConv via PyTorch Geometric
- Transformer-based text encoder with HuggingFace sentence-transformers backend
- Lightweight SimpleTextEncoder fallback (no external NLP dependencies)
- GeoFusionModel integrating geometry, text, and metadata encoders
- CLIP-style multimodal contrastive aligner with NT-Xent loss and learnable temperature
- Metadata encoder for mixed continuous and categorical manufacturing attributes
- Point cloud autoencoder for reconstruction-based anomaly detection
- Geometry anomaly detector with reconstruction, density, and ensemble scoring
- DDPM-style shape diffusion model for point cloud generation
- FAISS-based similarity search with Flat, IVFFlat, and IVFPQ index types
- Embedding store with incremental addition and disk persistence
- Cross-modal retriever for bidirectional shape-text search
- Part similarity workflow with near-duplicate detection and KMeans clustering
- Anomaly detection workflow with calibrated warning and critical thresholds
- Property prediction workflow with aleatoric uncertainty quantification
- Text-to-geometry search workflow with natural language queries
- ModelNet40 and ShapeNet dataset loaders
- Generic PointCloudDataset supporting .npy, .npz, .txt, .ply formats
- Composable augmentation pipeline (FPS, normalize, rotate, jitter, scale, flip)
- Synthetic text metadata generator with category-specific engineering templates
- Config-driven Trainer with AdamW, cosine annealing, warmup, early stopping
- NT-Xent, triplet, classification, and multi-task loss functions
- Classification accuracy, retrieval recall/precision/mAP, cross-modal metrics
- YAML configuration files for PointNet++, DGCNN, GNN, and multimodal training
- Streamlit interactive dashboard with 7 pages
- Four Jupyter notebooks covering data exploration through industrial workflows
- Training, evaluation, index building, and demo scripts
- Self-contained local demo using synthetic data (no downloads required)
- Docker deployment configuration
- 37 unit tests covering data, models, and retrieval modules
- CI pipeline with GitHub Actions (lint, format, test, Docker build)
