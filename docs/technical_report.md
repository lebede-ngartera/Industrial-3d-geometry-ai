# GeoFusion AI Technical Report

[Open the collaboration landing page](/e:/GIThub_Prohect/docs/index.html)

## 1. Introduction

GeoFusion AI is a multimodal deep learning platform for industrial 3D CAD geometry understanding. It combines geometric encoders (PointNet++, DGCNN), transformer-based text encoders, and a CLIP-style contrastive alignment layer to build a shared embedding space across 3D shapes, natural language descriptions, and manufacturing metadata. The system supports five downstream tasks: part similarity retrieval, text-to-geometry search, manufacturing anomaly detection, property prediction with uncertainty quantification, and diffusion-based shape generation.

## 2. System Architecture

The architecture follows a multi-encoder fusion pattern. Each input modality is processed by a dedicated encoder:

Geometry Encoder (PointNet++ or DGCNN) transforms raw point clouds (N x 3 or N x 6 with normals) into 256-dimensional embeddings through hierarchical set abstraction or dynamic graph convolutions.

Text Encoder (Transformer-based) maps engineering text descriptions to the same 256-dimensional space using a pretrained sentence-transformer backbone with a learned projection head.

Metadata Encoder processes continuous and categorical manufacturing attributes (material, tolerance, process parameters) through batch normalization, embedding layers, and fully connected projections.

A Multimodal Contrastive Aligner computes a temperature-scaled similarity matrix between geometry and text embeddings and optimizes a symmetric NT-Xent loss, learning to align semantically related geometry-text pairs in the shared space.

## 3. Model Details

### 3.1 PointNet++ Encoder

Three Set Abstraction layers with progressively coarser sampling (512, 128, global). Each layer applies farthest point sampling, ball query neighborhood grouping, and a shared MLP with batch normalization. The global feature (1024 dimensions) is projected to 256 dimensions through a two-layer head with dropout.

### 3.2 DGCNN Encoder

Four EdgeConv blocks construct dynamic k-nearest-neighbor graphs (k=20) in feature space at each layer. Edge features h(x_i, x_j - x_i) are processed through shared MLPs. All layer outputs are concatenated, passed through a 1D convolution bottleneck, and pooled (global max + mean) to produce the final embedding.

### 3.3 Diffusion Model

A DDPM-style generative model with 1000 timesteps and a linear noise schedule (beta from 1e-4 to 0.02). The denoiser network uses sinusoidal position embeddings for timestep conditioning, optional geometry/text conditioning, and residual blocks with group normalization. Generation proceeds by iterative denoising from Gaussian noise.

### 3.4 Anomaly Detector

An ensemble of reconstruction-based scoring (point cloud autoencoder with Chamfer distance) and density-based scoring (embedding distance to nearest normal cluster). Thresholds are calibrated on normal data at configurable percentiles (90th for warning, 99th for critical).

## 4. Data Pipeline

The system supports ModelNet40 (40-class classification, 12K models), ShapeNet (part segmentation, 16 categories), and custom point cloud data in .npy, .npz, .txt, and .ply formats.

The augmentation pipeline includes farthest point sampling, unit sphere normalization, random rotation, Gaussian jitter, random scaling, and axis flipping. Transforms are composable and applied at training time.

A synthetic text metadata generator produces engineering-style descriptions using category-specific templates, enabling multimodal training without manual annotation.

## 5. Training Protocol

Training uses AdamW optimization with cosine annealing learning rate schedule and linear warmup. Gradient clipping (max norm 1.0) and early stopping (patience 20 epochs) are applied by default. All hyperparameters are managed through YAML configuration files.

Loss functions include NT-Xent for contrastive alignment, triplet loss for metric learning, cross-entropy with label smoothing for classification, and a weighted multi-task loss for joint training.

## 6. Retrieval System

FAISS-based vector search supports three index types: Flat (exact), IVFFlat (approximate, 100 Voronoi cells), and IVFPQ (product-quantized). Cosine similarity is computed via L2 distance on normalized vectors. The system supports bidirectional cross-modal retrieval (shape-to-text, text-to-shape, shape-to-shape, text-to-text).

## 7. Evaluation Metrics

Classification: Top-1 and Top-5 accuracy.
Retrieval: Recall@K, Precision@K, Mean Average Precision.
Cross-Modal: Bidirectional retrieval metrics.
Anomaly Detection: Threshold-calibrated detection at configurable percentiles.
Property Prediction: Mean absolute error with aleatoric uncertainty.

## 8. Industrial Workflows

Part Similarity: Encode query shape, retrieve top-k results from FAISS index, with optional near-duplicate detection and KMeans clustering.

Anomaly Detection: Calibrate on normal production data, score new parts with ensemble method, classify into normal/warning/critical risk levels.

Property Prediction: Predict mass, volume, surface area, max stress, and manufacturability score from geometry embeddings, with per-property uncertainty estimates.

Text Search: Accept natural language queries ("lightweight bracket with curved support arm"), retrieve 3D shapes by projected text-geometry embedding similarity.

## 9. Reproducibility

All experiments are reproducible through: fixed random seeds (default 42), YAML configuration files, deterministic data loading, checkpoint serialization (model + optimizer state), and optional W&B integration for experiment tracking.

## 10. References

Qi, C. R., Yi, L., Su, H., and Guibas, L. J. PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space. NeurIPS, 2017.

Wang, Y., Sun, Y., Liu, Z., Sarma, S. E., Bronstein, M. M., and Solomon, J. M. Dynamic Graph CNN for Learning on Point Clouds. ACM TOG (SIGGRAPH), 2019.

Radford, A., Kim, J. W., Hallacy, C., et al. Learning Transferable Visual Models From Natural Language Supervision. ICML, 2021.

Ho, J., Jain, A., and Abbeel, P. Denoising Diffusion Probabilistic Models. NeurIPS, 2020.
