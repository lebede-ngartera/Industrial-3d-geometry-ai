# API Reference

## geofusion.data

### datasets

**ModelNet40Dataset(root, split, num_points, use_normals, transform)**
Loads ModelNet40 point clouds. Returns (points, label) tuples where points is (num_points, 6) with XYZ and normals.

**ShapeNetDataset(root, categories, num_points, transform)**
Loads ShapeNet part segmentation data. Returns (points, seg_labels) tuples.

**PointCloudDataset(data_dir, file_ext, num_points, transform)**
Generic loader for .npy, .npz, .txt, .ply files. Returns (points, label) tuples.

### transforms

**FarthestPointSample(num_points)** Subsample point cloud to fixed size using farthest point sampling.

**NormalizePointCloud()** Center to origin and scale to unit sphere.

**RandomRotate(axis, max_angle)** Random rotation around specified axis.

**RandomJitter(sigma, clip)** Additive Gaussian noise with clipping.

**RandomScale(lo, hi)** Uniform random scaling.

**RandomFlip(axis, prob)** Reflection along specified axis.

**Compose(transforms)** Sequential application of transforms.

**ToTensor()** NumPy array to PyTorch tensor.

### text_metadata

**TextMetadataGenerator(seed)**
Generates synthetic engineering text descriptions for shape categories.

generate(category, num_points) Returns a text description string.

batch_generate(categories) Returns a list of description strings.

## geofusion.models

### pointnet2

**PointNet2Encoder(input_channels, embed_dim)**
Hierarchical point set encoder with three Set Abstraction layers. Returns (batch, embed_dim) embeddings.

**PointNet2Classifier(input_channels, num_classes, embed_dim)**
Wraps PointNet2Encoder with a classification head. Returns (logits, embeddings) tuple.

### gnn_encoder

**DGCNNEncoder(input_channels, embed_dim, k, num_layers)**
Dynamic Graph CNN encoder. Returns (batch, embed_dim) embeddings.

**GNNEncoder(input_channels, embed_dim, conv_type, num_layers)**
PyTorch Geometric GNN encoder. Supports EdgeConv, GATConv, SAGEConv.

### text_encoder

**TextEncoder(backbone, embed_dim, freeze_backbone)**
Transformer-based text encoder using HuggingFace sentence-transformers.

encode(texts) Returns (batch, embed_dim) normalized embeddings.

**SimpleTextEncoder(vocab_size, embed_dim, max_length)**
Lightweight text encoder using learned embeddings and 1D convolutions. No external dependencies.

### multimodal

**GeoFusionModel(geometry_encoder, text_encoder, embed_dim, num_classes)**
Main model integrating all encoders with contrastive alignment.

encode_geometry(points) Returns geometry embeddings.
encode_text(texts) Returns text embeddings.
forward(points, texts, metadata) Returns dict with embeddings, logits, and loss.

**MultimodalAligner(embed_dim, temperature)**
CLIP-style contrastive alignment with learnable temperature.

**MetadataEncoder(num_continuous, categorical_dims, embed_dim)**
Encodes mixed continuous and categorical manufacturing metadata.

### anomaly

**PointCloudAutoencoder(num_points, latent_dim)**
Encoder-decoder for point cloud reconstruction. Train on normal data.

**GeometryAnomalyDetector(autoencoder, embed_dim)**
Anomaly scoring via reconstruction error, embedding density, or ensemble.

fit_threshold(dataloader, percentile) Calibrate detection threshold.
detect(points) Returns (is_anomaly, scores).

### diffusion

**ShapeDiffusionModel(num_points, embed_dim, num_timesteps)**
DDPM-style point cloud generator.

sample(batch_size, condition) Generate new point clouds by iterative denoising.

## geofusion.retrieval

### search

**FAISSIndex(dim, index_type, metric)**
FAISS index wrapper. Supports Flat, IVFFlat, IVFPQ indices with cosine or L2 metric.

**SimilaritySearch(dim, index_type, metric)**
High-level search interface with metadata management.

build_index(embeddings, metadata, labels) Initialize index.
search(query, top_k) Returns list of SearchResult.

### embeddings

**EmbeddingStore()**
Manages computed embeddings with metadata. Supports incremental addition and persistence.

build_from_model(model, dataloader, modality) Compute and store embeddings.
save(path) / load(path) Serialize to disk.

### cross_modal

**CrossModalRetriever(model, embed_dim)**
Bidirectional cross-modal retrieval.

text_to_shape(query_text, top_k) Find shapes matching text.
shape_to_shape(query_points, top_k) Find similar shapes.
shape_to_text(query_points, top_k) Find text matching shape.

## geofusion.training

### trainer

**Trainer(model, config)**
Config-driven training orchestrator with AdamW, cosine annealing, warmup, early stopping, checkpointing, and optional W&B integration.

train() Full training loop.

### losses

**NTXentLoss(temperature)** Symmetric contrastive loss for multimodal alignment.

**TripletLoss(margin)** Margin-based metric learning loss.

**ClassificationLoss(num_classes, label_smoothing)** Cross-entropy with label smoothing.

**MultiTaskLoss(task_weights)** Weighted combination of multiple losses.

### metrics

**compute_accuracy(logits, targets, topk)** Top-k classification accuracy.

**compute_retrieval_metrics(queries, index, labels, k_values)** Recall@K, Precision@K, mAP.

**compute_cross_modal_metrics(geo_embeds, text_embeds, labels, k_values)** Bidirectional retrieval metrics.

## geofusion.workflows

### part_similarity

**PartSimilarityWorkflow(model, search_index)**

find_similar(query_points, top_k) Returns SimilarityReport with ranked candidates.
find_near_duplicates(embeddings, threshold) Returns duplicate pairs.
cluster_parts(embeddings, n_clusters) Returns KMeans cluster labels.

### anomaly_detection

**AnomalyDetectionWorkflow(detector)**

calibrate(normal_dataloader) Set warning and critical thresholds.
analyze(points, part_id) Returns AnomalyReport with risk level.
batch_analyze(dataloader) Batch analysis.

### property_prediction

**PropertyPredictor(encoder, properties)**

predict(points) Returns PropertyPrediction with values, uncertainties, and confidence.

### text_search

**TextToGeometrySearch(model, search_index)**

search(query_text, top_k, min_score) Returns TextSearchResult with ranked shapes.
