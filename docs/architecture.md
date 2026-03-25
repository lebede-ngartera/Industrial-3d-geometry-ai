# Architecture

![System overview](figures/system_overview.svg)

## System Goal

GeoFusion AI learns shared representations for 3D geometry, engineering text, and structured metadata so that downstream industrial workflows can use a common semantic space.

## Logical Pipeline

![Workflow map](figures/workflow_map.svg)

1. Data ingestion
   Public datasets or custom point cloud files are loaded through dataset adapters.

2. Geometric preprocessing
   Point clouds are sampled, normalized, and augmented.

3. Representation learning
   Geometry is encoded with PointNet++ or DGCNN.
   Text is encoded with a transformer based model.
   Metadata is encoded with embedding layers plus fully connected projections.

4. Shared latent alignment
   The multimodal aligner projects modalities into a common 256 dimensional space using NT Xent contrastive learning.

5. Workflow layer
   The shared embedding is consumed by retrieval, anomaly scoring, property prediction, and cross modal search.

6. Product layer
   Streamlit, notebooks, and scripts expose the system for exploration and validation.

## Module Interfaces

### geofusion.data
Responsible for dataset loading, transforms, and synthetic metadata generation.

### geofusion.models
Contains the geometry encoders, text encoders, multimodal aligner, anomaly model, and diffusion model.

### geofusion.retrieval
Provides FAISS indexing, embedding storage, and cross modal retrieval.

### geofusion.training
Provides losses, metrics, and config driven training orchestration.

### geofusion.workflows
Packages the models into industrial tasks such as part similarity, anomaly analysis, and text based search.

## Why This Architecture

The repository separates representation learning from workflow execution. This makes it easier to swap encoders, compare modeling choices, and keep downstream applications stable while the representation layer evolves.
