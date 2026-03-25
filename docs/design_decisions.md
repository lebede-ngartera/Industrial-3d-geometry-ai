# Design Decisions

## 1. Why point clouds are the primary runnable baseline

Point clouds provide a practical common denominator across public datasets and synthetic generation. They are easier to standardize for experimentation than full CAD kernel representations while still preserving meaningful geometry for retrieval and anomaly detection.

## 2. Why both PointNet++ and DGCNN are included

The project is not tied to a single geometric assumption.

PointNet++ is strong for hierarchical local geometry aggregation.
DGCNN is strong when local neighborhood relations matter more explicitly.
Keeping both models enables a more honest comparison.

## 3. Why a shared embedding space was chosen

The repository targets workflows, not only model training. Similarity search, text retrieval, and metadata fusion become simpler once all modalities can be compared in one aligned latent space.

## 4. Why FAISS is used for retrieval

A realistic engineering search system needs scalable nearest neighbor infrastructure. FAISS provides production relevant indexing strategies while remaining easy to benchmark locally.

## 5. Why anomaly detection combines multiple signals

Reconstruction error alone can miss density anomalies. Density scoring alone can miss structurally meaningful reconstruction failures. An ensemble gives more robust behavior for exploratory industrial use.

## 6. Why the repository includes notebooks and a product style app

Senior level presentation requires more than source code. The notebooks support technical exploration. The Streamlit app shows how the components behave as an end user facing workflow.

## 7. What was intentionally not overstated

The repository does not claim production deployment on an automotive CAD corpus. It also does not claim benchmark superiority without a published domain dataset. The current project is a well structured and defensible prototype.
