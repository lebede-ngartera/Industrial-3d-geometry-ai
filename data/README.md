# Data Directory

This directory stores datasets used by GeoFusion AI. Data files are not tracked by git due to their size.

## Supported Datasets

### ModelNet40

40-class 3D object classification benchmark containing 12,311 CAD models.

Download:

```bash
python scripts/download_data.py --dataset modelnet40 --data-root data/raw
```

Source: https://modelnet.cs.princeton.edu/

### ShapeNet

Large-scale 3D shape repository with part segmentation annotations across 16 categories.

Download:

```bash
python scripts/download_data.py --dataset shapenet --data-root data/raw
```

Source: https://shapenet.org/

## Directory Structure After Download

```
data/
    raw/
        modelnet40/
            airplane/
            bathtub/
            ...
        shapenet/
            02691156/
            02958343/
            ...
```

## Custom Data

Place custom point cloud files in any subdirectory under data/. Supported formats: .npy, .npz, .txt, .ply.

Use `PointCloudDataset` from `geofusion.data.datasets` to load custom data.
