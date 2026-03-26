# Public Metrics

These metrics are generated from the reproducible synthetic public subset used by `scripts/public_eval.py`.

| Area | Metric | Value | Notes |
| ---- | ------ | ----- | ----- |
| Classification | Top-1 accuracy | 0.2000 | Synthetic public subset |
| Classification | Top-5 accuracy | 1.0000 | Synthetic public subset |
| Retrieval | Recall@1 | 1.0000 | Test queries vs train gallery |
| Retrieval | Recall@5 | 1.0000 | Test queries vs train gallery |
| Retrieval | Recall@10 | 1.0000 | Test queries vs train gallery |
| Retrieval | Precision@5 | 1.0000 | Test queries vs train gallery |
| Retrieval | mAP | 1.0000 | Test queries vs train gallery |
| Cross-modal | G2T Recall@5 | 1.0000 | Test geometry vs projected text |
| Cross-modal | T2G Recall@5 | 1.0000 | Test text vs geometry |
| Anomaly workflow | Warning threshold | 1.843298 | Calibrated on normal subset |
| Anomaly workflow | Critical threshold | 1.843348 | Calibrated on normal subset |
| Anomaly workflow | Warning or higher count | 3 | Mixed public subset |
| Anomaly workflow | Critical count | 3 | Mixed public subset |
