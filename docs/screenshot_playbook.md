# Screenshot Playbook

This playbook documents the four real screenshots now used in the public README and how to recapture them consistently.

## Capture Setup

Run the demo locally:

```bash
streamlit run app.py
```

Use these baseline settings before each capture:

- Browser width: 1440 to 1600 px
- Browser zoom: 90% to 100%
- Theme: default light theme for clean GitHub rendering
- Crop target: app content only, not browser chrome
- Export format: PNG

## Shot List

### 1. Overview dashboard

- Streamlit module: `Overview`
- Keep visible: branded header, workflow summary table, and the five synthetic shape panels
- Use for: README visual overview and LinkedIn hero crop
- Target file: `docs/figures/demo_interface_snapshot.png`

### 2. Shape classification results

- Streamlit module: `Shape Classification`
- Suggested controls: 66 samples per class, 512 points, embed dim 64, 4 epochs, learning rate 1e-3, batch size 16
- Trigger: `Train & Evaluate`
- Keep visible: synthetic-demo note, test accuracy banner, details panel, training loss chart, per-class accuracy chart
- Use for: README proof of model execution
- Target file: `docs/figures/classification_training_snapshot.png`

### 3. Similarity search results

- Streamlit module: `Similarity Search`
- Suggested controls: 20 samples per class, 512 points, query `Sphere` or `Cylinder`, top-k 5
- Trigger: `Build Index & Search`
- Keep visible: query panel and first five ranked results
- Use for: README retrieval proof and LinkedIn technical slide
- Target file: `docs/figures/retrieval_query_vs_results.png`

### 4. Anomaly review results

- Streamlit module: `Anomaly Review`
- Suggested controls: normal shape `Sphere`, anomaly type `Random noise`, normal samples 8, anomaly samples 4
- Trigger: `Run Anomaly Detection`
- Keep visible: normal vs anomaly panels, score distribution, summary metrics
- Use for: README anomaly proof and LinkedIn methods slide
- Target file: `docs/figures/anomaly_normal_vs_anomalous.png`

## Refinement Rules

- Keep one visual style across all four screenshots: same browser width, same crop margins, same light theme
- Remove browser tabs, bookmarks bar, and OS chrome unless the UI frame is intentionally part of the composition
- Prefer horizontal crops for README and save a second square or 1.91:1 crop for LinkedIn
- Keep text readable at thumbnail size; if a chart is too dense, crop tighter instead of shrinking the whole frame
- Do not add decorative overlays inside the product UI capture

## Export Variants

For each captured master PNG, export two derived versions:

- GitHub README: 1600 px wide PNG, light compression, no text overlays
- LinkedIn post: 1200 x 627 PNG or JPG, optionally with a short headline strip added outside the UI region

## Swap-In Step

If the four PNGs are recaptured:

1. Replace the stand-in SVG references in `README.md` with the PNG files.
2. Keep the same basenames where possible to avoid unnecessary README churn.
3. Recheck the README on GitHub dark and light backgrounds before pushing.