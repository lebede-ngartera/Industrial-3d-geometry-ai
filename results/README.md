# Results

This directory records verified outcomes and evaluation status for the GeoFusion AI repository.

## Verified Repository Results

### Validation Status

1. Test suite status
   37 tests passed during verification on Windows with Python 3.12.

2. Static validation
   `app.py` compiled successfully with `python -W error -m py_compile app.py`.

3. Notebook maturity
   All four notebooks contain substantive runnable content rather than placeholders.

4. Functional coverage
   The repository includes implemented workflows for:
   shape classification
   similarity retrieval
   anomaly detection
   cross modal search
   diffusion based generation

5. Public benchmark evidence
   The reproducible synthetic benchmark is defined in [experiments/public_benchmark_manifest.md](../experiments/public_benchmark_manifest.md)
   and the generated metrics are recorded in [public_metrics.md](public_metrics.md).

## What Is Not Claimed Yet

This repository does not currently publish domain benchmark tables from a proprietary industrial CAD corpus. For that reason, no claim is made here about production retrieval uplift, manufacturing defect reduction, or deployment scale inside a real PLM environment.

That is intentional. The public repository is meant to validate capability and rigor without publishing the full strategic value of deeper case studies, internal evaluations, or collaboration specific results.

## Recommended Next Benchmark Additions

1. Recall at K on domain specific part retrieval
2. Text to shape retrieval quality on engineering prompts
3. Inference latency by model family
4. Memory and index size for FAISS variants
5. Failure cases for thin structures and topologically complex parts
