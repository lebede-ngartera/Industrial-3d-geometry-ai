# Contributing to GeoFusion AI

Thank you for your interest in contributing. This document describes the development workflow and conventions used in this project.


## Development Setup

```bash
git clone https://github.com/lebede-ngartera/GeoFusion-AI.git
cd GeoFusion-AI
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate      # Windows
pip install -e ".[dev]"
```


## Code Style

This project uses Black for formatting and Ruff for linting. Both are configured in pyproject.toml with a line length of 100 characters.

```bash
black geofusion/ scripts/ tests/
ruff check geofusion/ scripts/ tests/
```


## Running Tests

```bash
pytest tests/ -v --tb=short
```

All tests use synthetic data and should pass without downloading external datasets. If you add a new module under geofusion/, add corresponding tests in tests/.


## Project Conventions

Source code for the library lives under geofusion/. Training and evaluation scripts live in scripts/. Configuration is managed through YAML files in configs/. The Streamlit dashboard is in app.py.

All model architectures should output embeddings of dimension embed_dim (default 256) to remain compatible with the shared embedding space and downstream retrieval workflows.


## Submitting Changes

1. Create a branch from main with a descriptive name.
2. Make your changes. Add tests for new functionality.
3. Run Black, Ruff, and pytest before committing.
4. Open a pull request with a clear description of what changed and why.
