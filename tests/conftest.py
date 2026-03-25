"""Shared test configuration."""

import os

# Prevent OpenMP crash from PyTorch + FAISS loading different runtime libs
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
