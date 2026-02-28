"""Synthetic data generation and EDA utilities."""

from .generators import DATASET_TYPES, generate_dataset
from .eda import (
    build_numeric_correlation,
    build_numeric_distributions,
    build_target_breakdown,
    compute_summary,
)

__all__ = [
    "DATASET_TYPES",
    "generate_dataset",
    "compute_summary",
    "build_numeric_distributions",
    "build_numeric_correlation",
    "build_target_breakdown",
]
