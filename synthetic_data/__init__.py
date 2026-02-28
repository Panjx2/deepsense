"""Synthetic data generation and EDA utilities."""

from .eda import (
    build_numeric_correlation,
    build_numeric_distributions,
    build_scatter_matrix,
    build_target_breakdown,
    build_time_trend,
    compute_column_profiles,
    compute_summary,
)
from .generators import DATASET_TYPES, generate_dataset, get_dataset_parameters

__all__ = [
    "DATASET_TYPES",
    "generate_dataset",
    "get_dataset_parameters",
    "compute_summary",
    "compute_column_profiles",
    "build_numeric_distributions",
    "build_numeric_correlation",
    "build_target_breakdown",
    "build_time_trend",
    "build_scatter_matrix",
]
