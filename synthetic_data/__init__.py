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
from .generators import (
    CHURN_LOGIT_FORMULA,
    DATASET_TYPES,
    estimate_expected_churn_rate,
    generate_dataset,
    get_dataset_parameters,
)
from .quality import compute_distribution_distance, summarize_quality, validate_schema

__all__ = [
    "DATASET_TYPES",
    "CHURN_LOGIT_FORMULA",
    "generate_dataset",
    "get_dataset_parameters",
    "estimate_expected_churn_rate",
    "compute_summary",
    "compute_column_profiles",
    "build_numeric_distributions",
    "build_numeric_correlation",
    "build_target_breakdown",
    "build_time_trend",
    "build_scatter_matrix",
    "validate_schema",
    "summarize_quality",
    "compute_distribution_distance",
]
