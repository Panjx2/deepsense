import pandas as pd
import pytest

from synthetic_data.generators import DATASET_TYPES, generate_dataset


def test_all_dataset_types_generate_rows_for_default_method():
    for dataset_type, cfg in DATASET_TYPES.items():
        df = generate_dataset(dataset_type, rows=120, seed=7, method=cfg.default_method)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 120


def test_all_dataset_types_support_all_declared_methods():
    for dataset_type, cfg in DATASET_TYPES.items():
        for method in cfg.methods:
            df = generate_dataset(dataset_type, rows=25, seed=1, method=method)
            assert len(df) == 25


def test_invalid_dataset_type_raises():
    with pytest.raises(ValueError, match="Unsupported dataset_type"):
        generate_dataset("unknown")


def test_invalid_method_raises():
    with pytest.raises(ValueError, match="Unsupported method"):
        generate_dataset("customer_churn", method="ai_model")
