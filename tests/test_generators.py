import pandas as pd
import pytest

from synthetic_data.generators import DATASET_TYPES, estimate_expected_churn_rate, generate_dataset


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


def test_retail_discount_and_revenue_constraints_hold():
    df = generate_dataset("retail_sales", rows=300, seed=4)
    assert df["discount_pct"].between(0, 100).all()
    assert (df["revenue"] >= 0).all()


def test_churn_core_properties_hold():
    df = generate_dataset("customer_churn", rows=300, seed=9)
    assert df["tenure_months"].between(1, 72).all()
    assert set(df["churn"].unique()).issubset({0, 1})


def test_expected_churn_rate_is_probability():
    rate = estimate_expected_churn_rate(rows=1500, seed=11)
    assert 0 <= rate <= 1
