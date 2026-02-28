import pandas as pd

from synthetic_data.generators import DATASET_TYPES, generate_dataset


def test_all_dataset_types_generate_rows():
    for dataset_type in DATASET_TYPES:
        df = generate_dataset(dataset_type, rows=120, seed=7)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 120


def test_invalid_dataset_type_raises():
    try:
        generate_dataset("unknown")
    except ValueError as exc:
        assert "Unsupported dataset_type" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
