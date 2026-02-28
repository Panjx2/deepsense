from __future__ import annotations

import pandas as pd
import plotly.express as px


def compute_summary(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return df.describe(include="all").transpose(), df.isna().sum().to_frame("missing_values")


def build_numeric_distributions(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        return None
    long_df = df[numeric_cols].melt(var_name="feature", value_name="value")
    return px.histogram(
        long_df,
        x="value",
        facet_col="feature",
        facet_col_wrap=3,
        nbins=25,
        title="Numeric Feature Distributions",
    )


def build_numeric_correlation(df: pd.DataFrame):
    corr = df.select_dtypes(include="number").corr(numeric_only=True)
    if corr.empty:
        return None
    return px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Heatmap")


def build_target_breakdown(df: pd.DataFrame):
    preferred_targets = ["churn", "risk_band", "category"]
    target_col = next((col for col in preferred_targets if col in df.columns), None)
    if target_col is None:
        return None
    counts = df[target_col].value_counts(dropna=False).reset_index()
    counts.columns = [target_col, "count"]
    return px.bar(counts, x=target_col, y="count", title=f"{target_col} distribution")
