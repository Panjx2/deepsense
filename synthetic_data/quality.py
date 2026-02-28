from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


DATASET_CONSTRAINTS: dict[str, dict[str, Any]] = {
    "customer_churn": {
        "ranges": {
            "tenure_months": (1, 72),
            "monthly_charges": (15, 180),
            "support_tickets": (0, None),
            "churn": (0, 1),
        },
        "categorical_allowed": {
            "contract": {"month-to-month", "one-year", "two-year"},
            "payment_method": {"credit-card", "bank-transfer", "e-wallet"},
        },
    },
    "retail_sales": {
        "ranges": {
            "discount_pct": (0, 100),
            "base_price": (0, None),
            "units_sold": (1, None),
            "ad_spend": (0, None),
            "revenue": (0, None),
            "is_weekend": (0, 1),
            "is_holiday": (0, 1),
        }
    },
    "health_risk": {
        "ranges": {
            "age": (18, 85),
            "bmi": (15, 48),
            "systolic_bp": (85, 210),
            "cholesterol": (110, 380),
            "smoker": (0, 1),
            "exercise_days": (0, 6),
        },
        "categorical_allowed": {"risk_band": {"low", "medium", "high"}},
    },
}


def _ks_statistic(x: np.ndarray, y: np.ndarray) -> float:
    x_sorted = np.sort(x)
    y_sorted = np.sort(y)
    grid = np.sort(np.concatenate([x_sorted, y_sorted]))
    cdf_x = np.searchsorted(x_sorted, grid, side="right") / len(x_sorted)
    cdf_y = np.searchsorted(y_sorted, grid, side="right") / len(y_sorted)
    return float(np.max(np.abs(cdf_x - cdf_y)))


def _wasserstein_1d(x: np.ndarray, y: np.ndarray) -> float:
    x_sorted = np.sort(x)
    y_sorted = np.sort(y)
    n = max(len(x_sorted), len(y_sorted))
    q = np.linspace(0, 1, n)
    x_q = np.quantile(x_sorted, q)
    y_q = np.quantile(y_sorted, q)
    return float(np.mean(np.abs(x_q - y_q)))


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    eps = 1e-12
    p = np.clip(p, eps, 1)
    q = np.clip(q, eps, 1)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    return float(0.5 * (kl_pm + kl_qm))


def validate_schema(df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
    checks: list[dict[str, Any]] = []
    cfg = DATASET_CONSTRAINTS.get(dataset_type, {})

    for col, (low, high) in cfg.get("ranges", {}).items():
        if col not in df.columns:
            checks.append({"check": f"{col} present", "status": "fail", "violations": len(df)})
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        invalid = series.isna()
        if low is not None:
            invalid |= series < low
        if high is not None:
            invalid |= series > high
        checks.append({"check": f"{col} in [{low}, {high}]", "status": "pass" if invalid.sum() == 0 else "fail", "violations": int(invalid.sum())})

    for col, allowed in cfg.get("categorical_allowed", {}).items():
        if col not in df.columns:
            checks.append({"check": f"{col} present", "status": "fail", "violations": len(df)})
            continue
        invalid = ~df[col].astype(str).isin({str(x) for x in allowed})
        checks.append({"check": f"{col} allowed categories", "status": "pass" if invalid.sum() == 0 else "fail", "violations": int(invalid.sum())})

    return pd.DataFrame(checks)


def compute_outlier_rate(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    numeric = df.select_dtypes(include="number")
    for col in numeric.columns:
        q1 = numeric[col].quantile(0.25)
        q3 = numeric[col].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            outlier_rate = 0.0
        else:
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outlier_rate = ((numeric[col] < lower) | (numeric[col] > upper)).mean()
        rows.append({"feature": col, "outlier_rate": round(float(outlier_rate), 4)})
    return pd.DataFrame(rows)


def compute_distribution_distance(df_a: pd.DataFrame, df_b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    num_rows = []
    cat_rows = []

    numeric_cols = sorted(set(df_a.select_dtypes(include="number").columns) & set(df_b.select_dtypes(include="number").columns))
    for col in numeric_cols:
        x = pd.to_numeric(df_a[col], errors="coerce").dropna().to_numpy()
        y = pd.to_numeric(df_b[col], errors="coerce").dropna().to_numpy()
        if len(x) > 1 and len(y) > 1:
            num_rows.append({"feature": col, "ks_stat": round(_ks_statistic(x, y), 4), "wasserstein": round(_wasserstein_1d(x, y), 4)})

    categorical_cols = sorted(set(df_a.select_dtypes(exclude="number").columns) & set(df_b.select_dtypes(exclude="number").columns))
    for col in categorical_cols:
        a_freq = df_a[col].astype(str).value_counts(normalize=True)
        b_freq = df_b[col].astype(str).value_counts(normalize=True)
        all_idx = sorted(set(a_freq.index) | set(b_freq.index))
        p = np.array([a_freq.get(k, 0.0) for k in all_idx])
        q = np.array([b_freq.get(k, 0.0) for k in all_idx])
        cat_rows.append({"feature": col, "js_divergence": round(_js_divergence(p, q), 4)})

    return pd.DataFrame(num_rows), pd.DataFrame(cat_rows)


def summarize_quality(df: pd.DataFrame, dataset_type: str) -> dict[str, Any]:
    schema_df = validate_schema(df, dataset_type)
    outlier_df = compute_outlier_rate(df)

    violation_total = int(schema_df["violations"].sum()) if not schema_df.empty else 0
    outlier_avg = float(outlier_df["outlier_rate"].mean()) if not outlier_df.empty else 0.0

    return {
        "schema_checks": schema_df,
        "outliers": outlier_df,
        "violation_total": violation_total,
        "avg_outlier_rate": round(outlier_avg, 4),
    }
