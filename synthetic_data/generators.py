from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DatasetType:
    key: str
    label: str
    description: str


DATASET_TYPES = {
    "customer_churn": DatasetType(
        key="customer_churn",
        label="Customer Churn",
        description="Binary classification dataset with usage, support, and billing features.",
    ),
    "retail_sales": DatasetType(
        key="retail_sales",
        label="Retail Sales",
        description="Daily retail transactions with product category and pricing signals.",
    ),
    "health_risk": DatasetType(
        key="health_risk",
        label="Health Risk",
        description="Patient profile dataset with biometrics and a risk score.",
    ),
}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def _generate_customer_churn(rows: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tenure_months = rng.integers(1, 72, rows)
    monthly_charges = rng.normal(75, 28, rows).clip(15, 180)
    support_tickets = rng.poisson(1.8, rows)
    contract = rng.choice(["month-to-month", "one-year", "two-year"], p=[0.55, 0.25, 0.20], size=rows)
    payment_method = rng.choice(["credit-card", "bank-transfer", "e-wallet"], p=[0.45, 0.35, 0.20], size=rows)

    logit = (
        1.2
        + 0.02 * monthly_charges
        + 0.35 * support_tickets
        - 0.03 * tenure_months
        + 1.0 * (contract == "month-to-month")
        - 0.7 * (contract == "two-year")
    )
    churn_probability = _sigmoid(logit - 4)
    churn = rng.binomial(1, churn_probability)

    return pd.DataFrame(
        {
            "tenure_months": tenure_months,
            "monthly_charges": monthly_charges.round(2),
            "support_tickets": support_tickets,
            "contract": contract,
            "payment_method": payment_method,
            "churn": churn,
        }
    )


def _generate_retail_sales(rows: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    categories = ["electronics", "apparel", "home", "beauty", "sports"]
    category = rng.choice(categories, p=[0.2, 0.25, 0.2, 0.15, 0.2], size=rows)
    discount = rng.uniform(0, 0.45, rows)
    base_price = rng.normal(60, 30, rows).clip(5, 300)
    units_sold = rng.poisson(3, rows) + 1
    ad_spend = rng.gamma(shape=2.0, scale=20.0, size=rows)

    weekend = rng.binomial(1, 0.32, rows)
    noise = rng.normal(0, 12, rows)
    revenue = (base_price * (1 - discount) * units_sold) + 1.5 * ad_spend + 18 * weekend + noise

    dates = pd.to_datetime("2025-01-01") + pd.to_timedelta(rng.integers(0, 180, size=rows), unit="D")

    return pd.DataFrame(
        {
            "date": dates,
            "category": category,
            "base_price": base_price.round(2),
            "discount_pct": (discount * 100).round(1),
            "units_sold": units_sold,
            "ad_spend": ad_spend.round(2),
            "is_weekend": weekend,
            "revenue": revenue.round(2).clip(0),
        }
    ).sort_values("date", ignore_index=True)


def _generate_health_risk(rows: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 85, rows)
    bmi = rng.normal(27, 5.5, rows).clip(15, 48)
    systolic_bp = rng.normal(123, 18, rows).clip(85, 210)
    cholesterol = rng.normal(198, 38, rows).clip(110, 380)
    smoker = rng.binomial(1, 0.22, rows)
    exercise_days = rng.integers(0, 7, rows)

    risk_score = (
        0.05 * age
        + 0.11 * bmi
        + 0.035 * systolic_bp
        + 0.01 * cholesterol
        + 11 * smoker
        - 1.2 * exercise_days
        + rng.normal(0, 4, rows)
    )
    risk_band = pd.cut(
        risk_score,
        bins=[-np.inf, 25, 40, np.inf],
        labels=["low", "medium", "high"],
    )

    return pd.DataFrame(
        {
            "age": age,
            "bmi": bmi.round(2),
            "systolic_bp": systolic_bp.round(1),
            "cholesterol": cholesterol.round(1),
            "smoker": smoker,
            "exercise_days": exercise_days,
            "risk_score": risk_score.round(2),
            "risk_band": risk_band,
        }
    )


def generate_dataset(dataset_type: str, rows: int = 500, seed: int = 42) -> pd.DataFrame:
    if rows <= 0:
        raise ValueError("rows must be > 0")
    if dataset_type == "customer_churn":
        return _generate_customer_churn(rows, seed)
    if dataset_type == "retail_sales":
        return _generate_retail_sales(rows, seed)
    if dataset_type == "health_risk":
        return _generate_health_risk(rows, seed)

    valid = ", ".join(DATASET_TYPES)
    raise ValueError(f"Unsupported dataset_type '{dataset_type}'. Expected one of: {valid}")
