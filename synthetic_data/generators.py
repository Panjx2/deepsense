from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DatasetType:
    key: str
    label: str
    description: str
    default_method: str
    methods: tuple[str, ...]


DATASET_TYPES = {
    "customer_churn": DatasetType(
        key="customer_churn",
        label="Customer Churn",
        description="Binary classification dataset with usage, support, and billing behavior.",
        default_method="rule_based",
        methods=("rule_based", "distribution_based"),
    ),
    "retail_sales": DatasetType(
        key="retail_sales",
        label="Retail Sales",
        description="Daily retail transactions with product/category, discounting, and demand dynamics.",
        default_method="distribution_based",
        methods=("distribution_based", "rule_based"),
    ),
    "health_risk": DatasetType(
        key="health_risk",
        label="Health Risk",
        description="Patient biometrics with configurable risk-banding and prevalence shifts.",
        default_method="rule_based",
        methods=("rule_based", "distribution_based"),
    ),
}


DATASET_PARAMETERS: dict[str, dict[str, dict[str, Any]]] = {
    "customer_churn": {
        "monthly_charge_mean": {"type": "float", "min": 30.0, "max": 130.0, "default": 75.0, "step": 1.0},
        "ticket_intensity": {"type": "float", "min": 0.2, "max": 5.0, "default": 1.8, "step": 0.1},
        "monthly_contract_share": {"type": "float", "min": 0.1, "max": 0.9, "default": 0.55, "step": 0.01},
        "price_sensitivity": {"type": "float", "min": 0.0, "max": 0.06, "default": 0.02, "step": 0.005},
    },
    "retail_sales": {
        "discount_cap": {"type": "float", "min": 0.1, "max": 0.7, "default": 0.45, "step": 0.01},
        "avg_units": {"type": "float", "min": 1.0, "max": 10.0, "default": 3.0, "step": 0.2},
        "ad_spend_scale": {"type": "float", "min": 5.0, "max": 80.0, "default": 20.0, "step": 1.0},
        "weekend_lift": {"type": "float", "min": 0.0, "max": 80.0, "default": 18.0, "step": 1.0},
    },
    "health_risk": {
        "smoking_rate": {"type": "float", "min": 0.05, "max": 0.6, "default": 0.22, "step": 0.01},
        "exercise_bias": {"type": "float", "min": -1.0, "max": 2.0, "default": 0.0, "step": 0.1},
        "bmi_mean": {"type": "float", "min": 20.0, "max": 35.0, "default": 27.0, "step": 0.2},
        "risk_noise": {"type": "float", "min": 1.0, "max": 10.0, "default": 4.0, "step": 0.2},
    },
}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def _generate_customer_churn(rows: int, seed: int, method: str, **kwargs: Any) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    monthly_charge_mean = float(kwargs.get("monthly_charge_mean", 75.0))
    ticket_intensity = float(kwargs.get("ticket_intensity", 1.8))
    monthly_contract_share = float(kwargs.get("monthly_contract_share", 0.55))
    price_sensitivity = float(kwargs.get("price_sensitivity", 0.02))

    if method == "distribution_based":
        tenure_months = rng.gamma(shape=2.5, scale=12.0, size=rows).clip(1, 72).round().astype(int)
        monthly_charges = rng.lognormal(mean=np.log(max(monthly_charge_mean, 1.0)), sigma=0.25, size=rows).clip(15, 180)
    else:
        tenure_months = rng.integers(1, 72, rows)
        monthly_charges = rng.normal(monthly_charge_mean, 28, rows).clip(15, 180)

    support_tickets = rng.poisson(ticket_intensity, rows)
    remaining = max(1.0 - monthly_contract_share, 1e-6)
    one_year = remaining * 0.56
    two_year = remaining * 0.44
    contract = rng.choice(["month-to-month", "one-year", "two-year"], p=[monthly_contract_share, one_year, two_year], size=rows)
    payment_method = rng.choice(["credit-card", "bank-transfer", "e-wallet"], p=[0.45, 0.35, 0.20], size=rows)

    logit = (
        1.2
        + price_sensitivity * monthly_charges
        + 0.35 * support_tickets
        - 0.03 * tenure_months
        + 1.0 * (contract == "month-to-month")
        - 0.7 * (contract == "two-year")
    )
    churn = rng.binomial(1, _sigmoid(logit - 4))

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


def _generate_retail_sales(rows: int, seed: int, method: str, **kwargs: Any) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    discount_cap = float(kwargs.get("discount_cap", 0.45))
    avg_units = float(kwargs.get("avg_units", 3.0))
    ad_spend_scale = float(kwargs.get("ad_spend_scale", 20.0))
    weekend_lift = float(kwargs.get("weekend_lift", 18.0))

    categories = ["electronics", "apparel", "home", "beauty", "sports"]
    category = rng.choice(categories, p=[0.2, 0.25, 0.2, 0.15, 0.2], size=rows)
    discount = rng.uniform(0, discount_cap, rows)
    base_price = (rng.normal(60, 30, rows) if method == "rule_based" else rng.gamma(shape=3.0, scale=20.0, size=rows)).clip(5, 300)
    units_sold = rng.poisson(avg_units, rows) + 1
    ad_spend = rng.gamma(shape=2.0, scale=ad_spend_scale, size=rows)

    weekend = rng.binomial(1, 0.32, rows)
    holiday = rng.binomial(1, 0.08, rows)
    noise = rng.normal(0, 12, rows)
    revenue = (base_price * (1 - discount) * units_sold) + 1.5 * ad_spend + weekend_lift * weekend + 35 * holiday + noise

    dates = pd.to_datetime("2025-01-01") + pd.to_timedelta(rng.integers(0, 240, size=rows), unit="D")
    df = pd.DataFrame(
        {
            "date": dates,
            "category": category,
            "base_price": base_price.round(2),
            "discount_pct": (discount * 100).round(1),
            "units_sold": units_sold,
            "ad_spend": ad_spend.round(2),
            "is_weekend": weekend,
            "is_holiday": holiday,
            "revenue": revenue.round(2).clip(0),
        }
    )
    return df.sort_values("date", ignore_index=True)


def _generate_health_risk(rows: int, seed: int, method: str, **kwargs: Any) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    smoking_rate = float(kwargs.get("smoking_rate", 0.22))
    exercise_bias = float(kwargs.get("exercise_bias", 0.0))
    bmi_mean = float(kwargs.get("bmi_mean", 27.0))
    risk_noise = float(kwargs.get("risk_noise", 4.0))

    age = rng.integers(18, 85, rows)
    bmi = (rng.normal(bmi_mean, 5.5, rows) if method == "rule_based" else rng.lognormal(mean=np.log(max(bmi_mean, 1.0)), sigma=0.2, size=rows)).clip(15, 48)
    systolic_bp = rng.normal(123, 18, rows).clip(85, 210)
    cholesterol = rng.normal(198, 38, rows).clip(110, 380)
    smoker = rng.binomial(1, smoking_rate, rows)
    exercise_days = (rng.integers(0, 7, rows) + exercise_bias).clip(0, 6).round().astype(int)

    risk_score = (
        0.05 * age
        + 0.11 * bmi
        + 0.035 * systolic_bp
        + 0.01 * cholesterol
        + 11 * smoker
        - 1.2 * exercise_days
        + rng.normal(0, risk_noise, rows)
    )
    risk_band = pd.cut(risk_score, bins=[-np.inf, 25, 40, np.inf], labels=["low", "medium", "high"])

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


def get_dataset_parameters(dataset_type: str) -> dict[str, dict[str, Any]]:
    return DATASET_PARAMETERS.get(dataset_type, {})


def generate_dataset(dataset_type: str, rows: int = 500, seed: int = 42, method: str | None = None, **kwargs: Any) -> pd.DataFrame:
    if rows <= 0:
        raise ValueError("rows must be > 0")
    if dataset_type not in DATASET_TYPES:
        valid = ", ".join(DATASET_TYPES)
        raise ValueError(f"Unsupported dataset_type '{dataset_type}'. Expected one of: {valid}")

    selected_method = method or DATASET_TYPES[dataset_type].default_method
    if selected_method not in DATASET_TYPES[dataset_type].methods:
        raise ValueError(f"Unsupported method '{selected_method}' for {dataset_type}")

    if dataset_type == "customer_churn":
        return _generate_customer_churn(rows, seed, selected_method, **kwargs)
    if dataset_type == "retail_sales":
        return _generate_retail_sales(rows, seed, selected_method, **kwargs)
    return _generate_health_risk(rows, seed, selected_method, **kwargs)
