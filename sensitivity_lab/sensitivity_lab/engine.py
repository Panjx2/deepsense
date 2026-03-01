from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Iterable
import sys

import numpy as np
import pandas as pd

if find_spec("synthetic_data") is None:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from synthetic_data.generators import generate_dataset


@dataclass(frozen=True)
class SweepResult:
    parameter: str
    baseline_value: float
    results: pd.DataFrame


def _wasserstein_1d(x: np.ndarray, y: np.ndarray, points: int = 200) -> float:
    """Approximate 1D Wasserstein distance via quantile integration."""
    q = np.linspace(0, 1, points)
    xq = np.quantile(x, q)
    yq = np.quantile(y, q)
    return float(np.trapezoid(np.abs(xq - yq), q))


def _price_elasticity(df: pd.DataFrame) -> float:
    effective_price = (df["base_price"] * (1 - df["discount_pct"] / 100.0)).clip(lower=1e-6)
    units = df["units_sold"].clip(lower=1e-6)

    x = np.log(effective_price.to_numpy())
    y = np.log(units.to_numpy())

    slope, _intercept = np.polyfit(x, y, deg=1)
    return float(slope)


def _weekend_effect_size(df: pd.DataFrame) -> float:
    weekend = df.loc[df["is_weekend"] == 1, "revenue"].to_numpy()
    weekday = df.loc[df["is_weekend"] == 0, "revenue"].to_numpy()
    if len(weekend) < 2 or len(weekday) < 2:
        return 0.0

    weekend_mean = weekend.mean()
    weekday_mean = weekday.mean()

    pooled_std = np.sqrt(((len(weekend) - 1) * weekend.var(ddof=1) + (len(weekday) - 1) * weekday.var(ddof=1)) / (len(weekend) + len(weekday) - 2))
    if pooled_std == 0:
        return 0.0

    return float((weekend_mean - weekday_mean) / pooled_std)


def run_parameter_sweep(
    parameter: str,
    values: Iterable[float],
    *,
    rows: int = 2500,
    seed: int = 42,
    method: str = "distribution_based",
    baseline_params: dict[str, float] | None = None,
) -> SweepResult:
    """Run sensitivity analysis for retail_sales by sweeping one parameter."""
    baseline_params = dict(baseline_params or {})

    baseline_df = generate_dataset(
        "retail_sales",
        rows=rows,
        seed=seed,
        method=method,
        **baseline_params,
    )
    baseline_value = float(baseline_params.get(parameter, np.nan))
    baseline_revenue = baseline_df["revenue"].to_numpy()

    records: list[dict[str, float]] = []
    for idx, value in enumerate(values):
        params = dict(baseline_params)
        params[parameter] = float(value)
        df = generate_dataset(
            "retail_sales",
            rows=rows,
            seed=seed + idx + 1,
            method=method,
            **params,
        )

        records.append(
            {
                "parameter": parameter,
                "value": float(value),
                "revenue_distribution_shift": _wasserstein_1d(baseline_revenue, df["revenue"].to_numpy()),
                "price_elasticity_estimate": _price_elasticity(df),
                "weekend_effect_size": _weekend_effect_size(df),
            }
        )

    result_df = pd.DataFrame(records).sort_values("value", ignore_index=True)
    return SweepResult(parameter=parameter, baseline_value=baseline_value, results=result_df)
