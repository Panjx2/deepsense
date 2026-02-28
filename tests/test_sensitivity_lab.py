import numpy as np

from sensitivity_lab.engine import run_parameter_sweep
from sensitivity_lab.visualization import build_sensitivity_figure


def test_run_parameter_sweep_outputs_expected_columns():
    sweep = run_parameter_sweep(
        parameter="discount_cap",
        values=np.linspace(0.1, 0.7, 5),
        rows=500,
        seed=123,
        baseline_params={
            "discount_cap": 0.45,
            "avg_units": 3.0,
            "ad_spend_scale": 20.0,
            "weekend_lift": 18.0,
        },
    )

    assert sweep.parameter == "discount_cap"
    assert len(sweep.results) == 5
    assert set(sweep.results.columns) == {
        "parameter",
        "value",
        "revenue_distribution_shift",
        "price_elasticity_estimate",
        "weekend_effect_size",
    }
    assert (sweep.results["revenue_distribution_shift"] >= 0).all()


def test_build_sensitivity_figure_has_three_panels():
    sweep = run_parameter_sweep(
        parameter="avg_units",
        values=[1.0, 3.0, 5.0],
        rows=400,
        seed=999,
        baseline_params={
            "discount_cap": 0.45,
            "avg_units": 3.0,
            "ad_spend_scale": 20.0,
            "weekend_lift": 18.0,
        },
    )

    fig = build_sensitivity_figure(sweep.results, parameter="avg_units")
    assert len(fig.data) == 3
