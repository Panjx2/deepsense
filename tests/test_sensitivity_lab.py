import numpy as np
import subprocess
import sys

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


def test_cli_module_execution_from_sensitivity_lab_directory(tmp_path):
    outdir = tmp_path / "sweep_output"
    cmd = [
        sys.executable,
        "-m",
        "sensitivity_lab.run_sweep",
        "--parameter",
        "discount_cap",
        "--start",
        "0.1",
        "--stop",
        "0.2",
        "--steps",
        "3",
        "--rows",
        "100",
        "--outdir",
        str(outdir),
    ]

    completed = subprocess.run(cmd, cwd="sensitivity_lab", check=True, capture_output=True, text=True)
    assert "Saved metrics" in completed.stdout
    assert (outdir / "discount_cap_sweep.csv").exists()
    assert (outdir / "discount_cap_sweep.html").exists()
