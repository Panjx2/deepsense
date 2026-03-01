from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from sensitivity_lab.engine import run_parameter_sweep
from sensitivity_lab.visualization import build_sensitivity_figure

DEFAULT_BASELINE = {
    "discount_cap": 0.45,
    "avg_units": 3.0,
    "ad_spend_scale": 20.0,
    "weekend_lift": 18.0,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Retail sales parameter sensitivity analysis.")
    parser.add_argument("--parameter", required=True, choices=tuple(DEFAULT_BASELINE), help="Parameter to sweep")
    parser.add_argument("--start", type=float, required=True, help="Grid start value")
    parser.add_argument("--stop", type=float, required=True, help="Grid end value")
    parser.add_argument("--steps", type=int, default=12, help="Number of grid points")
    parser.add_argument("--rows", type=int, default=3000, help="Rows per simulation")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--method", default="distribution_based", choices=["distribution_based", "rule_based"], help="Generator method")
    parser.add_argument("--outdir", default="sensitivity_lab/output", help="Output directory")
    args = parser.parse_args()

    grid = np.linspace(args.start, args.stop, args.steps)
    sweep = run_parameter_sweep(
        args.parameter,
        grid,
        rows=args.rows,
        seed=args.seed,
        method=args.method,
        baseline_params=DEFAULT_BASELINE,
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / f"{args.parameter}_sweep.csv"
    html_path = outdir / f"{args.parameter}_sweep.html"

    sweep.results.to_csv(csv_path, index=False)
    fig = build_sensitivity_figure(sweep.results, parameter=args.parameter)
    fig.write_html(html_path)

    print(f"Saved metrics: {csv_path}")
    print(f"Saved visualization: {html_path}")


if __name__ == "__main__":
    main()
