# Sensitivity Lab (separate project)

This module provides a mini simulation lab for **retail_sales** parameter sensitivity analysis.

## What it does

It sweeps one parameter (`discount_cap`, `avg_units`, `ad_spend_scale`, or `weekend_lift`) over a grid and, for each run:

1. Regenerates synthetic retail sales data.
2. Computes:
   - `revenue_distribution_shift` (Wasserstein drift vs baseline revenue distribution)
   - `price_elasticity_estimate` (log-log slope of units sold vs effective price)
   - `weekend_effect_size` (Cohen's d in revenue: weekend vs weekday)
3. Produces a curve visualization (parameter -> metric).

## CLI usage

```bash
python -m sensitivity_lab.run_sweep \
  --parameter discount_cap \
  --start 0.1 \
  --stop 0.7 \
  --steps 10
```

Outputs are written to `sensitivity_lab/output/` as:

- `<parameter>_sweep.csv`
- `<parameter>_sweep.html`
