# deepsense

Interactive synthetic data generation and EDA studio built with Streamlit.

## What makes this project useful

- **Multiple synthetic dataset blueprints**
  - Customer churn
  - Retail sales
  - Health risk scoring
- **Adjustable generation strategy**
  - Per-dataset generation methods (rule-based and distribution-based)
  - Tunable controls for behavior, prevalence, and noise
- **Explainable generation**
  - Churn logit formula exposed in UI
  - Expected churn-rate preview before generation
  - Export of data plus generation config
- **Quality engineering layer**
  - Schema/range validation checks
  - Outlier-rate profiling
  - Constraint violation counts
  - Method-to-method distribution distance (KS, Wasserstein, JS divergence)
- **Rich EDA workflow in-app**
  - KPI cards and data preview
  - Summary stats + missing-value report
  - Numeric/categorical profiling
  - Histograms, correlation heatmap, scatter matrix, target breakdown, and time trend (when applicable)
- **Export-ready output**
What’s “missing” for a Synthetic Data Engineer portfolio (high leverage fixes)
1) Add synthetic data quality checks (biggest upgrade)

Right now you generate + visualize. Add a “Quality” tab with metrics like:

    schema validation (types/ranges)

    outlier rate

    constraint violations (e.g., discount_pct in [0, 100])

    simple distribution distance between methods (KS test / Wasserstein for numeric, JS divergence for categorical)

This is the difference between “cool demo” and “engineering tool”.
2) Make generation logic more explainable / controllable

Your churn generator uses a logistic-style logit (nice!) but users can’t see what’s happening.
Add:

    show the churn logit formula in the UI

    show “expected churn rate” before generation (approx)

    optionally export the generation config with the dataset

Generator credibility matters.
3) One small realism improvemenhttps://github.com/Panjx2/deepsense/pull/3t: categorical dependence

Some categorical columns are independent draws (payment method doesn’t depend on contract, etc.). That’s fine, but synthetic data folks often expect at least one dependency.
Add 1–2 simple conditional probabilities:

    payment method distribution depends on contract type

    support ticket intensity depends on tenure band

This will make the data feel more realistic without making code complex.
4) Tests: add 2–3 “property-based” checks

Keep it lightweight but add assertions like:

    discount_pct always within bounds

    tenure_months within [1, 72]

    churn is binary

    no negative revenue
  - Download generated data as CSV, JSON, or JSON-with-config bundle

## Project structure

- `app.py`: Streamlit dashboard UI
- `synthetic_data/generators.py`: synthetic generation logic + parameter metadata
- `synthetic_data/eda.py`: EDA utilities and chart builders
- `synthetic_data/quality.py`: schema validation and distribution-distance quality metrics
- `tests/test_generators.py`: generation behavior + property checks

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Tests

```bash
pytest -q
```
