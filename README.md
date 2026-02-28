# deepsense

Interactive synthetic data generation and EDA studio built with Streamlit.

## What makes this portfolio project strong

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
