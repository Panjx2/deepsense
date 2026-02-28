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
- **Rich EDA workflow in-app**
  - KPI cards and data preview
  - Summary stats + missing-value report
  - Numeric/categorical profiling
  - Histograms, correlation heatmap, scatter matrix, target breakdown, and time trend (when applicable)
- **Export-ready output**
  - Download generated data as CSV or JSON

## Project structure

- `app.py`: Streamlit dashboard UI
- `synthetic_data/generators.py`: synthetic generation logic + parameter metadata
- `synthetic_data/eda.py`: EDA utilities and chart builders
- `tests/test_generators.py`: generation behavior tests

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
