# deepsense

Interactive synthetic data generation + EDA dashboard built with Streamlit.

## Features

- Generate synthetic datasets for:
  - Customer churn
  - Retail sales
  - Health risk scoring
- Adjust generation parameters (rows + random seed)
- Explore EDA outputs:
  - Summary statistics
  - Missing-value report
  - Numeric feature distributions
  - Correlation heatmap
  - Target/category breakdown chart
- Download the generated dataset as CSV

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
