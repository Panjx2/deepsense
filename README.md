# DeepSense — Synthetic Data Studio

DeepSense is an interactive **Streamlit app + Python toolkit** for generating synthetic tabular data, profiling it, and validating quality in one place.

It is designed for portfolio demos, ML experimentation, and rapid prototyping where you need realistic-enough data structures without using production data.

---

## Why DeepSense exists

Most synthetic data demos stop at “generate a CSV.” DeepSense focuses on a fuller workflow:

- Generate multiple domain datasets.
- Tune generation behavior with configurable controls.
- Inspect distributions and feature relationships in-app.
- Evaluate quality with schema checks, outlier rates, and method-to-method drift metrics.
- Export both data and reproducible generation configuration.

---

## Core capabilities

### 1) Multiple dataset blueprints

DeepSense currently supports three synthetic domains:

- **Customer Churn** (classification-style behavioral dataset)
- **Retail Sales** (transaction/time-series flavored dataset)
- **Health Risk** (patient biometrics + risk scoring)

Each dataset has:

- A label and description.
- A default generation method.
- A set of supported generation methods.

### 2) Two generation strategies per dataset

For each dataset type, DeepSense supports:

- **`rule_based`** generation
- **`distribution_based`** generation

This lets you compare outcomes when generation logic is hand-shaped (rules/heuristics) vs distribution-driven.

### 3) Explainability for churn generation

For the customer churn dataset, the app exposes:

- The exact churn logit formula used.
- An **expected churn rate estimate** before generating a full dataset.

### 4) EDA in the same interface

Built-in exploratory analysis includes:

- Summary statistics + missing value report
- Numeric/categorical column profiles
- Histograms (faceted by numeric feature)
- Correlation heatmap
- Scatter matrix
- Target/category breakdown chart
- Revenue trend chart (when `date` + `revenue` are present)

### 5) Quality and validation layer

DeepSense includes a dedicated quality workflow:

- **Schema/range checks** per dataset
- **Allowed category checks** for categorical fields
- **IQR outlier rates** by numeric feature
- **Method distance comparison** between generation methods using:
  - KS statistic (numeric)
  - 1D Wasserstein distance (numeric)
  - Jensen–Shannon divergence (categorical)

### 6) Export options for reproducibility

You can export:

- CSV
- JSON records
- Combined JSON bundle containing:
  - generation config
  - generated rows

---

## Architecture at a glance

```text
app.py
└── synthetic_data/
    ├── generators.py   # dataset schemas, parameters, and generation logic
    ├── eda.py          # profiling + plot builders
    ├── quality.py      # validation + distance metrics
    └── __init__.py     # public API exports
```

Tests:

```text
tests/test_generators.py
```

---



## Sensitivity simulation lab (retail)

A separate mini-project lives under `sensitivity_lab/` for parameter sensitivity analysis on retail data generation.
It supports grid sweeps over `discount_cap`, `avg_units`, `ad_spend_scale`, and `weekend_lift`, and outputs both metrics and an HTML visualization.

Run example:

```bash
python -m sensitivity_lab.run_sweep --parameter discount_cap --start 0.1 --stop 0.7 --steps 10
```

## Installation

### Prerequisites

- Python 3.10+
- `pip`

### Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run the Streamlit app

```bash
streamlit run app.py
```

Once started, open the local URL shown in your terminal (typically `http://localhost:8501`).

---

## Using the app

### Sidebar controls

- **Dataset type**: choose churn, retail, or health.
- **Generation method**: select `rule_based` or `distribution_based` (available methods depend on dataset).
- **Rows**: choose sample size.
- **Random seed**: deterministic reproducibility.
- **Advanced dataset parameters**: domain-specific sliders/inputs.

For churn, a live **Expected churn rate** metric appears before generation.

### Main tabs

- **Data Preview**
  - First 100 rows
  - JSON generation config
  - Churn formula display (for churn dataset)
- **Profiling**
  - Summary stats
  - Missing values
  - Numeric and categorical profile tables
- **Visual EDA**
  - Distribution, correlation, scatter, breakdown, and trend charts
- **Quality**
  - Constraint and schema checks
  - Outlier rates
  - Method distance comparison against alternate generation method
- **Export**
  - CSV, JSON, and JSON-with-config downloads

---

## Dataset reference

## 1) Customer Churn

Columns:

- `tenure_months`
- `monthly_charges`
- `support_tickets`
- `contract`
- `payment_method`
- `churn`

Notable modeling details:

- Contract type is sampled with configurable monthly-contract share.
- Payment method depends on contract type (conditional probabilities).
- Support ticket intensity depends on tenure band.
- Churn probability uses a logistic formulation over tenure, charges, tickets, and contract effects.

Key tunable parameters:

- `monthly_charge_mean`
- `ticket_intensity`
- `monthly_contract_share`
- `price_sensitivity`

## 2) Retail Sales

Columns:

- `date`
- `category`
- `base_price`
- `discount_pct`
- `units_sold`
- `ad_spend`
- `is_weekend`
- `is_holiday`
- `revenue`

Notable modeling details:

- Category sampled from fixed priors.
- Discount sampled up to configurable `discount_cap`.
- Revenue combines price, discount, units, ad spend, weekend/holiday lift, and noise.
- Rows are sorted by date for trend analysis.

Key tunable parameters:

- `discount_cap`
- `avg_units`
- `ad_spend_scale`
- `weekend_lift`

## 3) Health Risk

Columns:

- `age`
- `bmi`
- `systolic_bp`
- `cholesterol`
- `smoker`
- `exercise_days`
- `risk_score`
- `risk_band`

Notable modeling details:

- Biometrics and lifestyle factors are sampled with method-dependent distributions.
- Risk score is a weighted combination of vitals/behaviors plus configurable noise.
- Risk band is derived from score thresholds (`low`, `medium`, `high`).

Key tunable parameters:

- `smoking_rate`
- `exercise_bias`
- `bmi_mean`
- `risk_noise`

---

## Quality checks and constraints

DeepSense ships with built-in constraints by dataset, including:

- Churn constraints such as `tenure_months` range and allowed contract/payment categories.
- Retail constraints such as `discount_pct` in `[0, 100]` and non-negative `revenue`.
- Health constraints such as realistic ranges for age/BMI/BP/cholesterol and valid `risk_band` values.

Quality summary in the app reports:

- total schema/constraint violations
- average numeric-feature outlier rate (IQR rule)

---

## Python API usage

You can use DeepSense directly as a Python library.

### Generate a dataset

```python
from synthetic_data import generate_dataset

# Customer churn (distribution-based)
df = generate_dataset(
    dataset_type="customer_churn",
    rows=2000,
    seed=42,
    method="distribution_based",
    monthly_charge_mean=80,
    ticket_intensity=2.2,
)
```

### Inspect available parameters

```python
from synthetic_data import get_dataset_parameters

params = get_dataset_parameters("retail_sales")
print(params)
```

### Estimate churn prevalence before generating

```python
from synthetic_data import estimate_expected_churn_rate

rate = estimate_expected_churn_rate(rows=4000, seed=42, method="rule_based")
print(f"Expected churn rate: {rate:.2%}")
```

### Run quality checks programmatically

```python
from synthetic_data import summarize_quality

quality = summarize_quality(df, dataset_type="customer_churn")
print(quality["violation_total"], quality["avg_outlier_rate"])
```

### Compare two generation methods

```python
from synthetic_data import compute_distribution_distance, generate_dataset

a = generate_dataset("retail_sales", rows=2000, seed=7, method="rule_based")
b = generate_dataset("retail_sales", rows=2000, seed=7, method="distribution_based")

num_dist, cat_dist = compute_distribution_distance(a, b)
print(num_dist.head())
print(cat_dist.head())
```

---

## Testing

Run unit/property-oriented checks:

```bash
pytest -q
```

Current tests verify:

- all dataset types generate expected row counts
- all declared methods are supported
- invalid dataset/method validation errors
- core constraints (e.g., churn binary, discount bounds, non-negative revenue)
- churn estimate returns a valid probability in `[0, 1]`

---

## Reproducibility notes

- Set a fixed `seed` to reproduce outputs.
- Export **Data + Config** JSON to preserve generation metadata.
- Use identical `dataset_type`, `method`, row count, and advanced parameters when comparing runs.

---

## Limitations

- Synthetic realism is intentionally lightweight (not a GAN/LLM-backed simulator).
- No privacy guarantees are implied (this is synthetic generation logic, not a formal privacy mechanism).
- Constraints are practical defaults and can be extended for stricter domains.

---

## Potential extensions

If you want to evolve this project further:

- Add domain-specific constraint packs and custom validation rules.
- Add drift benchmarking against a real reference dataset.
- Add richer dependence structures (copulas/graph-based generation).
- Add synthetic-data utility benchmarks for downstream ML tasks.
- Add CI test matrix and lint/type checks.

---

## License

Add a project license (e.g., MIT) if you plan to distribute or reuse this repository broadly.
