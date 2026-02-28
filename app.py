from __future__ import annotations

import json

import streamlit as st

from synthetic_data import (
    CHURN_LOGIT_FORMULA,
    DATASET_TYPES,
    build_numeric_correlation,
    build_numeric_distributions,
    build_scatter_matrix,
    build_target_breakdown,
    build_time_trend,
    compute_column_profiles,
    compute_distribution_distance,
    compute_summary,
    estimate_expected_churn_rate,
    generate_dataset,
    get_dataset_parameters,
    summarize_quality,
)

st.set_page_config(page_title="Synthetic Data Studio", layout="wide")
st.title("Interactive Synthetic Data Generation & EDA Studio")
st.caption("Build realistic synthetic datasets with adjustable controls, explainable logic, and quality checks.")


def render_param_control(name: str, cfg: dict):
    label = name.replace("_", " ").title()
    if cfg["type"] == "float":
        return st.slider(
            label,
            min_value=float(cfg["min"]),
            max_value=float(cfg["max"]),
            value=float(cfg["default"]),
            step=float(cfg.get("step", 0.1)),
        )
    return st.number_input(label, value=cfg["default"])


with st.sidebar:
    st.header("Generator Controls")
    dataset_key = st.selectbox("Dataset type", options=list(DATASET_TYPES), format_func=lambda key: DATASET_TYPES[key].label)
    ds_info = DATASET_TYPES[dataset_key]
    st.info(ds_info.description)

    method = st.selectbox("Generation method", options=ds_info.methods, index=ds_info.methods.index(ds_info.default_method))
    rows = st.slider("Rows", min_value=100, max_value=20000, value=1200, step=100)
    seed = st.number_input("Random seed", min_value=0, max_value=999_999, value=42, step=1)

    st.subheader("Advanced dataset parameters")
    dataset_params = {p: render_param_control(p, cfg) for p, cfg in get_dataset_parameters(dataset_key).items()}

    if dataset_key == "customer_churn":
        est_churn = estimate_expected_churn_rate(rows=4000, seed=int(seed), method=method, **dataset_params)
        st.metric("Expected churn rate", f"{est_churn:.1%}")

    regen = st.button("Generate dataset", type="primary", use_container_width=True)

state_key = f"generated_{dataset_key}_{method}"
if regen or state_key not in st.session_state:
    st.session_state[state_key] = generate_dataset(
        dataset_type=dataset_key,
        rows=rows,
        seed=int(seed),
        method=method,
        **dataset_params,
    )

df = st.session_state[state_key]
config_payload = {
    "dataset_type": dataset_key,
    "method": method,
    "rows": rows,
    "seed": int(seed),
    "parameters": dataset_params,
}

st.success(f"Generated {len(df):,} rows using `{dataset_key}` + `{method}` strategy.")

stats_df, missing_df = compute_summary(df)
numeric_profile, categorical_profile = compute_column_profiles(df)
quality = summarize_quality(df, dataset_key)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Rows", f"{len(df):,}")
k2.metric("Columns", len(df.columns))
k3.metric("Constraint violations", quality["violation_total"])
k4.metric("Avg outlier rate", f"{quality['avg_outlier_rate']:.2%}")

tab_data, tab_profile, tab_charts, tab_quality, tab_export = st.tabs([
    "Data Preview",
    "Profiling",
    "Visual EDA",
    "Quality",
    "Export",
])

with tab_data:
    st.dataframe(df.head(100), use_container_width=True)
    st.subheader("Generation config")
    st.code(json.dumps(config_payload, indent=2), language="json")

    if dataset_key == "customer_churn":
        st.subheader("Churn model formula")
        st.code(CHURN_LOGIT_FORMULA)

with tab_profile:
    left, right = st.columns(2)
    with left:
        st.subheader("Summary statistics")
        st.dataframe(stats_df, use_container_width=True)
        st.subheader("Numeric profile")
        st.dataframe(numeric_profile, use_container_width=True)
    with right:
        st.subheader("Missing values")
        st.dataframe(missing_df, use_container_width=True)
        st.subheader("Categorical profile")
        st.dataframe(categorical_profile, use_container_width=True)

with tab_charts:
    figs = [
        build_numeric_distributions(df),
        build_numeric_correlation(df),
        build_scatter_matrix(df),
        build_target_breakdown(df),
        build_time_trend(df),
    ]
    for fig in figs:
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)

with tab_quality:
    st.subheader("Schema & constraint validation")
    st.dataframe(quality["schema_checks"], use_container_width=True)

    st.subheader("Outlier rate by feature (IQR rule)")
    st.dataframe(quality["outliers"], use_container_width=True)

    st.subheader("Method distance comparison")
    alt_method = next((m for m in ds_info.methods if m != method), None)
    if alt_method is not None:
        baseline_df = generate_dataset(dataset_type=dataset_key, rows=rows, seed=int(seed), method=alt_method, **dataset_params)
        num_dist, cat_dist = compute_distribution_distance(df, baseline_df)

        if not num_dist.empty:
            st.caption(f"Current method `{method}` vs `{alt_method}` (numeric)")
            st.dataframe(num_dist, use_container_width=True)
            st.metric("Avg KS", f"{num_dist['ks_stat'].mean():.3f}")
        if not cat_dist.empty:
            st.caption(f"Current method `{method}` vs `{alt_method}` (categorical)")
            st.dataframe(cat_dist, use_container_width=True)
            st.metric("Avg JS divergence", f"{cat_dist['js_divergence'].mean():.3f}")

with tab_export:
    st.download_button(
        label="Download CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=f"{dataset_key}_{method}_synthetic_data.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.download_button(
        label="Download JSON",
        data=df.to_json(orient="records", indent=2).encode("utf-8"),
        file_name=f"{dataset_key}_{method}_synthetic_data.json",
        mime="application/json",
        use_container_width=True,
    )

    combined = {
        "config": config_payload,
        "rows": json.loads(df.to_json(orient="records", date_format="iso")),
    }
    st.download_button(
        label="Download Data + Config (JSON)",
        data=json.dumps(combined, indent=2).encode("utf-8"),
        file_name=f"{dataset_key}_{method}_with_config.json",
        mime="application/json",
        use_container_width=True,
    )
