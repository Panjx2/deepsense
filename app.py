from __future__ import annotations

import json

import streamlit as st

from synthetic_data import (
    DATASET_TYPES,
    build_numeric_correlation,
    build_numeric_distributions,
    build_scatter_matrix,
    build_target_breakdown,
    build_time_trend,
    compute_column_profiles,
    compute_summary,
    generate_dataset,
    get_dataset_parameters,
)

st.set_page_config(page_title="Synthetic Data Studio", layout="wide")
st.title("Interactive Synthetic Data Generation & EDA Studio")
st.caption("Build realistic synthetic datasets with adjustable generation controls and instant visual profiling.")


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
    return st.number_input(label, value=cfg["default"])  # fallback


with st.sidebar:
    st.header("Generator Controls")
    dataset_key = st.selectbox("Dataset type", options=list(DATASET_TYPES), format_func=lambda key: DATASET_TYPES[key].label)
    ds_info = DATASET_TYPES[dataset_key]
    st.info(ds_info.description)

    method = st.selectbox("Generation method", options=ds_info.methods, index=ds_info.methods.index(ds_info.default_method))
    rows = st.slider("Rows", min_value=100, max_value=20000, value=1200, step=100)
    seed = st.number_input("Random seed", min_value=0, max_value=999_999, value=42, step=1)

    st.subheader("Advanced dataset parameters")
    dataset_params = {}
    for param_name, param_cfg in get_dataset_parameters(dataset_key).items():
        dataset_params[param_name] = render_param_control(param_name, param_cfg)

    regen = st.button("Generate dataset", type="primary", use_container_width=True)

state_key = f"generated_df_{dataset_key}"
if regen or state_key not in st.session_state:
    st.session_state[state_key] = generate_dataset(
        dataset_type=dataset_key,
        rows=rows,
        seed=int(seed),
        method=method,
        **dataset_params,
    )

df = st.session_state[state_key]
st.success(f"Generated {len(df):,} rows using `{dataset_key}` + `{method}` strategy.")

stats_df, missing_df = compute_summary(df)
numeric_profile, categorical_profile = compute_column_profiles(df)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Rows", f"{len(df):,}")
k2.metric("Columns", len(df.columns))
k3.metric("Missing", int(missing_df["missing_values"].sum()))
k4.metric("Memory (KB)", f"{df.memory_usage(deep=True).sum() / 1024:.1f}")

tab_data, tab_profile, tab_charts, tab_export = st.tabs(["Data Preview", "Profiling", "Visual EDA", "Export"])

with tab_data:
    st.dataframe(df.head(100), use_container_width=True)
    st.code(json.dumps(dataset_params, indent=2), language="json")

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
    rendered = False
    for fig in figs:
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
            rendered = True
    if not rendered:
        st.warning("No compatible columns found for plotting.")

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
