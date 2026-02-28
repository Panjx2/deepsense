from __future__ import annotations

import streamlit as st

from synthetic_data import (
    DATASET_TYPES,
    build_numeric_correlation,
    build_numeric_distributions,
    build_target_breakdown,
    compute_summary,
    generate_dataset,
)

st.set_page_config(page_title="Synthetic Data + EDA Dashboard", layout="wide")
st.title("Interactive Synthetic Data Generation & EDA Dashboard")
st.caption("Generate realistic synthetic datasets and inspect them with instant EDA visuals.")

with st.sidebar:
    st.header("Generation Controls")
    dataset_key = st.selectbox(
        "Dataset type",
        options=list(DATASET_TYPES.keys()),
        format_func=lambda key: DATASET_TYPES[key].label,
    )
    st.info(DATASET_TYPES[dataset_key].description)
    rows = st.slider("Rows", min_value=100, max_value=10000, value=1000, step=100)
    seed = st.number_input("Random seed", min_value=0, max_value=999_999, value=42, step=1)

if st.button("Generate synthetic dataset", type="primary"):
    st.session_state["df"] = generate_dataset(dataset_key, rows=rows, seed=int(seed))

if "df" not in st.session_state:
    st.session_state["df"] = generate_dataset(dataset_key, rows=rows, seed=int(seed))

df = st.session_state["df"]

stats_df, missing_df = compute_summary(df)

kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("Rows", f"{len(df):,}")
kpi2.metric("Columns", len(df.columns))
kpi3.metric("Missing values", int(missing_df["missing_values"].sum()))

st.subheader("Dataset preview")
st.dataframe(df.head(30), use_container_width=True)

col_a, col_b = st.columns([2, 1])
with col_a:
    st.subheader("Summary statistics")
    st.dataframe(stats_df, use_container_width=True)

with col_b:
    st.subheader("Missing values")
    st.dataframe(missing_df, use_container_width=True)

st.subheader("Visual EDA")
hist_fig = build_numeric_distributions(df)
corr_fig = build_numeric_correlation(df)
target_fig = build_target_breakdown(df)

if hist_fig is not None:
    st.plotly_chart(hist_fig, use_container_width=True)
if corr_fig is not None:
    st.plotly_chart(corr_fig, use_container_width=True)
if target_fig is not None:
    st.plotly_chart(target_fig, use_container_width=True)

st.download_button(
    label="Download CSV",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name=f"{dataset_key}_synthetic_data.csv",
    mime="text/csv",
)
