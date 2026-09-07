
import streamlit as st
import pandas as pd

from common_app import (
    require_auth,
    setup_css,
    get_project_data,
    sidebar_controls,
    active_model_data,
    score_claims,
)

st.set_page_config(
    page_title="Insurance Claim Risk Detection - Tables",
    page_icon="📑",
    layout="wide",
)

setup_css()
require_auth()

st.markdown(
    '<div class="main-title">📑 All Tables & Model Results</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">Data quality, model comparison, risk segments and scored claims</div>',
    unsafe_allow_html=True,
)

try:
    raw, cleaned, featured, info, comparison, fitted, test = get_project_data()
except Exception as e:
    st.error(f"Could not process the dataset: {e}")
    st.stop()

selected_model, comparison_metric, risk_filter = sidebar_controls(comparison)
active_model_name, active_pipe, active_metrics = active_model_data(
    selected_model, comparison, fitted
)
scored_all, scored, all_probs = score_claims(
    cleaned, featured, active_pipe, risk_filter
)

# 1 Data quality
st.header("1. Data Quality")

quality = pd.DataFrame(
    {"Missing values": pd.Series(info["missing_by_column"])}
)
st.dataframe(quality, use_container_width=True)

st.write(
    f"Duplicates removed: **{info['duplicate_rows_removed']}**"
)

# 2 Model comparison
st.header("2. Model Comparison")

comparison_display = comparison.copy()

existing_metric = [
    c for c in [
        "CV PR-AUC",
        "Precision",
        "Recall",
        "F1",
        "ROC-AUC",
        "PR-AUC",
    ]
    if c in comparison_display.columns
]

if comparison_metric in comparison_display.columns:
    comparison_display = comparison_display.sort_values(
        comparison_metric, ascending=False
    )

st.dataframe(
    comparison_display.style.format(
        {c: "{:.3f}" for c in existing_metric}
    ),
    use_container_width=True,
)

st.success(
    f"Active algorithm: {active_model_name}"
)

# 3 Risk segment table
st.header("3. Top Risk-Rate Segments")

segment = (
    cleaned.assign(_risk=cleaned["Fraud or risk label"].values)
    .groupby("Incident type")["_risk"]
    .agg(["mean", "count"])
    .sort_values("mean", ascending=False)
)

segment["mean"] = (segment["mean"] * 100).round(1)

st.dataframe(
    segment.rename(
        columns={"mean": "Risk rate %", "count": "Claims"}
    ),
    use_container_width=True,
)

# 4 Scored claims
st.header("4. Scored Claims")

display_columns = [
    "Claim ID",
    "Fraud Risk Probability",
    "Predicted Risk",
    "Claim amount",
    "Repair estimate",
    "Final invoice amount",
    "Days to submit",
    "Previous claim count",
    "Invoice variance pct",
    "Missing document count",
]

available = [c for c in display_columns if c in scored.columns]

st.write(
    f"Showing **{len(scored)}** claims after the selected risk filter."
)

st.dataframe(
    scored.sort_values(
        "Fraud Risk Probability",
        ascending=False
    )[available],
    use_container_width=True,
    hide_index=True,
)

# 5 Download
st.header("5. Download")

scored_csv = scored_all.to_csv(index=False).encode("utf-8")

st.download_button(
    "⬇️ Download All Scored Claims CSV",
    data=scored_csv,
    file_name="scored_insurance_claims.csv",
    mime="text/csv",
)
