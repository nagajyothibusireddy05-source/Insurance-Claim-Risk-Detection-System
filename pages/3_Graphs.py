
import streamlit as st
import matplotlib.pyplot as plt

from common_app import (
    require_auth,
    setup_css,
    get_project_data,
    sidebar_controls,
    active_model_data,
    score_claims,
)

st.set_page_config(
    page_title="Insurance Claim Risk Detection - Graphs",
    page_icon="📊",
    layout="wide",
)

setup_css()
require_auth()

st.markdown(
    '<div class="main-title">📊 All Graphs & Exploratory Data Analysis</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">Dedicated visualization page for insurance claim patterns</div>',
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

st.info(
    f"Active algorithm: **{active_model_name}**. "
    "Use the sidebar links to move between pages."
)

# 1
col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots()
    cleaned["Fraud or risk label"].value_counts().sort_index().plot(
        kind="bar", ax=ax
    )
    ax.set_title("Fraud / Risk Label Distribution")
    ax.set_xlabel("Label (0 = normal, 1 = risk)")
    ax.set_ylabel("Claims")
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Shows the class distribution and highlights potential class imbalance.")

with col2:
    plot = (
        cleaned.assign(_days=featured["Days to submit"])
        .groupby("Fraud or risk label")["_days"]
        .mean()
    )
    fig, ax = plt.subplots()
    plot.plot(kind="bar", ax=ax)
    ax.set_title("Average Submission Delay by Label")
    ax.set_xlabel("Label")
    ax.set_ylabel("Average days")
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Longer reporting delays can be a useful risk signal but are not proof of fraud.")

# 2
col1, col2 = st.columns(2)

with col1:
    grp = (
        cleaned.assign(_prev=featured["Previous claim count"])
        .groupby("Fraud or risk label")["_prev"]
        .mean()
    )
    fig, ax = plt.subplots()
    grp.plot(kind="bar", ax=ax)
    ax.set_title("Average Previous Claim Count")
    ax.set_xlabel("Label")
    ax.set_ylabel("Previous claims")
    st.pyplot(fig)
    plt.close(fig)

with col2:
    tmp = featured.copy()
    tmp["label"] = cleaned["Fraud or risk label"].values

    fig, ax = plt.subplots()
    ax.scatter(
        tmp.loc[tmp["label"] == 0, "Claim amount"],
        tmp.loc[tmp["label"] == 0, "Invoice variance pct"],
        alpha=0.35,
        label="Normal",
    )
    ax.scatter(
        tmp.loc[tmp["label"] == 1, "Claim amount"],
        tmp.loc[tmp["label"] == 1, "Invoice variance pct"],
        alpha=0.55,
        label="Risk",
    )
    ax.set_xlabel("Claim amount")
    ax.set_ylabel("Invoice variance %")
    ax.set_title("Claim Size vs Invoice Variance")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

# 3
col1, col2 = st.columns(2)

with col1:
    tenure_bins = pd.cut(
        cleaned["Policy tenure"],
        bins=[0, 1, 3, 5, 10, 100],
        right=False,
    )

    tenure_rate = (
        cleaned.assign(_bin=tenure_bins)
        .groupby("_bin", observed=False)["Fraud or risk label"]
        .mean()
        * 100
    )

    fig, ax = plt.subplots()
    tenure_rate.plot(kind="bar", ax=ax)
    ax.set_title("Risk Rate by Policy Tenure")
    ax.set_xlabel("Policy tenure (years)")
    ax.set_ylabel("Risk rate %")
    st.pyplot(fig)
    plt.close(fig)

with col2:
    invoice_stats = (
        featured.assign(label=cleaned["Fraud or risk label"].values)
        .groupby("label")["Invoice variance pct"]
        .mean()
    )

    fig, ax = plt.subplots()
    invoice_stats.plot(kind="bar", ax=ax)
    ax.set_title("Average Invoice Variance by Label")
    ax.set_xlabel("Label")
    ax.set_ylabel("Invoice variance %")
    st.pyplot(fig)
    plt.close(fig)

# 4
st.subheader("Risk Probability Distribution")

fig, ax = plt.subplots()
ax.hist(all_probs, bins=20)
ax.set_title(f"Predicted Risk Probability — {active_model_name}")
ax.set_xlabel("Risk probability")
ax.set_ylabel("Claims")
st.pyplot(fig)
plt.close(fig)

st.caption(
    "This distribution shows how the active model scores claims across the dataset."
)
