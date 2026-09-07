
import os
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
    page_title="Insurance Claim Risk Detection",
    page_icon="🛡️",
    layout="wide",
)

setup_css()
require_auth()

st.markdown(
    '<div class="main-title">🛡️ Insurance Claim Risk Detection</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">AI-powered insurance claim risk triage, explainability and investigator support</div>',
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# DATASET UPLOAD
# ------------------------------------------------------------
uploaded = st.file_uploader(
    "Upload an insurance claims CSV",
    type=["csv"],
)

if uploaded is not None:
    st.session_state["_claims_uploaded_file"] = uploaded.getvalue()
else:
    st.session_state["_claims_uploaded_file"] = None
    st.info(
        "Using the included synthetic dataset. Upload your own CSV to rerun the prototype."
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
    cleaned,
    featured,
    active_pipe,
    risk_filter,
)

# ------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------
st.header("📋 Overview & Remaining Content")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Claims", len(cleaned))
c2.metric("Observed Risk/Fraud Rate", f"{cleaned['Fraud or risk label'].mean()*100:.1f}%")
c3.metric("High-Risk Claims", int((scored_all["Predicted Risk"] == "High").sum()))
c4.metric("Active Model", active_model_name)

st.divider()

st.subheader("Project Objective")
st.write(
    """
    The system identifies potentially suspicious or high-risk insurance claims
    so investigators can prioritize them for manual review. The machine-learning
    model predicts a risk probability, SHAP explains the important contributing
    factors, and Gemini converts those model-derived signals into a short,
    investigator-friendly explanation.
    """
)

st.subheader("1. Data Understanding & Cleaning")
st.write(
    f"**Dataset size:** {len(raw)} rows and {len(raw.columns)} columns."
)
st.write(
    f"**Duplicates removed:** {info['duplicate_rows_removed']}."
)
st.write(
    "Invalid numeric values are converted to missing values and handled inside "
    "the modeling pipeline."
)
st.info(
    "Leakage assumption: Claim ID and free-text customer description are excluded "
    "from modeling, and the target label is never used as an input feature."
)

st.subheader("2. Feature Engineering")
feature_descriptions = pd.DataFrame(
    {
        "Engineered Feature": [
            "Days to submit",
            "Claim to repair ratio",
            "Invoice variance percentage",
            "Missing document count",
            "Historical claim frequency",
            "High-value claim indicator",
        ],
        "Why It Is Useful": [
            "Measures the delay between the incident and claim submission.",
            "Compares claim amount with the repair estimate.",
            "Measures the difference between final invoice and repair estimate.",
            "Captures missing supporting documentation.",
            "Represents historical claim frequency.",
            "Flags unusually large claims as a possible risk signal.",
        ],
    }
)
st.dataframe(feature_descriptions, use_container_width=True, hide_index=True)

st.subheader("3. Machine-Learning Models")
st.write(
    """
    The project compares a simple baseline, Logistic Regression and Random Forest.
    Logistic Regression provides an interpretable linear baseline, while Random
    Forest captures nonlinear relationships and feature interactions. Because
    fraud/risk data can be imbalanced, the system evaluates Precision, Recall,
    F1, ROC-AUC and PR-AUC instead of relying only on accuracy.
    """
)

st.subheader("4. Model Selection")
st.write(
    f"""
    **Selected algorithm:** {active_model_name}

    The Controls panel lets the reviewer switch algorithms. Selecting a specific
    model makes that model the active predictor used for risk scoring and claim
    explanations. Selecting "All Models" uses the model ranked first by the
    project's comparison results.
    """
)

st.subheader("5. SHAP Explainability")
st.write(
    """
    SHAP identifies the features that contributed to an individual prediction.
    Positive contributions push the model toward higher risk, while negative
    contributions push it toward lower risk. SHAP explains the model prediction;
    it does not prove fraud.
    """
)

st.subheader("6. Generative AI — Gemini")
st.write(
    """
    Gemini is used as a natural-language explanation layer. The ML model produces
    the risk score and SHAP provides contributing factors. Gemini receives the
    supplied claim facts and model-derived signals and generates a concise
    investigator-friendly summary. It does not make the fraud decision.
    """
)

st.subheader("7. End-to-End Workflow")
st.code(
    """
Insurance Claims
       ↓
Data Understanding & Cleaning
       ↓
Feature Engineering
       ↓
Preprocessing
       ↓
Baseline / Logistic Regression / Random Forest
       ↓
Risk Probability
       ↓
Low / Medium / High Risk
       ↓
SHAP Explanation
       ↓
Gemini Natural-Language Summary
       ↓
Human Investigator Review
    """,
    language="text",
)

st.subheader("8. Current Claim Investigation")

visible = scored.sort_values("Fraud Risk Probability", ascending=False)

if len(visible) == 0:
    st.warning("No claims match the selected risk filter.")
else:
    claim_id = st.selectbox(
        "Select a claim",
        visible["Claim ID"].tolist(),
        key="overview_claim",
    )

    idx = visible.index[visible["Claim ID"] == claim_id][0]
    row = featured.loc[[idx]]
    p = float(all_probs[idx])
    risk = str(visible.loc[idx, "Predicted Risk"])

    if risk == "High":
        risk_class = "risk-high"
    elif risk == "Medium":
        risk_class = "risk-medium"
    else:
        risk_class = "risk-low"

    st.markdown(
        f'<div class="{risk_class}">{risk.upper()} RISK</div>',
        unsafe_allow_html=True,
    )
    st.metric("Fraud / Risk Probability", f"{p:.1%}")

    st.write("### Main Contributing Factors")

    contributors = top_contributors(
        active_pipe,
        row.drop(columns=["Fraud or risk label"]),
    )

    if contributors:
        for name, value in contributors:
            direction = "increases" if value > 0 else "decreases"
            st.write(
                f"- **{name}**: {direction} model risk contribution ({value:+.3f})."
            )
    else:
        st.info("SHAP contributors are unavailable in this environment.")

    claim = scored_all.loc[idx]

    prompt_context = (
        f"Claim ID: {claim_id}; "
        f"risk probability: {p:.3f}; "
        f"risk category: {risk}; "
        f"claim amount: {claim['Claim amount']}; "
        f"repair estimate: {claim['Repair estimate']}; "
        f"final invoice: {claim['Final invoice amount']}; "
        f"days to submit: {claim['Days to submit']}; "
        f"previous claims: {claim['Previous claim count']}; "
        f"incident type: {claim['Incident type']}; "
        f"policy tenure: {claim['Policy tenure']}; "
        f"police report available: {claim['Police report available']}; "
        f"witness available: {claim['Witness available']}."
    )

    st.write("### 🤖 Investigator-Friendly GenAI Explanation")

    api_key = os.getenv("GEMINI_API_KEY")

    if api_key:
        try:
            from google import genai

            client = genai.Client(api_key=api_key)

            prompt = (
                "You are assisting an insurance investigator. "
                "Write a short, professional investigator-friendly explanation "
                "of why this claim received its model risk score. Use ONLY the "
                "supplied facts. Do not invent facts. Do not state that fraud is "
                "proven. Do not make the final claim decision. Describe the "
                "information as risk signals and recommend human review.\n\n"
                + prompt_context
            )

            response = client.models.generate_content(
                model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
                contents=prompt,
            )

            st.success(response.text)

        except Exception as e:
            st.warning(
                f"Gemini call unavailable ({e}). Showing the safe fallback explanation."
            )
            st.write(
                f"This claim is {risk.lower()} risk at {p:.1%}. "
                f"Key review signals include a claim amount of "
                f"{claim['Claim amount']:.0f}, invoice variance of "
                f"{claim['Invoice variance pct']:.1%}, "
                f"{claim['Days to submit']:.0f} days to submit, and "
                f"{claim['Previous claim count']} previous claims. "
                "These are risk indicators for investigation, not a final fraud decision."
            )
    else:
        st.info("GEMINI_API_KEY is not configured. Showing the safe fallback explanation.")
        st.write(
            f"This claim is {risk.lower()} risk at {p:.1%}. "
            f"Key review signals include a claim amount of "
            f"{claim['Claim amount']:.0f}, invoice variance of "
            f"{claim['Invoice variance pct']:.1%}, "
            f"{claim['Days to submit']:.0f} days to submit, and "
            f"{claim['Previous claim count']} previous claims. "
            "These are risk indicators for investigation, not a final fraud decision."
        )

st.divider()
st.subheader("Navigation")
st.write(
    "Use the **Controls → Pages** links on the left to open the dedicated "
    "Graphs or Tables pages."
)
