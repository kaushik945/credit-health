import streamlit as st
import pandas as pd
import joblib
import math
import shap

from feature_engineering import CreditFeatureEngineer

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Credit Health Score",
    page_icon="💳",
    layout="centered"
)

# ---------------- LOAD ARTIFACTS ----------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("credit_model.pkl")
    explainer = joblib.load("shap_explainer.pkl")
    feature_names = joblib.load("feature_names.pkl")
    return model, explainer, feature_names

model, explainer, feature_names = load_artifacts()

# ---------------- SCORE FUNCTIONS ----------------
def credit_health_score(prob):
    prob = max(min(prob, 0.99), 0.01)
    score = 850 - 200 * math.log(prob / (1 - prob))
    score = max(min(score, 900), 300)
    return int((score - 300) / 6)

def credit_health_label(score):
    if score >= 80:
        return "Excellent"
    elif score >= 65:
        return "Good"
    elif score >= 50:
        return "Fair"
    else:
        return "Poor"

# ---------------- UI ----------------
st.title("💳 Credit Health Score")
st.write("Understand your credit health using behavior and capacity signals.")

st.divider()

with st.form("credit_form"):
    col1, col2 = st.columns(2)

    with col1:
        limit_bal = st.number_input("Credit Limit (₹)", 10000, 1000000, 200000)
        age = st.number_input("Age", 18, 100, 30)

    with col2:
        bill1 = st.number_input("Bill Amount (Last Month)", 0, 1000000, 50000)
        bill2 = st.number_input("Bill Amount (2 Months Ago)", 0, 1000000, 48000)
        bill3 = st.number_input("Bill Amount (3 Months Ago)", 0, 1000000, 46000)

    st.subheader("Payment History")
    pay_0 = st.selectbox("Last Month", [0, 1, 2, 3])
    pay_2 = st.selectbox("2 Months Ago", [0, 1, 2, 3])
    pay_3 = st.selectbox("3 Months Ago", [0, 1, 2, 3])

    submitted = st.form_submit_button("Check Credit Health")

# ---------------- PREDICTION ----------------
if submitted:
    input_df = pd.DataFrame([{
        "LIMIT_BAL": limit_bal,
        "AGE": age,
        "BILL_AMT1": bill1,
        "BILL_AMT2": bill2,
        "BILL_AMT3": bill3,
        "PAY_0": pay_0,
        "PAY_2": pay_2,
        "PAY_3": pay_3
    }])

    prob = model.predict_proba(input_df)[0, 1]
    score = credit_health_score(prob)
    label = credit_health_label(score)

    st.divider()
    st.metric("Credit Health Score", f"{score} / 100", label)
    st.progress(score / 100)

    # ---------------- SHAP ----------------
    fe = model.named_steps["feature_engineering"]
    pre = model.named_steps["preprocessing"]

    X_fe = fe.transform(input_df)
    X_tr = pre.transform(X_fe)
    if hasattr(X_tr, "toarray"):
        X_tr = X_tr.toarray()

    shap_vals = explainer.shap_values(X_tr)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]

    shap_dict = {
        feature_names[i]: shap_vals[0][i]
        for i in range(len(feature_names))
    }

    st.subheader("🧠 Key Factors")
    for k, v in sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:5]:
        st.write(f"- **{k.replace('_',' ').title()}**")

