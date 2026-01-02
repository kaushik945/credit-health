from fastapi import FastAPI
import pandas as pd
import joblib
import numpy as np
from feature_engineering import CreditFeatureEngineer

# ------------------- LOAD ARTIFACTS -------------------
model = joblib.load("credit_model.pkl")
explainer = joblib.load("shap_explainer.pkl")
feature_names = joblib.load("feature_names.pkl")

THRESHOLD = 0.35

app = FastAPI(title="Credit Health Score API")

# ------------------- UTIL FUNCTIONS -------------------

def credit_health_score(prob_default: float) -> int:
    return int(round((1 - prob_default) * 100))


def credit_health_label(score: int) -> str:
    if score >= 80:
        return "Excellent"
    elif score >= 65:
        return "Good"
    elif score >= 50:
        return "Fair"
    else:
        return "Poor"


def explain_credit_score(shap_values: dict, top_k=3):
    positives = []
    negatives = []

    for feat, val in shap_values.items():
        if val > 0:
            positives.append((feat, val))
        else:
            negatives.append((feat, val))

    positives = sorted(positives, key=lambda x: x[1], reverse=True)[:top_k]
    negatives = sorted(negatives, key=lambda x: abs(x[1]), reverse=True)[:top_k]

    return {
        "risk_increasing_factors": positives,
        "risk_reducing_factors": negatives
    }

# ------------------- ROUTES -------------------

@app.get("/")
def root():
    return {"status": "Credit Health API running"}


@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    prob = model.predict_proba(df)[0, 1]
    decision = int(prob >= THRESHOLD)

    return {
        "default_probability": round(float(prob), 4),
        "decision": decision,
        "risk_label": "High Risk" if decision else "Low Risk"
    }


@app.post("/credit-health")
def credit_health(data: dict):
    try:
        df = pd.DataFrame([data])

        # ---------- PREDICTION ----------
        prob = model.predict_proba(df)[0, 1]
        score = credit_health_score(prob)
        label = credit_health_label(score)

        # ---------- SHAP ----------
        fe = model.named_steps["feature_engineering"]
        pre = model.named_steps["preprocessing"]

        X_fe = fe.transform(df)
        X_trans = pre.transform(X_fe)

        if hasattr(X_trans, "toarray"):
            X_trans = X_trans.toarray()

        shap_vals = explainer.shap_values(X_trans)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]

        shap_dict = {
            feature_names[i]: float(shap_vals[0][i])
            for i in range(len(feature_names))
        }

        explanation = explain_credit_score(shap_dict)

        return {
            "credit_health_score": score,
            "credit_health_label": label,
            "default_probability": round(float(prob), 4),
            "decision_threshold": THRESHOLD,
            "why_this_score": explanation
        }

    except Exception as e:
        return {"error": str(e)}
