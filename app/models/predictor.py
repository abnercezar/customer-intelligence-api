import os
import joblib
import pandas as pd
from app.ml.features import FEATURES, extract_features

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "ml", "model.joblib")

_model = None


def get_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


def _segment(recency: int, frequency: int, monetary_avg: float, churn_risk: float) -> str:
    if churn_risk < 0.25 and frequency >= 6 and monetary_avg >= 300:
        return "champion"
    if churn_risk < 0.4 and frequency >= 3:
        return "loyal"
    if churn_risk >= 0.65:
        return "at_risk"
    if frequency <= 1:
        return "new"
    return "potential"


def _trend(slope: float) -> str:
    if slope > 10:
        return "growing"
    if slope < -10:
        return "declining"
    return "stable"


def _value(total: float) -> str:
    if total >= 3000:
        return "high"
    if total >= 800:
        return "medium"
    return "low"


def _action(segment: str) -> str:
    return {
        "champion": "maintain_engagement",
        "loyal": "upsell",
        "at_risk": "reactivation",
        "new": "onboarding",
        "potential": "nurture",
    }.get(segment, "monitor")


def _reasons(features: dict, trend: str) -> list:
    reasons = []
    if features["recency"] > 60:
        reasons.append(f"tempo desde última compra acima do padrão ({features['recency']} dias)")
    if features["frequency"] < 3:
        reasons.append("baixa frequência de compras")
    if trend == "declining":
        reasons.append("valor das compras em queda")
    if features["monetary_avg"] < 100:
        reasons.append("ticket médio abaixo da média")
    if features["avg_days_between"] > 45:
        reasons.append("intervalo médio entre compras elevado")
    if not reasons:
        reasons.append("comportamento dentro do padrão esperado")
    return reasons


def predict(orders: list, customer_id: str) -> dict:
    features = extract_features(orders)

    X = pd.DataFrame([features], columns=FEATURES)

    churn_risk = round(float(get_model().predict_proba(X)[0][1]), 2)
    segment = _segment(features["recency"], features["frequency"], features["monetary_avg"], churn_risk)
    trend = _trend(features["trend_slope"])
    value = _value(features["monetary_total"])
    action = _action(segment)
    reasons = _reasons(features, trend)

    return {
        "customer_id": customer_id,
        "churn_risk": churn_risk,
        "segment": segment,
        "purchase_trend": trend,
        "customer_value": value,
        "recommended_action": action,
        "reasons": reasons,
    }
