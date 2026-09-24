import os
import joblib
import pandas as pd
from app.ml.features import FEATURES, extract_features
from app.ml.segmenter import predict_segment

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "ml", "model.joblib")

_model = None


def get_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model



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
        "loyal": "maintain_relationship",
        "at_risk": "retention",
        "new": "onboarding",
        "potential": "nurture",
    }.get(segment, "monitor")


def _reasons(features: dict, trend: str) -> list:
    reasons = []
    expected = features.get("expected_interval")
    recency = features["recency"]

    if expected is not None:
        # Compara com o próprio histórico do cliente.
        deviation = recency - expected
        if deviation > expected * 0.5:
            ratio = recency / expected
            reasons.append(
                f"{recency} dias desde última atividade "
                f"(habitual: {int(expected)} dias — {ratio:.1f}x acima do esperado)"
            )
    else:
        # Sem baseline individual: usa threshold global como fallback.
        if recency > 60:
            reasons.append(
                f"{recency} dias desde última atividade "
                f"(histórico insuficiente para calcular intervalo habitual)"
            )

    if features["frequency"] < 3:
        reasons.append("poucos eventos no histórico — padrão ainda não estabelecido")

    if trend == "declining":
        reasons.append("valor das transações em queda ao longo do tempo")

    if features["monetary_avg"] < 100:
        reasons.append(f"ticket médio baixo (R${features['monetary_avg']:.0f})")

    if not reasons:
        reasons.append("comportamento dentro do padrão esperado")

    return reasons


def predict(orders: list, customer_id: str) -> dict:
    features = extract_features(orders)

    X = pd.DataFrame([features], columns=FEATURES)

    churn_risk = round(float(get_model().predict_proba(X)[0][1]), 2)
    segment = predict_segment(features)
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
        "confidence": features["confidence"],
    }
