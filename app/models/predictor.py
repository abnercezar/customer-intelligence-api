import os
import joblib
import pandas as pd
from app.ml.features import FEATURES, extract_features
from app.ml.paths import MODEL_PATH, THRESHOLDS_PATH
from app.ml.segmenter import predict_segment

# Usados só quando ainda não existe thresholds.joblib desta base.
DEFAULT_THRESHOLDS = {
    "value_high": 3000.0,
    "value_medium": 800.0,
    "ticket_low": 100.0,
    "trend_delta": 10.0,
}

_model = None
_thresholds = None


def get_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


def get_thresholds() -> dict:
    global _thresholds
    if _thresholds is None:
        loaded = dict(DEFAULT_THRESHOLDS)
        if os.path.exists(THRESHOLDS_PATH):
            stored = joblib.load(THRESHOLDS_PATH)
            for key in DEFAULT_THRESHOLDS:
                if key in stored:
                    loaded[key] = float(stored[key])
        _thresholds = loaded
    return _thresholds


def _trend(slope: float, trend_delta: float) -> str:
    if slope > trend_delta:
        return "growing"
    if slope < -trend_delta:
        return "declining"
    return "stable"


def _value(total: float, value_high: float, value_medium: float) -> str:
    if total >= value_high:
        return "high"
    if total >= value_medium:
        return "medium"
    return "low"


RISK_MEDIUM_FROM = 0.35
RISK_HIGH_FROM = 0.65


def _risk_level(churn_risk: float) -> str:
    if churn_risk >= RISK_HIGH_FROM:
        return "high"
    if churn_risk >= RISK_MEDIUM_FROM:
        return "medium"
    return "low"


def _action(segment: str, risk_level: str, frequency: int) -> str:
    # O risco vem do modelo com validação temporal; o segmento, do K-Means.
    # Quando discordam, vale o risco — evita "acolher cliente novo" para quem está sumindo.
    if risk_level == "high" and frequency >= 2:
        return "retention"
    if risk_level == "low" and segment == "at_risk":
        return "monitor"
    return {
        "champion": "maintain_engagement",
        "loyal": "maintain_relationship",
        "at_risk": "retention",
        "new": "onboarding",
        "potential": "nurture",
    }.get(segment, "monitor")


def _reasons(features: dict, trend: str, ticket_low: float) -> list:
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

    if features["monetary_avg"] < ticket_low:
        reasons.append(f"ticket médio baixo (R${features['monetary_avg']:.0f})")

    if not reasons:
        reasons.append("comportamento dentro do padrão esperado")

    return reasons


def predict(orders: list, customer_id: str) -> dict:
    features = extract_features(orders)

    X = pd.DataFrame([features], columns=FEATURES)

    limits = get_thresholds()
    churn_risk = round(float(get_model().predict_proba(X)[0][1]), 2)
    risk_level = _risk_level(churn_risk)
    segment = predict_segment(features)
    trend = _trend(features["trend_slope"], limits["trend_delta"])
    value = _value(features["monetary_total"], limits["value_high"], limits["value_medium"])
    action = _action(segment, risk_level, features["frequency"])
    reasons = _reasons(features, trend, limits["ticket_low"])

    return {
        "customer_id": customer_id,
        "churn_risk": churn_risk,
        "risk_level": risk_level,
        "segment": segment,
        "purchase_trend": trend,
        "customer_value": value,
        "recommended_action": action,
        "reasons": reasons,
        "confidence": features["confidence"],
        "value_high_from": limits["value_high"],
        "value_medium_from": limits["value_medium"],
    }
