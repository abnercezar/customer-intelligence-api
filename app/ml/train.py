import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from app.ml.features import FEATURES

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")


def generate_synthetic_data(n: int = 2000) -> pd.DataFrame:
    """
    Gera dados RFM sintéticos para treino inicial.
    Substitua por dados reais quando disponíveis.
    """
    np.random.seed(42)
    rows = []

    for _ in range(n):
        frequency = np.random.randint(1, 25)
        recency = np.random.randint(1, 400)
        monetary_avg = np.random.uniform(50, 1500)
        monetary_total = monetary_avg * frequency
        trend_slope = np.random.uniform(-80, 80)
        avg_days_between = np.random.uniform(5, 120)

        # Regra de negócio: alto recency + baixa frequência = risco de churn
        score = (recency / 400) * 0.45 + (1 - min(frequency, 12) / 12) * 0.35
        if trend_slope < 0:
            score += 0.15
        score += np.random.uniform(-0.1, 0.1)
        churn = 1 if score > 0.5 else 0

        rows.append({
            "recency": recency,
            "frequency": frequency,
            "monetary_avg": monetary_avg,
            "monetary_total": monetary_total,
            "trend_slope": trend_slope,
            "avg_days_between": avg_days_between,
            "churn": churn,
        })

    return pd.DataFrame(rows)


def train():
    df = generate_synthetic_data()

    X = df[FEATURES]
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42)),
    ])

    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"Acurácia: {accuracy:.2%}")

    joblib.dump(model, MODEL_PATH)
    print(f"Modelo salvo em {MODEL_PATH}")


if __name__ == "__main__":
    train()
