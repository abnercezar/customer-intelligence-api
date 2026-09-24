import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from app.ml.features import FEATURES
from app.ml.segmenter import train_segmenter

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


def build_model() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=200, min_samples_leaf=5, random_state=42, n_jobs=-1)),
    ])


def save(model: Pipeline, df: pd.DataFrame):
    joblib.dump(model, MODEL_PATH)
    print(f"Modelo de churn salvo em {MODEL_PATH}")

    print("\nTreinando segmentador K-Means...")
    train_segmenter(df)
    print("Segmentador salvo.")


def train_synthetic():
    df = generate_synthetic_data()

    X = df[FEATURES]
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = build_model()
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"Acurácia (dados sintéticos — só confirma que o modelo copiou a fórmula): {accuracy:.2%}")

    save(model, df)


# Cortes trimestrais; o último + 90 dias ainda cabe no dataset (termina em 09/12/2011).
TRAIN_CUTOFFS = ["2010-06-01", "2010-09-01", "2010-12-01", "2011-03-01", "2011-06-01"]
TEST_CUTOFF = "2011-09-01"
HORIZON_DAYS = 90


def train_online_retail():
    from app.ml.online_retail import build_training_set, load_orders

    orders = load_orders()
    print(f"{len(orders)} pedidos de {orders['customer_id'].nunique()} clientes")

    print("Montando snapshots por data de corte...")
    train_df = build_training_set(orders, TRAIN_CUTOFFS, HORIZON_DAYS)
    test_df = build_training_set(orders, [TEST_CUTOFF], HORIZON_DAYS)

    # Avaliação temporal: o teste é um corte posterior que o modelo nunca viu.
    model = build_model()
    model.fit(train_df[FEATURES], train_df["churn"])

    y_test = test_df["churn"]
    proba = model.predict_proba(test_df[FEATURES])[:, 1]
    accuracy = accuracy_score(y_test, proba >= 0.5)
    baseline = max(y_test.mean(), 1 - y_test.mean())

    print(f"\nTeste no corte {TEST_CUTOFF} ({len(test_df)} clientes, churn real = {y_test.mean():.1%})")
    print(f"  Acurácia:            {accuracy:.2%}")
    print(f"  Baseline (chutar a classe mais comum): {baseline:.2%}")
    print(f"  ROC AUC:             {roc_auc_score(y_test, proba):.3f}  (0.5 = aleatório, 1.0 = perfeito)")

    # Modelo final usa todos os cortes, incluindo o de teste.
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    final_model = build_model()
    final_model.fit(full_df[FEATURES], full_df["churn"])

    # Só o modelo de churn: o K-Means com esses dados gera segmentos muito
    # desbalanceados (valores de atacado dominam os clusters). O segmentador
    # continua vindo de `--synthetic` até isso ser tratado.
    joblib.dump(final_model, MODEL_PATH)
    print(f"Modelo de churn salvo em {MODEL_PATH}")


if __name__ == "__main__":
    if "--synthetic" in sys.argv:
        train_synthetic()
    else:
        train_online_retail()
