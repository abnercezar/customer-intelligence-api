"""
Treina o motor numa base de pedidos.

  python -m app.ml.train --orders pedidos.csv
  python -m app.ml.train --benchmark          # Online Retail II, só comparação
  python -m app.ml.train --synthetic          # fórmula artificial, sem corte temporal

O modelo final usa só os cortes de treino. O último corte mede e não entra no fit.
Os dois modelos e os limiares só substituem o artefato se a medição passar.
"""
import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.ml.features import FEATURES
from app.ml.gates import acceptance_failures
from app.ml.orders import HORIZON_DAYS, build_training_set, choose_cutoffs, derive_thresholds, load_orders_csv
from app.ml.paths import ARTIFACT_DIR, MODEL_PATH, THRESHOLDS_PATH
from app.ml.segmenter import N_CLUSTERS, fit_segmenter, save_segmenter, train_segmenter


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
    thresholds = derive_thresholds(df)
    thresholds["source"] = "synthetic"
    joblib.dump(thresholds, THRESHOLDS_PATH)
    print(f"Limiares salvos em {THRESHOLDS_PATH}")
    print("Dados sintéticos não passam por corte temporal. Não use estas métricas como evidência.")


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


def _blank_report(source: str) -> dict:
    return {
        "source": source,
        "saved": False,
        "failures": [],
        "train_cutoffs": [],
        "test_cutoff": None,
        "train_rows": 0,
        "train_customers": 0,
        "test_customers": 0,
        "test_churn_rate": None,
        "accuracy": None,
        "baseline": None,
        "auc": None,
        "segment_sizes": None,
        "segment_note": None,
        "thresholds": None,
        "final_fit_rows": None,
    }


def _iso(value) -> str:
    return pd.Timestamp(value).date().isoformat()


def _churn_probability(model: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    classes = list(model.named_steps["clf"].classes_)
    if 1 not in classes:
        return np.zeros(len(frame))
    column = classes.index(1)
    return model.predict_proba(frame[FEATURES])[:, column]


def train_from_orders(orders: pd.DataFrame, source: str, artifact_dir: str = None) -> dict:
    """Treina churn e K-Means na mesma base. Só grava se a medição do corte de teste passar."""
    report = _blank_report(source)
    artifact_dir = artifact_dir or ARTIFACT_DIR
    orders = orders.copy()
    orders["date"] = pd.to_datetime(orders["date"])

    train_cutoffs, test_cutoff = choose_cutoffs(orders, HORIZON_DAYS)
    if test_cutoff is None:
        report["failures"] = [
            "histórico curto demais para um corte de treino e um de teste com "
            f"{HORIZON_DAYS} dias de futuro observável"
        ]
        return report

    report["train_cutoffs"] = [_iso(cutoff) for cutoff in train_cutoffs]
    report["test_cutoff"] = _iso(test_cutoff)

    train_df = build_training_set(orders, train_cutoffs, HORIZON_DAYS)
    test_df = build_training_set(orders, [test_cutoff], HORIZON_DAYS)
    report["train_rows"] = int(len(train_df))
    report["train_customers"] = int(train_df["customer_id"].nunique()) if len(train_df) else 0
    report["test_customers"] = int(test_df["customer_id"].nunique()) if len(test_df) else 0

    if train_df.empty or test_df.empty:
        report["failures"] = ["algum corte ficou sem clientes com histórico anterior"]
        return report

    if train_df["churn"].nunique() < 2:
        report["failures"] = ["o treino ficou com uma classe só — não há o que separar"]
        report["test_churn_rate"] = float(test_df["churn"].mean())
        return report

    model = build_model()
    model.fit(train_df[FEATURES], train_df["churn"])
    report["final_fit_rows"] = int(len(train_df))

    proba = _churn_probability(model, test_df)
    y_test = test_df["churn"].astype(int)
    report["accuracy"] = float(accuracy_score(y_test, proba >= 0.5))
    report["baseline"] = float(max(y_test.mean(), 1 - y_test.mean()))
    report["test_churn_rate"] = float(y_test.mean())
    if y_test.nunique() == 2:
        report["auc"] = float(roc_auc_score(y_test, proba))

    latest_cutoff = pd.to_datetime(train_df["cutoff"]).max()
    latest = train_df.loc[pd.to_datetime(train_df["cutoff"]) == latest_cutoff].copy()
    report["thresholds"] = derive_thresholds(latest)

    pipeline = None
    segment_map = None
    if len(latest) < N_CLUSTERS:
        report["segment_note"] = (
            f"só {len(latest)} clientes no último corte de treino; "
            f"o K-Means precisa de pelo menos {N_CLUSTERS}"
        )
    else:
        pipeline, segment_map, report["segment_sizes"] = fit_segmenter(latest)

    report["failures"] = acceptance_failures(report, report["segment_sizes"])
    report["saved"] = not report["failures"]
    if report["saved"]:
        os.makedirs(artifact_dir, exist_ok=True)
        joblib.dump(model, os.path.join(artifact_dir, "model.joblib"))
        save_segmenter(pipeline, segment_map, artifact_dir)
        payload = dict(report["thresholds"])
        payload["source"] = source
        payload["test_cutoff"] = report["test_cutoff"]
        payload["test_auc"] = report["auc"]
        payload["test_customers"] = report["test_customers"]
        joblib.dump(payload, os.path.join(artifact_dir, "thresholds.joblib"))

    return report


def _pct(value) -> str:
    if value is None:
        return "—"
    return f"{value:.1%}"


def _plain(value, digits: int) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


def print_report(report: dict) -> None:
    print(f"Base: {report['source']}")
    print("A contagem de clientes não aprova o modelo. Ele só é gravado se o corte de teste")
    print("ganhar do baseline e cada segmento tiver tamanho mínimo.")
    train_cutoffs = ", ".join(report["train_cutoffs"]) or "—"
    print(f"Cortes de treino: {train_cutoffs}")
    print(f"Corte de teste (não entra no modelo): {report['test_cutoff'] or '—'}")
    print(f"Clientes no treino: {report['train_customers']} ({report['train_rows']} linhas)")
    print(f"Clientes no teste: {report['test_customers']}")
    print(f"Churn no teste: {_pct(report['test_churn_rate'])}")
    print(f"Acurácia: {_pct(report['accuracy'])}")
    print(f"Baseline (chute da classe mais comum): {_pct(report['baseline'])}")
    print(f"ROC AUC: {_plain(report['auc'], 3)}  (0.5 = aleatório, 1.0 = perfeito)")
    print("O score não é probabilidade calibrada.")
    if report["segment_sizes"]:
        print(f"Segmentos no último corte de treino: {report['segment_sizes']}")
    if report["thresholds"]:
        limits = report["thresholds"]
        print(
            "Limiares desta base: "
            f"alto a partir de {limits['value_high']:.2f}, "
            f"médio a partir de {limits['value_medium']:.2f}, "
            f"ticket baixo abaixo de {limits['ticket_low']:.2f}, "
            f"tendência ±{limits['trend_delta']:.2f}"
        )
    if report["saved"]:
        print("Artefato gravado. Reinicie a API para carregar.")
    else:
        print("Artefato não foi gravado.")
        for failure in report["failures"]:
            print(f"  - {failure}")


def main(argv: list = None) -> int:
    parser = argparse.ArgumentParser(description="Treina o motor numa base de pedidos.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--orders", metavar="CSV", help="CSV com customer_id, date e value")
    group.add_argument("--benchmark", action="store_true", help="Online Retail II, só como benchmark")
    group.add_argument("--synthetic", action="store_true", help="Dados sintéticos, sem validação temporal")
    args = parser.parse_args(argv)

    if args.synthetic:
        train_synthetic()
        return 0

    if args.benchmark:
        from app.ml.online_retail import load_orders
        orders = load_orders()
        source = "online_retail_ii"
    else:
        orders = load_orders_csv(args.orders)
        source = args.orders

    print(f"{len(orders)} pedidos de {orders['customer_id'].nunique()} clientes")
    report = train_from_orders(orders, source)
    print_report(report)
    return 0 if report["saved"] else 1


if __name__ == "__main__":
    sys.exit(main())
