import os
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEGMENTER_PATH = os.path.join(os.path.dirname(__file__), "segmenter.joblib")
SEGMENT_MAP_PATH = os.path.join(os.path.dirname(__file__), "segment_map.joblib")

# Apenas features RFM — não usamos trend/intervalo para clustering
CLUSTER_FEATURES = ["recency", "frequency", "monetary_avg", "monetary_total"]
N_CLUSTERS = 5

_segmenter = None
_segment_map = None


def _label_clusters(pipeline: Pipeline) -> dict:
    """
    Rotula cada cluster analisando os centróides em escala original.

    Lógica:
      - Menor frequência média → "new"
      - Maior recency médio   → "at_risk"
      - Dos restantes, ordena por monetary_avg desc → champion > loyal > potential
    """
    scaler: StandardScaler = pipeline.named_steps["scaler"]
    kmeans: KMeans = pipeline.named_steps["kmeans"]

    # Centróides de volta à escala original
    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    # índices: 0=recency, 1=frequency, 2=monetary_avg, 3=monetary_total
    ids = list(range(N_CLUSTERS))

    # 1. "new" → menor frequência média
    new_id = min(ids, key=lambda i: centers[i][1])
    remaining = [i for i in ids if i != new_id]

    # 2. "at_risk" → maior recency entre os restantes
    at_risk_id = max(remaining, key=lambda i: centers[i][0])
    remaining = [i for i in remaining if i != at_risk_id]

    # 3. Ordena os 3 restantes por monetary_avg desc
    remaining.sort(key=lambda i: centers[i][2], reverse=True)
    labels = ["champion", "loyal", "potential"]

    segment_map = {new_id: "new", at_risk_id: "at_risk"}
    for cluster_id, label in zip(remaining, labels):
        segment_map[cluster_id] = label

    return segment_map


def train_segmenter(df: pd.DataFrame) -> dict:
    X = df[CLUSTER_FEATURES]

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)),
    ])
    pipeline.fit(X)

    segment_map = _label_clusters(pipeline)

    joblib.dump(pipeline, SEGMENTER_PATH)
    joblib.dump(segment_map, SEGMENT_MAP_PATH)

    # Log para conferir a distribuição durante o treino
    labels = [segment_map[c] for c in pipeline.named_steps["kmeans"].labels_]
    dist = pd.Series(labels).value_counts().to_dict()
    print(f"Distribuição dos segmentos: {dist}")

    return segment_map


def _load():
    global _segmenter, _segment_map
    if _segmenter is None:
        _segmenter = joblib.load(SEGMENTER_PATH)
        _segment_map = joblib.load(SEGMENT_MAP_PATH)


def predict_segment(features: dict) -> str:
    _load()
    X = pd.DataFrame(
        [[features["recency"], features["frequency"], features["monetary_avg"], features["monetary_total"]]],
        columns=CLUSTER_FEATURES,
    )
    cluster = int(_segmenter.predict(X)[0])
    return _segment_map.get(cluster, "potential")
