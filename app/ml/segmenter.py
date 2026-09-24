import os
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.ml.paths import ARTIFACT_DIR, SEGMENT_MAP_PATH, SEGMENTER_PATH

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


def fit_segmenter(df: pd.DataFrame):
    """Agrupa um retrato por cliente. Não grava arquivo."""
    X = df[CLUSTER_FEATURES]

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)),
    ])
    pipeline.fit(X)

    segment_map = _label_clusters(pipeline)
    labels = [segment_map[c] for c in pipeline.named_steps["kmeans"].labels_]
    sizes = {str(name): int(count) for name, count in pd.Series(labels).value_counts().items()}
    return pipeline, segment_map, sizes


def save_segmenter(pipeline: Pipeline, segment_map: dict, directory: str = None) -> None:
    directory = directory or ARTIFACT_DIR
    joblib.dump(pipeline, os.path.join(directory, "segmenter.joblib"))
    joblib.dump(segment_map, os.path.join(directory, "segment_map.joblib"))


def train_segmenter(df: pd.DataFrame) -> dict:
    pipeline, segment_map, sizes = fit_segmenter(df)
    save_segmenter(pipeline, segment_map)
    print(f"Distribuição dos segmentos: {sizes}")
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
