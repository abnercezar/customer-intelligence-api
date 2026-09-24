import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional

# Ordem das colunas usada no treino e na predição.
FEATURES = [
    "recency",
    "frequency",
    "monetary_avg",
    "monetary_total",
    "trend_slope",
    "avg_days_between",
]


def extract_features(orders: list, reference_date: Optional[datetime] = None) -> dict:
    """reference_date: "hoje" do cálculo. No treino é a data de corte; na API, agora."""
    df = pd.DataFrame(orders)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    now = reference_date or datetime.now()
    recency = (now - df["date"].max()).days
    frequency = len(df)
    monetary_avg = df["value"].mean()
    monetary_total = df["value"].sum()

    if len(df) >= 2:
        x = np.arange(len(df))
        trend_slope = float(np.polyfit(x, df["value"].values, 1)[0])
        avg_days_between = float(df["date"].diff().dropna().dt.days.mean())
    else:
        trend_slope = 0.0
        # Sem histórico suficiente para calcular intervalo — 0.0 indica desconhecido.
        avg_days_between = 0.0

    # Baseline individual: compara o cliente com o próprio histórico.
    # Só calculado com 3+ eventos para ter média estável.
    if len(df) >= 3:
        expected_interval: Optional[float] = avg_days_between
        interval_deviation: Optional[float] = float(recency) - avg_days_between
    else:
        expected_interval = None
        interval_deviation = None

    # Confiança baseada na quantidade de eventos disponíveis.
    if frequency >= 5:
        confidence = "high"
    elif frequency >= 3:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        # Features usadas pelo modelo ML (ordem fixa — ver FEATURES)
        "recency": recency,
        "frequency": frequency,
        "monetary_avg": monetary_avg,
        "monetary_total": monetary_total,
        "trend_slope": trend_slope,
        "avg_days_between": avg_days_between,
        # Features para lógica de negócio e explicabilidade (não entram no modelo)
        "expected_interval": expected_interval,
        "interval_deviation": interval_deviation,
        "confidence": confidence,
    }
