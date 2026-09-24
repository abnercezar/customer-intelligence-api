import pandas as pd
import numpy as np
from datetime import datetime

# Ordem das colunas usada no treino e na predição.
FEATURES = [
    "recency",
    "frequency",
    "monetary_avg",
    "monetary_total",
    "trend_slope",
    "avg_days_between",
]


def extract_features(orders: list) -> dict:
    df = pd.DataFrame(orders)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    now = datetime.now()
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
        avg_days_between = float(recency)

    return {
        "recency": recency,
        "frequency": frequency,
        "monetary_avg": monetary_avg,
        "monetary_total": monetary_total,
        "trend_slope": trend_slope,
        "avg_days_between": avg_days_between,
    }
