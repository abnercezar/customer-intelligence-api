from datetime import datetime

import pandas as pd

from app.ml.features import extract_features
from app.ml.online_retail import build_training_set, clean_transactions


def test_extract_features_uses_reference_date_for_recency():
    orders = [{"date": "2011-01-01", "value": 100}]
    features = extract_features(orders, reference_date=datetime(2011, 1, 31))
    assert features["recency"] == 30


def test_clean_transactions_drops_invalid_rows_and_groups_by_day():
    raw = pd.DataFrame({
        "Invoice":     ["1001", "1001", "1002", "C1003", "1004", "1005", "1006"],
        "Quantity":    [2,       1,      3,      5,       -1,     4,      1],
        "Price":       [10.0,    5.0,    2.0,    10.0,    10.0,   0.0,    7.0],
        "InvoiceDate": ["2011-01-01 09:00", "2011-01-01 09:00", "2011-01-01 15:00",
                        "2011-01-02 10:00", "2011-01-03 10:00", "2011-01-04 10:00",
                        "2011-01-05 10:00"],
        "Customer ID": [12345.0, 12345.0, 12345.0, 12345.0, 12345.0, 12345.0, None],
    })

    orders = clean_transactions(raw)

    # Cancelamento (C), quantidade negativa, preço zero e cliente sem ID saem.
    # Os 2 pedidos do mesmo dia viram um só: 2*10 + 1*5 + 3*2 = 31.
    assert len(orders) == 1
    row = orders.iloc[0]
    assert row["customer_id"] == "12345"
    assert row["date"] == pd.Timestamp("2011-01-01")
    assert row["value"] == 31


def test_build_training_set_labels_churn_from_future_orders():
    orders = pd.DataFrame({
        "customer_id": ["voltou", "voltou", "sumiu", "so_depois"],
        "date": pd.to_datetime(["2011-01-10", "2011-03-15", "2011-01-20", "2011-03-01"]),
        "value": [100.0, 120.0, 80.0, 50.0],
    })

    df = build_training_set(orders, ["2011-02-01"], horizon_days=90).set_index("customer_id")

    # "so_depois" não tem histórico antes do corte, então não entra.
    assert set(df.index) == {"voltou", "sumiu"}
    assert df.loc["voltou", "churn"] == 0
    assert df.loc["sumiu", "churn"] == 1
    # Features só enxergam o passado: 1 pedido, recência contada a partir do corte.
    assert df.loc["voltou", "frequency"] == 1
    assert df.loc["voltou", "recency"] == 22


def test_build_training_set_ignores_orders_after_horizon():
    orders = pd.DataFrame({
        "customer_id": ["tarde", "tarde"],
        "date": pd.to_datetime(["2011-01-10", "2011-06-01"]),
        "value": [100.0, 100.0],
    })

    df = build_training_set(orders, ["2011-02-01"], horizon_days=90)
    assert df.iloc[0]["churn"] == 1
