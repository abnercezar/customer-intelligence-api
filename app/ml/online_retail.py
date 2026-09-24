"""
Dataset real Online Retail II (UCI): transações de uma loja online do Reino Unido,
dez/2009 a dez/2011. Valores em libras (GBP); boa parte dos clientes é atacadista.
https://archive.ics.uci.edu/dataset/502/online+retail+ii

Rótulo de churn por data de corte:
  features → calculadas só com os pedidos ANTES do corte
  churn=1  → cliente não comprou nos `horizon_days` dias DEPOIS do corte
"""
import io
import os
import urllib.request
import zipfile

import pandas as pd

from app.ml.features import extract_features

DATASET_URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
XLSX_PATH = os.path.join(RAW_DIR, "online_retail_II.xlsx")
ORDERS_CACHE_PATH = os.path.join(RAW_DIR, "online_retail_orders.csv.gz")


def download() -> str:
    if os.path.exists(XLSX_PATH):
        return XLSX_PATH

    os.makedirs(RAW_DIR, exist_ok=True)
    print(f"Baixando {DATASET_URL} ...")
    with urllib.request.urlopen(DATASET_URL, timeout=300) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))

    name = next(n for n in archive.namelist() if n.endswith(".xlsx"))
    with open(XLSX_PATH, "wb") as f:
        f.write(archive.read(name))
    return XLSX_PATH


def clean_transactions(raw: pd.DataFrame) -> pd.DataFrame:
    """Linhas de item → um pedido por cliente por dia (customer_id, date, value)."""
    df = raw.dropna(subset=["Customer ID"])
    df = df[~df["Invoice"].astype(str).str.startswith("C")]  # cancelamentos
    df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]

    df = pd.DataFrame({
        "customer_id": df["Customer ID"].astype(int).astype(str),
        "date": pd.to_datetime(df["InvoiceDate"]).dt.normalize(),
        "value": df["Quantity"] * df["Price"],
    })
    return df.groupby(["customer_id", "date"], as_index=False)["value"].sum()


def load_orders() -> pd.DataFrame:
    if os.path.exists(ORDERS_CACHE_PATH):
        return pd.read_csv(ORDERS_CACHE_PATH, dtype={"customer_id": str}, parse_dates=["date"])

    print("Lendo planilha (demora alguns minutos na primeira vez)...")
    # O arquivo tem uma aba por período (2009-2010 e 2010-2011).
    sheets = pd.read_excel(download(), sheet_name=None)
    orders = clean_transactions(pd.concat(sheets.values(), ignore_index=True))
    orders.to_csv(ORDERS_CACHE_PATH, index=False)
    return orders


def build_training_set(orders: pd.DataFrame, cutoffs: list, horizon_days: int = 90) -> pd.DataFrame:
    rows = []
    for cutoff in pd.to_datetime(cutoffs):
        history = orders[orders["date"] < cutoff]
        future = orders[(orders["date"] >= cutoff) & (orders["date"] < cutoff + pd.Timedelta(days=horizon_days))]
        returned = set(future["customer_id"])

        for customer_id, customer_orders in history.groupby("customer_id"):
            features = extract_features(
                customer_orders[["date", "value"]].to_dict("records"),
                reference_date=cutoff.to_pydatetime(),
            )
            features["customer_id"] = customer_id
            features["cutoff"] = cutoff
            features["churn"] = 0 if customer_id in returned else 1
            rows.append(features)

    return pd.DataFrame(rows)
