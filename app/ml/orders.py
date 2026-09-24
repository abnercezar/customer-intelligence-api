"""Pedidos de uma base: CSV → cortes temporais → rótulo de churn → limiares."""

import pandas as pd

from app.ml.features import extract_features

HORIZON_DAYS = 90

COLUMN_ALIASES = {
    "customer_id": ("customer_id", "customer", "cliente", "cliente_id", "id_cliente"),
    "date": ("date", "data", "data_compra", "data_pedido", "invoice_date", "order_date"),
    "value": ("value", "valor", "total", "amount", "total_amount", "preco", "price"),
}


def load_orders_csv(path: str) -> pd.DataFrame:
    """Lê um CSV de pedidos e devolve customer_id, date, value (um total por cliente por dia)."""
    with open(path, encoding="utf-8-sig") as handle:
        header = handle.readline()
    separator = ";" if header.count(";") > header.count(",") else ","
    raw = pd.read_csv(path, sep=separator, encoding="utf-8-sig")
    df = _rename_columns(raw)

    df["customer_id"] = (
        df["customer_id"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    )
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    invalid_dates = int(df["date"].isna().sum())
    if invalid_dates:
        raise ValueError(f"{invalid_dates} linhas com data inválida. Use YYYY-MM-DD.")
    invalid_values = int((df["value"].isna() | (df["value"] <= 0)).sum())
    if invalid_values:
        raise ValueError(
            f"{invalid_values} linhas com valor ausente, zero ou negativo. "
            "O treino usa só pedidos positivos."
        )
    if (df["customer_id"] == "").any() or (df["customer_id"] == "nan").any():
        raise ValueError("Há linhas sem cliente.")

    orders = (
        df.groupby(["customer_id", "date"], as_index=False)["value"].sum()
    )
    if orders.empty:
        raise ValueError("O CSV não tem pedidos utilizáveis.")
    return orders


def _rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    found = {str(column).strip().lower(): column for column in df.columns}
    rename = {}
    missing = []
    for canonical, aliases in COLUMN_ALIASES.items():
        match = next((found[alias] for alias in aliases if alias in found), None)
        if match is None:
            missing.append(canonical)
        else:
            rename[match] = canonical
    if missing:
        raise ValueError(
            "Faltam colunas: "
            + ", ".join(missing)
            + f". Encontradas: {', '.join(map(str, df.columns))}."
        )
    return df.rename(columns=rename)


def observable_month_starts(orders: pd.DataFrame, horizon_days: int = HORIZON_DAYS) -> list:
    """Inícios de mês em que ainda cabem `horizon_days` de futuro dentro da base."""
    dates = pd.to_datetime(orders["date"])
    start = pd.Timestamp(dates.min()).normalize()
    end = pd.Timestamp(dates.max()).normalize()
    horizon = pd.Timedelta(days=horizon_days)
    cursor = pd.Timestamp(start) + pd.offsets.MonthBegin(1)
    points = []
    while cursor + horizon <= end:
        points.append(pd.Timestamp(cursor).normalize())
        nxt = pd.Timestamp(cursor) + pd.offsets.MonthBegin(1)
        if nxt <= cursor:
            break
        cursor = nxt
    return points


def choose_cutoffs(orders: pd.DataFrame, horizon_days: int = HORIZON_DAYS):
    """Cortes trimestrais. O último é só teste. Devolve (treino, teste) ou ([], None)."""
    monthly = observable_month_starts(orders, horizon_days)
    if len(monthly) < 2:
        return [], None

    chosen = monthly[::3]
    if chosen[-1] != monthly[-1]:
        chosen = chosen + [monthly[-1]]
    if len(chosen) < 2:
        chosen = [monthly[0], monthly[-1]]
    return chosen[:-1], chosen[-1]


def build_training_set(orders: pd.DataFrame, cutoffs: list, horizon_days: int = HORIZON_DAYS) -> pd.DataFrame:
    """Features só com pedidos antes do corte. churn=1 se não comprou nos `horizon_days` seguintes."""
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


def derive_thresholds(snapshot: pd.DataFrame) -> dict:
    """Cortes de valor e de tendência medidos nesta base, não trazidos de outra."""
    totals = snapshot["monetary_total"].astype(float)
    averages = snapshot["monetary_avg"].astype(float)
    value_high = float(totals.quantile(0.75))
    value_medium = float(totals.quantile(0.40))
    if value_medium >= value_high:
        value_medium = value_high / 2 if value_high > 0 else 0.0

    typical_ticket = float(averages.median())
    if not pd.notna(typical_ticket) or typical_ticket <= 0:
        typical_ticket = 1.0

    return {
        "value_high": value_high,
        "value_medium": value_medium,
        "ticket_low": float(averages.quantile(0.25)),
        "trend_delta": max(typical_ticket * 0.10, 1.0),
    }
