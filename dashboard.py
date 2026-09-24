import json
import os
import re
import unicodedata
from datetime import datetime
from io import BytesIO, StringIO
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

from app.ml.features import extract_features
from app.ml.orders import derive_thresholds

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/analyze")
API_KEY = os.getenv("API_KEY", "")
SAMPLES_PATH = Path(__file__).resolve().parent / "data" / "sample_customers.json"

SEGMENT_LABELS = {
    "champion": "Campeão",
    "loyal": "Leal",
    "potential": "Potencial",
    "at_risk": "Em risco",
    "new": "Novo",
}

TREND_LABELS = {
    "growing": "subindo",
    "stable": "estável",
    "declining": "caindo",
}

ACTION_LABELS = {
    "maintain_engagement": "Pode deixar quieto",
    "maintain_relationship": "Pode deixar quieto",
    "retention": "Vale chamar",
    "onboarding": "Mande um oi, é cliente novo",
    "nurture": "Vale um oi, sem pressa",
    "upsell": "Dá para oferecer algo a mais",
    "reactivation": "Vale chamar de volta",
    "monitor": "Só olhar de vez em quando",
}

RISK_LABELS = {
    "low": "Baixo",
    "medium": "Médio",
    "high": "Alto",
}

CONFIDENCE_LABELS = {
    "low": "Poucas compras. Posso errar.",
    "medium": "Já dá para ver um costume.",
    "high": "Bastante compras. A leitura fica mais firme.",
}

VALUE_LABELS = {
    "high": "Alto nesta base",
    "medium": "Médio nesta base",
    "low": "Baixo nesta base",
}

MAX_IMPORT_ROWS = 20000
CUSTOMER_COLUMNS = ("cliente", "cliente_id", "id_cliente", "customer_id", "customer")
DATE_COLUMNS = ("data", "data_compra", "data_pedido", "date", "invoice_date", "order_date")
VALUE_COLUMNS = ("valor", "total_amount", "value", "total", "amount", "preco", "price", "unit_price")

SAMPLE_TITLES = {
    "cliente_ativo": "Ana compra todo mês",
    "cliente_em_risco": "Bruno sumiu em abril",
    "cliente_novo": "Caio comprou uma vez",
}

SAMPLE_HINTS = {
    "cliente_ativo": "Cinco compras, de março a julho, com o valor subindo.",
    "cliente_em_risco": "Quatro compras, de janeiro a abril, com o valor caindo. A última foi há meses.",
    "cliente_novo": "Uma compra só, em 15 de setembro.",
}


def load_samples() -> list:
    with SAMPLES_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def risk_label(data: dict) -> str:
    """Faixa de risco. O score não é probabilidade calibrada, então não vira porcentagem."""
    level = data.get("risk_level")
    if level is None:
        # API antiga, sem risk_level: mesmos cortes do predictor.
        score = float(data["churn_risk"])
        level = "high" if score >= 0.65 else "medium" if score >= 0.35 else "low"
    return RISK_LABELS.get(level, level)


def value_label(data: dict) -> str:
    """Faixa de valor com os cortes da base que treinou o artefato."""
    level = data.get("customer_value", "")
    high = data.get("value_high_from")
    medium = data.get("value_medium_from")
    if level == "high" and isinstance(high, (int, float)):
        return f"Alto ({money(high)} ou mais)"
    if level == "medium" and isinstance(medium, (int, float)) and isinstance(high, (int, float)):
        return f"Médio (de {money(medium)} a {money(high)})"
    if level == "low" and isinstance(medium, (int, float)):
        return f"Baixo (menos de {money(medium)})"
    return VALUE_LABELS.get(level, level)


def money(value: float) -> str:
    formatted = f"{value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"R$ {formatted}"


def as_date(value) -> str:
    if hasattr(value, "isoformat"):
        return value.isoformat()[:10]
    return str(value)[:10]


def purchases_only(orders: list) -> tuple:
    """Devolução não é compra. Devolve as compras e quantas linhas saíram."""
    kept = [order for order in orders if float(order["value"]) > 0]
    return kept, len(orders) - len(kept)


def orders_from_frame(frame: pd.DataFrame) -> list:
    orders = []
    for _, row in frame.iterrows():
        if pd.isna(row["date"]) or pd.isna(row["value"]):
            continue
        value = float(row["value"])
        if value <= 0:
            continue
        orders.append({"date": as_date(row["date"]), "value": value})
    return orders


def copy_orders(orders: list) -> list:
    return [{"date": as_date(order["date"]), "value": float(order["value"])} for order in orders]


def apply_sample(customer_id: str) -> None:
    customer = next(item for item in load_samples() if item["customer_id"] == customer_id)
    orders = copy_orders(customer["orders"])
    st.session_state.setdefault("population", {})[customer_id] = orders
    st.session_state.customer_id = customer["customer_id"]
    st.session_state.orders = orders
    st.session_state.editor_version += 1
    st.session_state.selected_id = customer_id
    st.session_state.analyze_now = False


def _header(name: str) -> str:
    text = unicodedata.normalize("NFKD", str(name))
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"([a-z])([A-Z])", r"\1_\2", text.strip())
    return text.lower().replace(" ", "_")


def _find_column(columns, options: tuple) -> str | None:
    normalized = {_header(column): column for column in columns}
    for option in options:
        if option in normalized:
            return normalized[option]
    return None


def _customer_labels(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce")

        def one(value: float) -> str:
            if pd.isna(value):
                return ""
            if float(value).is_integer():
                return str(int(value))
            return str(value)

        return numeric.map(one)
    return series.astype(str).str.strip()


def _dates(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce")

    text = series.astype(str).str.strip()
    iso = text.str.match(r"^\d{4}-\d{2}-\d{2}")
    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    if iso.any():
        parsed.loc[iso] = pd.to_datetime(text.loc[iso], errors="coerce")
    if (~iso).any():
        parsed.loc[~iso] = pd.to_datetime(text.loc[~iso], dayfirst=True, errors="coerce")
    return parsed


def _money_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    def one(value: str) -> float:
        text = value.strip().replace("R$", "").replace(" ", "")
        if text.lower() in {"", "nan", "none"}:
            return float("nan")
        if "," in text and "." in text:
            if text.rfind(",") > text.rfind("."):
                text = text.replace(".", "").replace(",", ".")
            else:
                text = text.replace(",", "")
        elif "," in text:
            text = text.replace(",", ".")
        return float(text)

    return series.astype(str).map(one)


def _is_customer_summary(frame: pd.DataFrame) -> bool:
    headers = {_header(column) for column in frame.columns}
    has_person = bool(headers & {"user_id", "usuario_id", "nome"})
    has_aggregate = bool(headers & {"gasto_mensal", "total_compras", "dias_sem_login"})
    missing_purchase = (
        _find_column(frame.columns, DATE_COLUMNS) is None
        or _find_column(frame.columns, VALUE_COLUMNS) is None
    )
    return has_person and has_aggregate and missing_purchase


def group_orders(frame: pd.DataFrame) -> dict:
    """Agrupa uma planilha em {apelido: [{date, value}, ...]}.

    Aceita cabeçalhos em português ou inglês. Sem coluna de cliente, todas as
    linhas ficam sob o apelido "importado".
    """
    if frame.empty:
        raise ValueError("O arquivo não tem linhas.")
    if _is_customer_summary(frame):
        raise ValueError(
            "Este arquivo lista clientes, não compras. "
            "Cada linha traz cadastro, último acesso e gasto do mês. "
            "Para a análise, cada linha precisa ser uma compra, com cliente, data e valor."
        )
    if len(frame) > MAX_IMPORT_ROWS:
        raise ValueError(f"O arquivo passa de {MAX_IMPORT_ROWS} linhas. Envie um recorte.")

    date_column = _find_column(frame.columns, DATE_COLUMNS)
    value_column = _find_column(frame.columns, VALUE_COLUMNS)
    if date_column is None or value_column is None:
        found = ", ".join(str(column) for column in frame.columns)
        raise ValueError(
            "Faltou a coluna de data ou a coluna de valor. "
            "Os nomes podem ser cliente, data e valor. "
            f"Neste arquivo eu vi: {found}."
        )

    customer_column = _find_column(frame.columns, CUSTOMER_COLUMNS)
    dates = _dates(frame[date_column])
    values = _money_series(frame[value_column])
    if customer_column is None:
        customers = pd.Series(["importado"] * len(frame), index=frame.index)
    else:
        customers = _customer_labels(frame[customer_column])

    clean = pd.DataFrame({"customer_id": customers, "date": dates, "value": values})
    clean = clean.dropna(subset=["date", "value"])
    clean = clean[clean["customer_id"].str.lower() != "nan"]
    clean = clean[clean["customer_id"] != ""]
    if clean.empty:
        raise ValueError("Nenhuma linha tinha data e valor que desse para ler.")

    grouped = {}
    for customer_id, rows in clean.groupby("customer_id", sort=False):
        grouped[str(customer_id)] = [
            {"date": timestamp.strftime("%Y-%m-%d"), "value": float(amount)}
            for timestamp, amount in rows.sort_values("date")[["date", "value"]].itertuples(index=False)
        ]
    return grouped


def _frame_from_csv(raw: bytes) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            text = raw.decode(encoding)
        except UnicodeDecodeError:
            continue
        return pd.read_csv(StringIO(text), sep=None, engine="python")
    raise ValueError("Não consegui ler o texto do CSV. Salve o arquivo como UTF-8.")


def read_orders_upload(uploaded) -> dict:
    name = uploaded.name.lower()
    raw = uploaded.getvalue()
    if name.endswith(".csv") or name.endswith(".txt"):
        frame = _frame_from_csv(raw)
    elif name.endswith(".xlsx"):
        frame = pd.read_excel(BytesIO(raw), engine="openpyxl")
    else:
        raise ValueError("Use um arquivo .csv, .txt ou .xlsx. Se for .xls antigo, salve de novo como .xlsx.")
    return group_orders(frame)


def load_imported_customer(customer_id: str, orders: list) -> None:
    kept, skipped = purchases_only(orders)
    if customer_id != "importado" or not str(st.session_state.customer_id).strip():
        st.session_state.customer_id = customer_id
    st.session_state.orders = copy_orders(kept)
    st.session_state.setdefault("population", {})[customer_id] = copy_orders(kept)
    st.session_state.skipped_returns = skipped
    st.session_state.editor_version += 1
    st.session_state.selected_id = customer_id if kept else None
    st.session_state.analyze_now = False
    if not kept:
        st.session_state.result = {"error": "Essa pessoa só tem devoluções. Não há compra para ler."}


def load_base(grouped: dict) -> None:
    population = {}
    skipped = 0
    for customer_id, orders in grouped.items():
        kept, dropped = purchases_only(orders)
        skipped += dropped
        if kept:
            population[str(customer_id)] = copy_orders(kept)
    st.session_state.population = population
    st.session_state.skipped_returns = skipped
    st.session_state.selected_id = None
    st.session_state.result = None
    st.session_state.analyze_now = False


def open_person(customer_id: str) -> None:
    st.session_state.selected_id = customer_id
    st.session_state.customer_id = customer_id
    st.session_state.orders = copy_orders(st.session_state.population[customer_id])
    st.session_state.editor_version += 1


def back_home() -> None:
    st.session_state.selected_id = None


def start_blank() -> None:
    st.session_state.customer_id = ""
    st.session_state.orders = []
    st.session_state.population = {}
    st.session_state.editor_version += 1
    st.session_state.result = None
    st.session_state.skipped_returns = 0
    st.session_state.selected_id = None
    st.session_state.analyze_now = False


def init_state() -> None:
    if "orders" in st.session_state:
        st.session_state.setdefault("population", {})
        st.session_state.setdefault("selected_id", None)
        st.session_state.setdefault("skipped_returns", 0)
        return
    samples = load_samples()
    first = samples[0]
    st.session_state.population = {
        item["customer_id"]: copy_orders(item["orders"]) for item in samples
    }
    st.session_state.customer_id = first["customer_id"]
    st.session_state.orders = copy_orders(first["orders"])
    st.session_state.editor_version = 0
    st.session_state.result = None
    st.session_state.skipped_returns = 0
    st.session_state.selected_id = None
    st.session_state.analyze_now = False


def purchase_count(count: int) -> str:
    if count == 1:
        return "1 compra"
    return f"{count} compras"


def br_date(value) -> str:
    year, month, day = as_date(value)[:10].split("-")
    return f"{day}/{month}/{year}"


def customer_option(customer_id: str, orders: list) -> str:
    """Rótulo do menu: quem é, quando comprou por último e quanto somou."""
    kept, _skipped = purchases_only(orders)
    if not kept:
        return f"{customer_id} — só devoluções"
    last = max(order["date"] for order in kept)
    total = sum(float(order["value"]) for order in kept)
    return f"{customer_id} — última compra em {br_date(last)} — {money(total)}"


def purchase_series(orders: list) -> pd.DataFrame:
    frame = pd.DataFrame(orders)
    frame["Data"] = pd.to_datetime(frame["date"])
    frame["Valor"] = frame["value"].astype(float)
    return frame.groupby("Data", as_index=False)["Valor"].sum().sort_values("Data")


def headline(data: dict, orders: list) -> str:
    """A frase que dá para ler em voz alta, antes dos detalhes."""
    attention = risk_label(data)
    segment = data.get("segment")
    trend = data.get("purchase_trend")
    if len(orders) <= 2:
        return "Poucas compras. Ainda não dá para saber o costume."
    if segment == "new":
        if trend == "growing":
            return "O valor sobe, e ainda parece cliente novo. Mande um oi."
        if trend == "declining":
            return "O valor cai, e ainda parece cliente novo. Mande um oi."
        return "Ainda parece cliente novo. Mande um oi."
    if attention == "Alto" or segment == "at_risk":
        return "Faz tempo que não compra no ritmo de sempre. Vale chamar."
    if trend == "declining":
        return "Ainda compra, mas o valor está caindo."
    if trend == "growing" and attention == "Baixo":
        return "Compra seguido e o valor sobe. Pode deixar quieto."
    if attention == "Baixo":
        return "Está no costume. Pode deixar quieto."
    return "Mudou alguma coisa. Olhe as compras antes de decidir."


def summary(data: dict, orders: list) -> str:
    dates = sorted(as_date(order["date"]) for order in orders)
    period = br_date(dates[0]) if len(dates) == 1 else f"{br_date(dates[0])} a {br_date(dates[-1])}"
    total = sum(order["value"] for order in orders)
    segment = SEGMENT_LABELS.get(data["segment"], data["segment"])
    trend = TREND_LABELS.get(data["purchase_trend"], data["purchase_trend"])
    action = ACTION_LABELS.get(data["recommended_action"], data["recommended_action"])
    return (
        f"{purchase_count(len(orders)).capitalize()}, de {period}, somando {money(total)}. "
        f"O valor está {trend}. "
        f"O sinal de abandono é {risk_label(data).lower()}. "
        f"Parece com o grupo {segment}. "
        f"Sugestão: {action}. "
        f"O nome “{data['customer_id']}” só identifica a resposta — ele não muda o cálculo."
    )


def display_name(customer_id: str) -> str:
    title = SAMPLE_TITLES.get(customer_id, "")
    if title:
        return title.split()[0]
    return customer_id


ATTENTION_WORDS = {
    "Baixo": "Pode esperar",
    "Médio": "Olhe",
    "Alto": "Vale chamar",
}


def together(data: dict) -> str:
    """As três leituras não são alternativas: valem ao mesmo tempo."""
    segment = SEGMENT_LABELS.get(data.get("segment"), data.get("segment", ""))
    trend = TREND_LABELS.get(data.get("purchase_trend"), data.get("purchase_trend", ""))
    return (
        f"Atenção {risk_label(data).lower()}, grupo {segment} e compras {trend}. "
        "Os três valem juntos."
    )


def validation_message(response) -> str:
    """Primeira mensagem de um 422 do FastAPI, sem o prefixo 'Value error, '."""
    try:
        message = response.json()["detail"][0]["msg"]
    except (ValueError, KeyError, IndexError, TypeError):
        return "confira as datas e os valores."
    return message.removeprefix("Value error, ")


def analyze(customer_id: str, orders: list) -> None:
    if not orders:
        st.session_state.result = {"error": "Adicione pelo menos uma compra com data e valor."}
        return

    try:
        response = requests.post(
            API_URL,
            json={"customer_id": customer_id, "orders": orders},
            headers={"X-API-Key": API_KEY},
            timeout=10,
        )
        response.raise_for_status()
        st.session_state.result = {"data": response.json(), "orders": orders}
    except requests.exceptions.ConnectionError:
        st.session_state.result = {
            "error": "Não deu para calcular: o programa da análise não está ligado. "
            "No outro terminal: $env:API_KEY = \"teste123\" e depois uvicorn app.main:app --reload."
        }
    except requests.HTTPError as error:
        status = error.response.status_code
        if status == 403:
            message = "Não deu para calcular: a chave não confere. Suba a API com $env:API_KEY = \"teste123\" e use a mesma chave neste terminal."
        elif status == 422:
            message = "Não deu para calcular: " + validation_message(error.response)
        else:
            message = f"Não deu para calcular ({status})."
        st.session_state.result = {"error": message}
    except requests.RequestException as error:
        st.session_state.result = {"error": f"Não foi possível falar com a API: {error}"}


SLICE_LABELS = {
    "high": "Alto valor",
    "repeat": "Recorrentes",
    "new": "Novos",
    "inactive": "Inativos",
}

BEHAVIOR_LABELS = {
    "up": "↑ Comprando mais",
    "flat": "→ Estáveis",
    "down": "↓ Comprando menos",
    "silent": "⚠ Sem comprar",
}

BAND_LABELS = {
    "active": "Ativos",
    "watch": "Atenção",
    "risk": "Em risco",
}


def trend_of(slope: float, delta: float) -> str:
    if slope > delta:
        return "growing"
    if slope < -delta:
        return "declining"
    return "stable"


def classify_person(features: dict, trend: str) -> tuple:
    """Faixa e comportamento a partir do histórico da própria pessoa."""
    expected = features["expected_interval"]
    recency = features["recency"]
    late = expected is not None and recency > expected * 1.5
    stale = expected is None and recency > 60
    if late or stale:
        band = "risk"
    elif trend == "declining" or (expected is not None and recency > expected):
        band = "watch"
    else:
        band = "active"

    if band == "risk":
        behavior = "silent"
    elif trend == "growing":
        behavior = "up"
    elif trend == "declining":
        behavior = "down"
    else:
        behavior = "flat"
    return band, behavior


def file_slice(features: dict, band: str, value_high: float) -> str:
    if band == "risk":
        return "inactive"
    if features["frequency"] < 3:
        return "new"
    if features["monetary_total"] >= value_high:
        return "high"
    return "repeat"


def build_portraits(population: dict, reference: datetime | None = None) -> list:
    """Um retrato por pessoa, medido nesta planilha. Não chama o modelo gravado."""
    reference = reference or datetime.now()
    drafts = []
    for customer_id, orders in population.items():
        kept, _skipped = purchases_only(orders)
        if not kept:
            continue
        features = extract_features(kept, reference_date=reference)
        drafts.append({
            "customer_id": str(customer_id),
            "orders": kept,
            "features": features,
        })
    if not drafts:
        return []

    limits = derive_thresholds(pd.DataFrame([item["features"] for item in drafts]))
    portraits = []
    for item in drafts:
        features = item["features"]
        trend = trend_of(features["trend_slope"], limits["trend_delta"])
        band, behavior = classify_person(features, trend)
        portraits.append({
            **item,
            "trend": trend,
            "band": band,
            "behavior": behavior,
            "slice": file_slice(features, band, limits["value_high"]),
        })
    return portraits


def spoken(name: str, features: dict, trend: str) -> str:
    """A frase da pessoa, comparada com o costume dela."""
    if features["frequency"] <= 2:
        return "Poucas compras. Ainda não dá para saber o costume."
    expected = features["expected_interval"]
    recency = int(features["recency"])
    if expected and recency > expected:
        return (
            f"{name} costumava comprar a cada {int(expected)} dias. "
            f"Já se passaram {recency} dias desde a última compra."
        )
    if trend == "declining":
        return f"{name} ainda compra, mas o valor está caindo."
    if trend == "growing":
        return f"{name} compra seguido e o valor sobe."
    return f"{name} está no costume."


def attention_line(features: dict, trend: str, band: str) -> str:
    recency = int(features["recency"])
    if band == "risk":
        return f"{recency} dias sem comprar"
    if trend == "declining":
        return "comprando menos"
    return "saiu do ritmo"


def next_step(band: str, trend: str, frequency: int) -> str:
    if band == "risk":
        return "Entrar em contato com o cliente"
    if trend == "declining":
        return "Olhar o valor antes que caia mais"
    if frequency < 3:
        return "Mande um oi, é cliente novo"
    return "Pode deixar quieto"


def monthly_revenue(population: dict) -> pd.DataFrame:
    records = []
    for orders in population.values():
        for order in orders:
            value = float(order["value"])
            if value <= 0:
                continue
            records.append({"Mês": as_date(order["date"])[:7], "Receita": value})
    if not records:
        return pd.DataFrame(columns=["Mês", "Receita"])
    frame = pd.DataFrame(records)
    return frame.groupby("Mês", as_index=False)["Receita"].sum()


def money_short(value: float) -> str:
    if value >= 1_000_000:
        return f"R$ {value / 1_000_000:.1f} mi".replace(".", ",")
    if value >= 10_000:
        return f"R$ {round(value / 1000):.0f} mil"
    return money(value)


def percent(count: int, total: int) -> str:
    if total <= 0:
        return "0%"
    return f"{count / total:.0%}"


def render_result() -> None:
    result = st.session_state.result
    if not result:
        st.info("Escolha um exemplo em cima, ou escreva as compras e clique em **Ver o que está acontecendo**.")
        return

    if "error" in result:
        st.error(result["error"])
        return

    data = result["data"]
    orders = result["orders"]
    who = display_name(data["customer_id"])
    st.subheader(f"{who}: {headline(data, orders)}")

    if len(orders) <= 2:
        st.caption(f"Uma compra só, em {br_date(orders[-1]['date'])}, de {money(orders[-1]['value'])}." if len(orders) == 1
                   else f"{purchase_count(len(orders))}. Ainda é cedo para ver um costume.")
        return

    risk, segment, trend = st.columns(3)
    risk.metric("Precisa de atenção?", ATTENTION_WORDS.get(risk_label(data), risk_label(data)))
    segment.metric("Parece com", SEGMENT_LABELS.get(data["segment"], data["segment"]))
    trend.metric("O valor", TREND_LABELS.get(data["purchase_trend"], data["purchase_trend"]))

    dates = sorted(as_date(order["date"]) for order in orders)
    st.caption(
        f"{purchase_count(len(orders)).capitalize()}, de {br_date(dates[0])} a {br_date(dates[-1])}, "
        f"somando {money(sum(order['value'] for order in orders))}."
    )

    if data.get("reasons"):
        st.markdown("**Por que**")
        for reason in data["reasons"]:
            st.markdown(f"- {reason}")


def render_chart() -> None:
    result = st.session_state.result
    if not result or "error" in result:
        return
    orders = result["orders"]
    if len(orders) <= 2:
        return

    st.caption("Cada ponto é uma compra. Um buraco grande na linha é um tempo sem comprar.")
    st.line_chart(purchase_series(orders), x="Data", y="Valor")
    skipped = st.session_state.get("skipped_returns") or 0
    if skipped:
        st.caption(f"Tirei {skipped} devoluções. Elas não entram no gráfico.")


def render_import() -> None:
    st.caption(
        "Uma linha por compra, com quem comprou, o dia e o valor. "
        "Pode ser CSV ou Excel. "
        "Sem a coluna de cliente, todas as linhas entram juntas."
    )
    uploaded = st.file_uploader("Arquivo de compras", type=["csv", "txt", "xlsx"])
    if uploaded is None:
        return

    try:
        grouped = read_orders_upload(uploaded)
    except ValueError as error:
        st.error(str(error))
        return
    except Exception:
        st.error("Não consegui ler esse arquivo. Use uma planilha .csv ou .xlsx com as colunas data e valor.")
        return

    customer_ids = list(grouped)
    labels = {customer_id: customer_option(customer_id, grouped[customer_id]) for customer_id in customer_ids}

    def option_label(customer_id: str) -> str:
        return labels[customer_id]

    if len(customer_ids) == 1:
        chosen = customer_ids[0]
        st.caption(labels[chosen])
    else:
        chosen = st.selectbox(
            "Quem você quer ver?",
            customer_ids,
            key="import_customer",
            format_func=option_label,
        )
        st.caption(f"O arquivo tem {len(customer_ids)} pessoas. Escolher e clicar já mostra a resposta.")

    st.button(
        "Ver esta base",
        key="load_base",
        on_click=load_base,
        args=(grouped,),
    )
    st.button(
        "Ver esta pessoa",
        key="load_import",
        on_click=load_imported_customer,
        args=(chosen, grouped[chosen]),
    )


def render_examples() -> None:
    st.caption("Exemplos")
    columns = st.columns(3)
    for column, customer in zip(columns, load_samples()):
        customer_id = customer["customer_id"]
        with column:
            st.button(
                SAMPLE_TITLES.get(customer_id, customer_id.replace("_", " ")),
                key=f"load_{customer_id}",
                on_click=apply_sample,
                args=(customer_id,),
                width="stretch",
            )


def render_center(portraits: list) -> None:
    population = st.session_state.population
    customers = len(portraits)
    revenue = sum(item["features"]["monetary_total"] for item in portraits)
    repeating = sum(1 for item in portraits if item["features"]["frequency"] >= 2)

    count_col, revenue_col, repeat_col = st.columns(3)
    count_col.metric("Clientes", f"{customers}")
    revenue_col.metric("Receita", money_short(revenue) if customers else "R$ 0")
    repeat_col.metric("Recorrentes", percent(repeating, customers))

    active = sum(1 for item in portraits if item["band"] == "active")
    watch = sum(1 for item in portraits if item["band"] == "watch")
    risk = sum(1 for item in portraits if item["band"] == "risk")
    active_col, watch_col, risk_col = st.columns(3)
    active_col.metric("Ativos", active)
    watch_col.metric("Atenção", watch)
    risk_col.metric("Em risco", risk)

    st.subheader("Receita por mês")
    revenue_frame = monthly_revenue(population)
    if revenue_frame.empty:
        st.caption("Sem compras para desenhar o mês.")
    else:
        st.bar_chart(revenue_frame, x="Mês", y="Receita")

    groups, behavior = st.columns(2)
    with groups:
        st.subheader("Nesta planilha")
        st.caption("Conta feita nesta base. Não é o modelo gravado.")
        for key in ("high", "repeat", "new", "inactive"):
            count = sum(1 for item in portraits if item["slice"] == key)
            st.write(f"{SLICE_LABELS[key]}  ·  {percent(count, customers)}")
    with behavior:
        st.subheader("Comportamento")
        for key in ("up", "flat", "down", "silent"):
            count = sum(1 for item in portraits if item["behavior"] == key)
            st.write(f"{BEHAVIOR_LABELS[key]}  ·  {percent(count, customers)}")

    st.subheader("Clientes que precisam de atenção")
    queue = [item for item in portraits if item["band"] != "active"]
    queue.sort(
        key=lambda item: item["features"]["recency"] / (item["features"]["expected_interval"] or 1),
        reverse=True,
    )
    if not queue:
        st.caption("Ninguém saiu do ritmo nesta base.")
    for item in queue[:8]:
        features = item["features"]
        name = display_name(item["customer_id"])
        st.button(
            f"{name}   ·   {attention_line(features, item['trend'], item['band'])}   ·   {money(features['monetary_total'])}",
            key=f"pick_{item['customer_id']}",
            on_click=open_person,
            args=(item["customer_id"],),
            width="stretch",
        )
    if len(queue) > 8:
        st.caption(f"Mais {len(queue) - 8} pessoas na mesma situação. As oito de cima são as mais atrasadas.")

    skipped = st.session_state.get("skipped_returns") or 0
    if skipped:
        st.caption(f"Tirei {skipped} devoluções. Elas não entram nas contas.")


def render_person(portrait: dict) -> None:
    features = portrait["features"]
    name = display_name(portrait["customer_id"])
    st.button("Voltar", key="back_home", on_click=back_home)
    st.subheader(name)
    st.markdown(f"*{spoken(name, features, portrait['trend'])}*")

    total, purchases, ticket, last = st.columns(4)
    total.metric("Valor total", money(features["monetary_total"]))
    purchases.metric("Compras", int(features["frequency"]))
    ticket.metric("Ticket médio", money(features["monetary_avg"]))
    last.metric("Última compra", f"há {int(features['recency'])} dias")

    band = portrait["band"]
    if band == "risk":
        st.error(BAND_LABELS[band])
    elif band == "watch":
        st.warning(BAND_LABELS[band])
    else:
        st.success(BAND_LABELS[band])

    st.subheader("Histórico de compras")
    if len(portrait["orders"]) <= 2:
        only = portrait["orders"][-1]
        st.caption(
            f"Uma compra só, em {br_date(only['date'])}, de {money(only['value'])}."
            if len(portrait["orders"]) == 1
            else f"{purchase_count(len(portrait['orders']))}. Ainda é cedo para ver um costume."
        )
    else:
        st.caption("Cada ponto é uma compra. Um buraco grande na linha é um tempo sem comprar.")
        st.line_chart(purchase_series(portrait["orders"]), x="Data", y="Valor")

    st.subheader("Comportamento")
    expected = features["expected_interval"]
    interval = f"a cada {int(expected)} dias" if expected else "ainda sem costume"
    trend = TREND_LABELS.get(portrait["trend"], portrait["trend"])
    behavior_rows = st.columns(5)
    behavior_rows[0].metric("Recência", f"{int(features['recency'])} dias")
    behavior_rows[1].metric("Frequência", purchase_count(int(features["frequency"])))
    behavior_rows[2].metric("Valor", money(features["monetary_total"]))
    behavior_rows[3].metric("Tendência", trend)
    behavior_rows[4].metric("Intervalo", interval)

    st.subheader("O que fazer")
    st.info(next_step(band, portrait["trend"], int(features["frequency"])))


def render_tools() -> None:
    with st.expander("Usar outras compras"):
        render_import()
        render_examples()
        st.button("Começar do zero", key="start_blank", on_click=start_blank)
        st.text_input("Nome", key="customer_id")
        many = len(st.session_state.orders) > 8
        if many:
            st.caption(f"{purchase_count(len(st.session_state.orders))} carregadas. A leitura está na página da pessoa.")
            edited = None
            analyze_clicked = False
        else:
            if st.session_state.orders:
                frame = pd.DataFrame(st.session_state.orders)
                frame["date"] = pd.to_datetime(frame["date"]).dt.date
            else:
                frame = pd.DataFrame({
                    "date": pd.Series(dtype="datetime64[ns]"),
                    "value": pd.Series(dtype="float64"),
                })
            edited = st.data_editor(
                frame,
                key=f"orders_editor_{st.session_state.editor_version}",
                num_rows="dynamic",
                width="stretch",
                column_config={
                    "date": st.column_config.DateColumn("Data", format="DD/MM/YYYY"),
                    "value": st.column_config.NumberColumn("Valor (R$)", min_value=0.0, format="R$ %.2f"),
                },
            )
            analyze_clicked = st.button("Ver o que está acontecendo", type="primary")

    if analyze_clicked:
        customer_id = st.session_state.customer_id.strip() or "sem_nome"
        orders = orders_from_frame(edited)
        st.session_state.population[customer_id] = orders
        st.session_state.orders = orders
        st.session_state.selected_id = customer_id if orders else None
        if not orders:
            st.session_state.result = {"error": "Adicione pelo menos uma compra com data e valor."}
        else:
            st.session_state.result = None
            st.rerun()


def main() -> None:
    st.set_page_config(page_title="Clientes", page_icon="🧠", layout="wide")
    init_state()

    st.title("Clientes")
    st.caption("O apelido é só um nome para achar a pessoa. Ele não muda o cálculo.")

    population = st.session_state.population
    stamp = (
        len(population),
        sum(len(orders) for orders in population.values()),
        round(sum(float(order["value"]) for orders in population.values() for order in orders), 2),
        datetime.now().date().isoformat(),
    )
    if st.session_state.get("portrait_stamp") != stamp:
        st.session_state.portrait_cache = build_portraits(population)
        st.session_state.portrait_stamp = stamp
    portraits = st.session_state.portrait_cache
    selected = st.session_state.get("selected_id")
    chosen = next((item for item in portraits if item["customer_id"] == selected), None)
    if chosen:
        render_person(chosen)
    elif not portraits:
        st.info("Abra uma planilha em **Usar outras compras**, ou escolha um exemplo.")
    else:
        render_center(portraits)

    if st.session_state.get("result") and "error" in st.session_state.result:
        st.error(st.session_state.result["error"])

    render_tools()


if __name__ == "__main__":
    main()
