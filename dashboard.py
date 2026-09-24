import json
import os
import re
import unicodedata
from io import BytesIO, StringIO
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

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
    "maintain_engagement": "Manter o engajamento",
    "maintain_relationship": "Manter o relacionamento",
    "retention": "Fazer uma ação de retenção",
    "onboarding": "Acolher o cliente novo",
    "nurture": "Nutrir o relacionamento",
    "upsell": "Oferecer um upgrade",
    "reactivation": "Campanha de reativação",
    "monitor": "Só monitorar",
}

RISK_LABELS = {
    "low": "Baixo",
    "medium": "Médio",
    "high": "Alto",
}

CONFIDENCE_LABELS = {
    "low": "Baixa — 1 ou 2 compras. Dá para ver pouco.",
    "medium": "Média — 3 ou 4 compras. A tendência já aparece.",
    "high": "Alta — 5 ou mais compras. A leitura fica mais firme.",
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


def orders_from_frame(frame: pd.DataFrame) -> list:
    orders = []
    for _, row in frame.iterrows():
        if pd.isna(row["date"]) or pd.isna(row["value"]):
            continue
        orders.append({"date": as_date(row["date"]), "value": float(row["value"])})
    return orders


def copy_orders(orders: list) -> list:
    return [{"date": as_date(order["date"]), "value": float(order["value"])} for order in orders]


def apply_sample(customer_id: str) -> None:
    customer = next(item for item in load_samples() if item["customer_id"] == customer_id)
    st.session_state.customer_id = customer["customer_id"]
    st.session_state.orders = copy_orders(customer["orders"])
    st.session_state.editor_version += 1
    st.session_state.analyze_now = True


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
    if customer_id != "importado" or not str(st.session_state.customer_id).strip():
        st.session_state.customer_id = customer_id
    st.session_state.orders = copy_orders(orders)
    st.session_state.editor_version += 1
    st.session_state.analyze_now = True


def start_blank() -> None:
    st.session_state.customer_id = ""
    st.session_state.orders = []
    st.session_state.editor_version += 1
    st.session_state.result = None
    st.session_state.analyze_now = False


def init_state() -> None:
    if "orders" in st.session_state:
        return
    first = load_samples()[0]
    st.session_state.customer_id = first["customer_id"]
    st.session_state.orders = copy_orders(first["orders"])
    st.session_state.editor_version = 0
    st.session_state.result = None
    st.session_state.analyze_now = True


def purchase_count(count: int) -> str:
    if count == 1:
        return "1 compra"
    return f"{count} compras"


def summary(data: dict, orders: list) -> str:
    dates = sorted(order["date"] for order in orders)
    period = dates[0] if len(dates) == 1 else f"{dates[0]} a {dates[-1]}"
    total = sum(order["value"] for order in orders)
    segment = SEGMENT_LABELS.get(data["segment"], data["segment"])
    trend = TREND_LABELS.get(data["purchase_trend"], data["purchase_trend"])
    action = ACTION_LABELS.get(data["recommended_action"], data["recommended_action"])
    return (
        f"A conta usou {purchase_count(len(orders))}, de {period}, somando {money(total)}. "
        f"O valor está {trend}. "
        f"O sinal de abandono é {risk_label(data).lower()}. "
        f"O grupo mais parecido é {segment}. "
        f"Sugestão: {action}. "
        f"O nome “{data['customer_id']}” só identifica a resposta — ele não muda o cálculo."
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
            "error": "A API não está no ar. No outro terminal: $env:API_KEY = \"teste123\" e depois uvicorn app.main:app --reload."
        }
    except requests.HTTPError as error:
        status = error.response.status_code
        if status == 403:
            message = "A API recusou a chave. Suba a API com $env:API_KEY = \"teste123\" e use a mesma chave neste terminal do dashboard."
        elif status == 422:
            message = "A API recusou os dados: " + validation_message(error.response)
        else:
            message = f"A API recusou o pedido ({status})."
        st.session_state.result = {"error": message}
    except requests.RequestException as error:
        st.session_state.result = {"error": f"Não foi possível falar com a API: {error}"}


def render_result() -> None:
    result = st.session_state.result
    if not result:
        st.info("A tabela pode já estar preenchida. A análise só acontece quando você clica em **Analisar cliente** ou em um dos exemplos.")
        return

    if "error" in result:
        st.error(result["error"])
        return

    data = result["data"]
    orders = result["orders"]
    st.markdown(summary(data, orders))

    if data["segment"] == "new" and len(orders) > 1:
        st.caption("“Novo” aqui quer dizer histórico curto para o agrupamento. Não quer dizer que o cadastro acabou de ser criado.")

    risk, segment, trend = st.columns(3)
    risk.metric("Sinal de abandono", risk_label(data))
    segment.metric("Grupo", SEGMENT_LABELS.get(data["segment"], data["segment"]))
    trend.metric("Valor das compras", TREND_LABELS.get(data["purchase_trend"], data["purchase_trend"]))

    st.markdown(f"**O que fazer:** {ACTION_LABELS.get(data['recommended_action'], data['recommended_action'])}")
    st.markdown(f"**Valor somado:** {value_label(data)}")
    st.markdown(f"**Confiança:** {CONFIDENCE_LABELS.get(data.get('confidence', ''), data.get('confidence', ''))}")

    st.markdown("**Por que essa leitura**")
    for reason in data["reasons"]:
        st.markdown(f"- {reason}")


def render_import() -> None:
    st.subheader("Importar planilha")
    st.caption(
        "Uma linha por compra, com as colunas cliente, data e valor. "
        "Aceita CSV, TXT e Excel .xlsx. "
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
    if len(customer_ids) == 1:
        chosen = customer_ids[0]
        st.caption(f"1 cliente neste arquivo, {purchase_count(len(grouped[chosen]))}.")
    else:
        chosen = st.selectbox("Cliente neste arquivo", customer_ids, key="import_customer")
        st.caption(f"O arquivo tem {len(customer_ids)} clientes. Escolha um por vez e carregue na tabela.")

    if st.button("Carregar na tabela", key="load_import"):
        load_imported_customer(chosen, grouped[chosen])


def main() -> None:
    st.set_page_config(page_title="Inteligência do cliente", page_icon="🧠", layout="wide")
    init_state()

    st.title("O que está acontecendo com este cliente?")
    st.markdown(
        "Não existe uma lista de clientes aqui. Você traz o histórico de compras "
        "(ou escolhe um exemplo pronto) e a API lê **só as datas e os valores**. "
        "O ID é um apelido para você achar a resposta."
    )

    st.subheader("Exemplos prontos")
    st.caption("Cada botão preenche o apelido e as compras correspondentes, e já pede a análise.")
    columns = st.columns(3)
    for column, customer in zip(columns, load_samples()):
        customer_id = customer["customer_id"]
        with column:
            st.button(
                customer_id.replace("_", " "),
                key=f"load_{customer_id}",
                on_click=apply_sample,
                args=(customer_id,),
                use_container_width=True,
            )
            st.caption(SAMPLE_HINTS.get(customer_id, ""))

    st.button(
        "Limpar e digitar um cliente meu",
        key="start_blank",
        on_click=start_blank,
    )
    st.caption("Apaga o exemplo. O cliente não precisa existir no sistema: escreva um código seu e as compras reais.")

    render_import()

    form, result = st.columns([1, 1], gap="large")

    with form:
        st.subheader("Compras que entram na conta")
        st.text_input(
            "Apelido do cliente",
            key="customer_id",
            help="Pode ser o código da sua loja. A API devolve o mesmo texto e não usa ele para calcular.",
        )

        if st.session_state.orders:
            frame = pd.DataFrame(st.session_state.orders)
            frame["date"] = pd.to_datetime(frame["date"]).dt.date
        else:
            frame = pd.DataFrame(
                {
                    "date": pd.Series(dtype="datetime64[ns]"),
                    "value": pd.Series(dtype="float64"),
                }
            )
        edited = st.data_editor(
            frame,
            key=f"orders_editor_{st.session_state.editor_version}",
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "date": st.column_config.DateColumn("Data", format="YYYY-MM-DD"),
                "value": st.column_config.NumberColumn("Valor (R$)", min_value=0.0, format="R$ %.2f"),
            },
        )
        analyze_clicked = st.button("Analisar cliente", type="primary", use_container_width=True)

    if analyze_clicked or st.session_state.pop("analyze_now", False):
        customer_id = st.session_state.customer_id.strip() or "sem_nome"
        with st.spinner("Lendo o histórico..."):
            analyze(customer_id, orders_from_frame(edited))

    with result:
        st.subheader("Leitura")
        render_result()

    with st.expander("Como mandar isso de outro sistema"):
        st.markdown(
            "`POST /analyze` recebe **um** cliente, um objeto. "
            "O arquivo `data/sample_customers.json` é uma **lista** de três clientes: "
            "essa lista vai em `POST /batch`, não em `/analyze`."
        )
        st.code(
            """{
  "customer_id": "cliente_ativo",
  "orders": [
    {"date": "2026-03-10", "value": 300},
    {"date": "2026-04-15", "value": 350}
  ]
}""",
            language="json",
        )


if __name__ == "__main__":
    main()
