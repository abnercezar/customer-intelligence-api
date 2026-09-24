from io import BytesIO
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from dashboard import (
    SAMPLE_HINTS,
    group_orders,
    load_samples,
    money,
    orders_from_frame,
    read_orders_upload,
    risk_label,
    summary,
    validation_message,
    value_label,
)

DASHBOARD = Path(__file__).resolve().parents[1] / "dashboard.py"


def test_value_label_uses_the_cuts_from_the_trained_base():
    assert value_label({
        "customer_value": "medium",
        "value_medium_from": 800,
        "value_high_from": 3000,
    }) == f"Médio (de {money(800)} a {money(3000)})"


def test_value_label_without_cuts_does_not_invent_amounts():
    assert value_label({"customer_value": "high"}) == "Alto nesta base"


def test_risk_label_uses_api_level():
    assert risk_label({"churn_risk": 0.2, "risk_level": "high"}) == "Alto"


def test_risk_label_falls_back_to_score_for_old_api():
    assert risk_label({"churn_risk": 0.2}) == "Baixo"
    assert risk_label({"churn_risk": 0.5}) == "Médio"
    assert risk_label({"churn_risk": 0.8}) == "Alto"


class _Response:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


def test_validation_message_shows_the_api_reason():
    response = _Response({"detail": [{"msg": "Value error, Valor negativo: -10.0. Devoluções não são compras."}]})
    assert validation_message(response) == "Valor negativo: -10.0. Devoluções não são compras."


def test_validation_message_without_detail_has_a_generic_hint():
    assert validation_message(_Response({})) == "confira as datas e os valores."


def test_summary_explains_that_the_name_is_only_a_label():
    text = summary(
        {
            "customer_id": "cliente_ativo",
            "churn_risk": 0.37,
            "segment": "new",
            "purchase_trend": "declining",
            "recommended_action": "onboarding",
        },
        [
            {"date": "2026-06-26", "value": 500},
            {"date": "2026-07-26", "value": 350},
            {"date": "2026-08-25", "value": 200},
        ],
    )

    assert "3 compras" in text
    assert "2026-06-26 a 2026-08-25" in text
    assert money(1050) in text
    assert "caindo" in text
    # O score não é probabilidade calibrada: aparece como faixa, nunca como porcentagem.
    assert "sinal de abandono é médio" in text
    assert "37%" not in text
    assert "Novo" in text
    assert "Acolher o cliente novo" in text
    assert "não muda o cálculo" in text


def test_every_sample_customer_has_a_plain_description():
    ids = {item["customer_id"] for item in load_samples()}
    assert ids == set(SAMPLE_HINTS)


def test_empty_editor_rows_are_ignored():
    frame = pd.DataFrame(
        [
            {"date": "2026-03-10", "value": 300.0},
            {"date": None, "value": None},
        ]
    )
    assert orders_from_frame(frame) == [{"date": "2026-03-10", "value": 300.0}]


def test_blank_start_clears_the_sample_customer():
    app = AppTest.from_file(DASHBOARD, default_timeout=20).run()
    app.button(key="start_blank").click().run()

    assert not app.exception
    assert app.text_input[0].value == ""
    assert app.session_state["orders"] == []


class _Upload:
    def __init__(self, name: str, data: bytes):
        self.name = name
        self._data = data

    def getvalue(self) -> bytes:
        return self._data


def test_csv_with_semicolon_and_brazilian_amount_groups_customers():
    raw = "cliente;data;valor\n10;10/03/2026;1.050,50\n10;15/04/2026;80\n20;01/01/2026;10\n"
    grouped = read_orders_upload(_Upload("loja.csv", raw.encode("utf-8")))

    assert grouped["10"] == [
        {"date": "2026-03-10", "value": 1050.5},
        {"date": "2026-04-15", "value": 80.0},
    ]
    assert grouped["20"] == [{"date": "2026-01-01", "value": 10.0}]


def test_excel_reads_numeric_customer_ids():
    buffer = BytesIO()
    pd.DataFrame(
        {
            "cliente": [10, 10],
            "data": ["2026-03-10", "2026-04-15"],
            "valor": [300, 350],
        }
    ).to_excel(buffer, index=False, engine="openpyxl")

    grouped = read_orders_upload(_Upload("loja.xlsx", buffer.getvalue()))

    assert grouped["10"][0] == {"date": "2026-03-10", "value": 300.0}
    assert len(grouped["10"]) == 2


def test_sheet_without_customer_column_is_one_history():
    grouped = group_orders(pd.DataFrame({"data": ["2026-01-02"], "valor": [15]}))
    assert grouped == {"importado": [{"date": "2026-01-02", "value": 15.0}]}


def test_datetime_with_hour_is_kept_as_the_calendar_day():
    frame = pd.DataFrame(
        {
            "CustomerID": [10001],
            "InvoiceDate": ["2025-12-13 18:08:36"],
            "TotalAmount": [-179.87],
        }
    )
    grouped = group_orders(frame)
    assert grouped["10001"] == [{"date": "2025-12-13", "value": -179.87}]


def test_invoice_columns_in_english_are_read_as_cliente_data_e_valor():
    frame = pd.DataFrame(
        {
            "Invoice": ["A"],
            "CustomerID": [17850],
            "InvoiceDate": ["2026-03-10"],
            "UnitPrice": [2.5],
            "TotalAmount": [150.0],
        }
    )
    grouped = group_orders(frame)
    assert grouped["17850"] == [{"date": "2026-03-10", "value": 150.0}]


def test_customer_summary_txt_explains_that_rows_are_not_purchases():
    raw = (
        "user_id,nome,data_cadastro,ultimo_login,gasto_mensal,total_compras\n"
        "U0001,Mariana Rocha,2024-03-12,2026-08-30,72.5,6\n"
    )
    with pytest.raises(ValueError, match="lista clientes, não compras"):
        read_orders_upload(_Upload("deepseek_csv.txt", raw.encode("utf-8")))


def test_missing_value_column_names_the_headers_found():
    with pytest.raises(ValueError, match="coluna de valor"):
        group_orders(pd.DataFrame({"data": ["2026-01-02"], "nome": ["Ana"]}))


def test_dashboard_opens_on_cliente_ativo_and_explains_the_id():
    app = AppTest.from_file(DASHBOARD, default_timeout=20)
    app.run()

    assert not app.exception
    page = " ".join(block.value for block in list(app.markdown) + list(app.caption))
    assert "apelido" in page
    assert "Uma linha por compra" in page
    assert app.text_input[0].value == "cliente_ativo"
