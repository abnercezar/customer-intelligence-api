from io import BytesIO
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from datetime import datetime

from dashboard import (
    SAMPLE_HINTS,
    attention_line,
    build_portraits,
    classify_person,
    customer_option,
    file_slice,
    group_orders,
    headline,
    money_short,
    monthly_revenue,
    next_step,
    percent,
    purchases_only,
    spoken,
    together,
    trend_of,
    load_samples,
    money,
    orders_from_frame,
    purchase_series,
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
    assert "26/06/2026 a 25/08/2026" in text
    assert money(1050) in text
    assert "caindo" in text
    # O score não é probabilidade calibrada: aparece como faixa, nunca como porcentagem.
    assert "sinal de abandono é médio" in text
    assert "37%" not in text
    assert "Novo" in text
    assert "Mande um oi, é cliente novo" in text
    assert "não muda o cálculo" in text


def test_headline_says_we_cannot_tell_the_habit_from_a_short_history():
    text = headline(
        {"churn_risk": 0.37, "segment": "new", "purchase_trend": "declining"},
        [{"date": "2026-08-01", "value": 10}, {"date": "2026-09-01", "value": 8}],
    )
    assert text == "Poucas compras. Ainda não dá para saber o costume."


def test_headline_does_not_call_many_purchases_few_just_because_the_group_is_new():
    orders = [{"date": f"2026-0{month}-10", "value": 300 + month * 20} for month in range(3, 8)]
    text = headline(
        {"churn_risk": 0.2, "risk_level": "low", "segment": "new", "purchase_trend": "growing"},
        orders,
    )
    assert "Poucas compras" not in text
    assert "valor sobe" in text
    assert "Mande um oi" in text
    assert "deixar quieto" not in text


def test_together_says_the_three_readings_apply_at_once():
    text = together({
        "churn_risk": 0.2,
        "risk_level": "low",
        "segment": "new",
        "purchase_trend": "growing",
    })
    assert text == "Atenção baixo, grupo Novo e compras subindo. Os três valem juntos."


def test_headline_tells_to_call_when_the_person_left_their_rhythm():
    orders = [{"date": f"2026-0{month}-01", "value": 100} for month in range(1, 5)]
    text = headline(
        {"churn_risk": 0.8, "segment": "at_risk", "purchase_trend": "declining"},
        orders,
    )
    assert "Vale chamar" in text


def test_customer_option_shows_the_last_purchase_and_the_total():
    label = customer_option("10069", [
        {"date": "2026-01-02", "value": 40},
        {"date": "2026-03-12", "value": 800},
    ])
    assert label == f"10069 — última compra em 12/03/2026 — {money(840)}"


def test_purchase_series_sorts_by_date_and_sums_the_same_day():
    frame = purchase_series([
        {"date": "2026-03-01", "value": 10},
        {"date": "2026-01-01", "value": 5},
        {"date": "2026-01-01", "value": 7},
    ])
    assert list(frame["Valor"]) == [12, 10]
    assert frame["Data"].iloc[0] == pd.Timestamp("2026-01-01")


def test_every_sample_customer_has_a_plain_description():
    ids = {item["customer_id"] for item in load_samples()}
    assert ids == set(SAMPLE_HINTS)


def test_negative_values_are_left_out_of_the_reading():
    frame = pd.DataFrame(
        [
            {"date": "2026-03-10", "value": 300.0},
            {"date": "2026-04-01", "value": -135.17},
        ]
    )
    assert orders_from_frame(frame) == [{"date": "2026-03-10", "value": 300.0}]


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


def test_portraits_describe_this_file_without_the_saved_model():
    population = {
        item["customer_id"]: item["orders"] for item in load_samples()
    }
    portraits = build_portraits(population, reference=datetime(2026, 9, 24))
    assert len(portraits) == 3
    by_id = {item["customer_id"]: item for item in portraits}
    assert by_id["cliente_em_risco"]["band"] == "risk"
    assert by_id["cliente_novo"]["slice"] == "new"
    late = by_id["cliente_em_risco"]
    text = spoken("Bruno", late["features"], late["trend"])
    assert "costumava comprar a cada" in text
    assert "desde a última compra" in text


def test_open_page_shows_who_needs_attention_before_the_person():
    app = AppTest.from_file(DASHBOARD, default_timeout=20).run()

    assert not app.exception
    titles = [block.value for block in app.subheader]
    assert "Clientes que precisam de atenção" in titles
    assert "Receita por mês" in titles
    page = " ".join(block.value for block in list(app.markdown) + list(app.caption) + list(app.subheader))
    assert "probabilidade" not in page.lower()
    assert "churn" not in page.lower()

    app.button(key="load_cliente_ativo").click().run()
    assert not app.exception
    titles = [block.value for block in app.subheader]
    captions = [block.value for block in app.caption]
    assert "Ana" in titles
    assert any("Cada ponto é uma compra" in caption for caption in captions)


def _features(**overrides) -> dict:
    features = {
        "recency": 10,
        "frequency": 6,
        "monetary_avg": 100.0,
        "monetary_total": 600.0,
        "trend_slope": 0.0,
        "avg_days_between": 30.0,
        "expected_interval": 30.0,
        "interval_deviation": -20.0,
        "confidence": "high",
    }
    features.update(overrides)
    return features


def test_trend_of_treats_the_delta_boundary_as_stable():
    assert trend_of(10, 10) == "stable"
    assert trend_of(-10, 10) == "stable"
    assert trend_of(10.01, 10) == "growing"
    assert trend_of(-10.01, 10) == "declining"


def test_classify_person_marks_a_late_buyer_as_risk_and_silent():
    band, behavior = classify_person(_features(recency=50, expected_interval=30), "stable")
    assert (band, behavior) == ("risk", "silent")


def test_classify_person_marks_a_short_history_quiet_for_two_months_as_risk():
    band, behavior = classify_person(
        _features(recency=61, frequency=2, expected_interval=None),
        "stable",
    )
    assert (band, behavior) == ("risk", "silent")


def test_classify_person_watches_a_decline_that_is_still_on_time():
    band, behavior = classify_person(_features(recency=20, expected_interval=30), "declining")
    assert (band, behavior) == ("watch", "down")


def test_classify_person_watches_someone_just_past_their_habit():
    band, behavior = classify_person(_features(recency=40, expected_interval=30), "growing")
    assert (band, behavior) == ("watch", "up")


def test_classify_person_keeps_an_on_time_buyer_active():
    band, behavior = classify_person(_features(recency=20, expected_interval=30), "growing")
    assert (band, behavior) == ("active", "up")

    band, behavior = classify_person(_features(recency=20, expected_interval=30), "stable")
    assert (band, behavior) == ("active", "flat")


def test_file_slice_uses_risk_before_value_and_counts_a_short_history_as_new():
    assert file_slice(_features(frequency=8, monetary_total=9000), "risk", 1000) == "inactive"
    assert file_slice(_features(frequency=2, monetary_total=9000), "active", 1000) == "new"
    assert file_slice(_features(frequency=8, monetary_total=1000), "active", 1000) == "high"
    assert file_slice(_features(frequency=8, monetary_total=999), "active", 1000) == "repeat"


def test_spoken_sentence_follows_the_persons_own_habit():
    late = _features(recency=47, expected_interval=18, frequency=12)
    assert spoken("João", late, "declining") == (
        "João costumava comprar a cada 18 dias. Já se passaram 47 dias desde a última compra."
    )
    on_time = _features(recency=10, expected_interval=30, frequency=6)
    assert spoken("Ana", on_time, "declining") == "Ana ainda compra, mas o valor está caindo."
    assert spoken("Ana", on_time, "growing") == "Ana compra seguido e o valor sobe."
    assert spoken("Ana", on_time, "stable") == "Ana está no costume."
    assert spoken("Caio", _features(frequency=1, expected_interval=None), "stable") == (
        "Poucas compras. Ainda não dá para saber o costume."
    )


def test_attention_line_and_next_step_match_the_band():
    late = _features(recency=47)
    assert attention_line(late, "declining", "risk") == "47 dias sem comprar"
    assert attention_line(late, "declining", "watch") == "comprando menos"
    assert attention_line(late, "stable", "watch") == "saiu do ritmo"

    assert next_step("risk", "stable", 12) == "Entrar em contato com o cliente"
    assert next_step("watch", "declining", 12) == "Olhar o valor antes que caia mais"
    assert next_step("active", "stable", 1) == "Mande um oi, é cliente novo"
    assert next_step("active", "growing", 6) == "Pode deixar quieto"


def test_monthly_revenue_sums_a_month_and_skips_returns():
    frame = monthly_revenue({
        "10": [
            {"date": "2026-01-02", "value": 10},
            {"date": "2026-01-20", "value": 5},
            {"date": "2026-01-21", "value": -3},
            {"date": "2026-03-01", "value": 7},
        ]
    })
    assert list(frame["Mês"]) == ["2026-01", "2026-03"]
    assert list(frame["Receita"]) == [15, 7]
    empty = monthly_revenue({"10": [{"date": "2026-01-01", "value": -5}]})
    assert list(empty.columns) == ["Mês", "Receita"]
    assert empty.empty


def test_money_short_and_percent_keep_small_numbers_exact():
    assert money_short(9999) == money(9999)
    assert money_short(10_000) == "R$ 10 mil"
    assert money_short(1_500_000) == "R$ 1,5 mi"
    assert percent(0, 0) == "0%"
    assert percent(1, 3) == "33%"


def test_purchases_only_counts_the_returns_it_drops():
    kept, skipped = purchases_only([
        {"date": "2026-01-01", "value": 10},
        {"date": "2026-01-02", "value": -4},
        {"date": "2026-01-03", "value": 0},
    ])
    assert kept == [{"date": "2026-01-01", "value": 10}]
    assert skipped == 2


def test_customer_option_says_when_the_person_only_has_returns():
    assert customer_option("10", [{"date": "2026-01-01", "value": -5}]) == "10 — só devoluções"


def test_build_portraits_drops_returns_and_marks_a_long_silence_as_inactive():
    portraits = build_portraits(
        {
            "so_devolucao": [{"date": "2026-01-01", "value": -10}],
            "sumido": [
                {"date": "2026-01-01", "value": 100},
                {"date": "2026-02-01", "value": 80},
                {"date": "2026-03-01", "value": 60},
            ],
            "rico": [
                {"date": "2026-06-01", "value": 4000},
                {"date": "2026-07-01", "value": 4200},
                {"date": "2026-08-01", "value": 4500},
                {"date": "2026-09-01", "value": 4800},
            ],
        },
        reference=datetime(2026, 9, 24),
    )
    by_id = {item["customer_id"]: item for item in portraits}
    assert "so_devolucao" not in by_id
    assert by_id["sumido"]["band"] == "risk"
    assert by_id["sumido"]["slice"] == "inactive"
    assert by_id["rico"]["slice"] == "high"
    assert by_id["rico"]["band"] == "active"


def test_opening_a_late_person_shows_the_habit_and_the_chart():
    app = AppTest.from_file(DASHBOARD, default_timeout=20).run()
    app.button(key="pick_cliente_em_risco").click().run()

    assert not app.exception
    titles = [block.value for block in app.subheader]
    assert "Bruno" in titles
    assert "Histórico de compras" in titles
    assert "Comportamento" in titles
    assert "O que fazer" in titles
    assert any("costumava comprar a cada" in block.value for block in app.markdown)
    assert any("Cada ponto é uma compra" in block.value for block in app.caption)
    assert any("Entrar em contato com o cliente" in block.value for block in app.info)


def test_back_from_a_person_returns_to_the_base():
    app = AppTest.from_file(DASHBOARD, default_timeout=20).run()
    app.button(key="pick_cliente_em_risco").click().run()
    app.button(key="back_home").click().run()

    assert not app.exception
    titles = [block.value for block in app.subheader]
    assert "Clientes que precisam de atenção" in titles
    assert "Bruno" not in titles


def test_blank_start_asks_for_a_file_or_an_example():
    app = AppTest.from_file(DASHBOARD, default_timeout=20).run()
    app.button(key="start_blank").click().run()

    assert not app.exception
    assert any("Usar outras compras" in block.value for block in app.info)


def test_uploaded_base_replaces_the_examples_and_drops_returns():
    raw = (
        "cliente,data,valor\n"
        "10,2026-01-01,100\n"
        "10,2026-02-01,90\n"
        "10,2026-03-01,80\n"
        "10,2026-03-02,-20\n"
        "20,2026-09-01,50\n"
        "20,2026-09-10,60\n"
        "20,2026-09-20,70\n"
    ).encode()
    app = AppTest.from_file(DASHBOARD, default_timeout=20).run()
    app.file_uploader[0].upload("loja.csv", raw).run()
    assert not app.exception

    button = next(item for item in app.button if item.label == "Ver esta base")
    button.click().run()

    assert not app.exception
    assert app.session_state["selected_id"] is None
    assert set(app.session_state["population"]) == {"10", "20"}
    assert all(
        order["value"] > 0
        for orders in app.session_state["population"].values()
        for order in orders
    )
    metrics = {item.label: item.value for item in app.metric}
    assert metrics["Clientes"] == "2"
    assert any("Tirei 1 devoluções" in block.value for block in app.caption)
    assert "Ana" not in [block.value for block in app.subheader]


def test_opening_one_person_from_the_file_uses_that_id():
    raw = "cliente,data,valor\n30,2026-09-01,40\n30,2026-09-15,50\n30,2026-09-20,60\n".encode()
    app = AppTest.from_file(DASHBOARD, default_timeout=20).run()
    app.file_uploader[0].upload("loja.csv", raw).run()
    button = next(item for item in app.button if item.label == "Ver esta pessoa")
    button.click().run()

    assert not app.exception
    assert "30" in [block.value for block in app.subheader]
    assert "Ana" not in [block.value for block in app.subheader]


def test_manual_purchases_open_that_person():
    app = AppTest.from_file(DASHBOARD, default_timeout=20).run()
    button = next(item for item in app.button if item.label == "Ver o que está acontecendo")
    button.click().run()

    assert not app.exception
    assert "Ana" in [block.value for block in app.subheader]


def test_dashboard_opens_on_cliente_ativo_and_explains_the_id():
    app = AppTest.from_file(DASHBOARD, default_timeout=20)
    app.run()

    assert not app.exception
    page = " ".join(block.value for block in list(app.markdown) + list(app.caption))
    assert "apelido" in page
    assert "Uma linha por compra" in page
    assert app.text_input[0].value == "cliente_ativo"
    labels = [button.label for button in app.button]
    assert "Ana compra todo mês" in labels
    assert "Ver o que está acontecendo" in labels
