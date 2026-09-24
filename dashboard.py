import os
import streamlit as st
import requests
import pandas as pd
from datetime import date, timedelta

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/analyze")
API_KEY = os.getenv("API_KEY", "")

SEGMENT_LABELS = {
    "champion":  "🏆 Champion",
    "loyal":     "💚 Leal",
    "potential": "🌱 Potencial",
    "at_risk":   "⚠️ Em Risco",
    "new":       "🆕 Novo",
}

TREND_LABELS = {
    "growing":  "📈 Crescendo",
    "stable":   "➡️ Estável",
    "declining": "📉 Caindo",
}

ACTION_LABELS = {
    "maintain_engagement": "Manter engajamento",
    "upsell":              "Oferecer upgrade / upsell",
    "reactivation":        "Campanha de reativação",
    "onboarding":          "Iniciar onboarding",
    "nurture":             "Nutrir relacionamento",
    "monitor":             "Monitorar",
}

SEGMENT_COLORS = {
    "champion":  "green",
    "loyal":     "green",
    "potential": "orange",
    "at_risk":   "red",
    "new":       "blue",
}


def churn_color(risk: float) -> str:
    if risk < 0.35:
        return "normal"
    if risk < 0.65:
        return "off"
    return "inverse"


st.set_page_config(
    page_title="Customer Intelligence",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Customer Intelligence API")
st.caption("Analise o comportamento de um cliente e receba previsão de churn, segmento e ação recomendada.")

st.divider()

# --- Formulário ---
col_form, col_result = st.columns([1, 1], gap="large")

with col_form:
    st.subheader("Dados do cliente")

    customer_id = st.text_input("ID do cliente", value="cliente_001")

    st.markdown("**Histórico de compras**")
    st.caption("Adicione as compras do cliente abaixo.")

    if "orders" not in st.session_state:
        st.session_state.orders = [
            {"date": date.today() - timedelta(days=90), "value": 500.0},
            {"date": date.today() - timedelta(days=60), "value": 350.0},
            {"date": date.today() - timedelta(days=30), "value": 200.0},
        ]

    orders_df = pd.DataFrame(st.session_state.orders)
    orders_df["date"] = pd.to_datetime(orders_df["date"]).dt.date
    edited = st.data_editor(
        orders_df,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "date":  st.column_config.DateColumn("Data", format="YYYY-MM-DD"),
            "value": st.column_config.NumberColumn("Valor (R$)", min_value=0.0, format="R$ %.2f"),
        },
    )

    analyze = st.button("Analisar cliente", type="primary", use_container_width=True)

# --- Resultado ---
with col_result:
    st.subheader("Resultado da análise")

    if analyze:
        orders_payload = [
            {"date": str(row["date"]), "value": float(row["value"])}
            for _, row in edited.iterrows()
            if row["value"] and row["date"]
        ]

        if not orders_payload:
            st.error("Adicione pelo menos uma compra.")
        else:
            with st.spinner("Analisando..."):
                try:
                    response = requests.post(
                        API_URL,
                        json={"customer_id": customer_id, "orders": orders_payload},
                        headers={"X-API-Key": API_KEY},
                        timeout=10,
                    )
                    response.raise_for_status()
                    data = response.json()

                    churn = data["churn_risk"]
                    segment = data["segment"]
                    trend = data["purchase_trend"]
                    value = data["customer_value"]
                    action = data["recommended_action"]
                    reasons = data["reasons"]

                    # Métricas principais
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Risco de Churn", f"{churn:.0%}")
                    m2.metric("Segmento", SEGMENT_LABELS.get(segment, segment))
                    m3.metric("Tendência", TREND_LABELS.get(trend, trend))

                    st.divider()

                    # Barra de risco
                    st.markdown("**Nível de risco**")
                    st.progress(churn)

                    # Ação recomendada
                    color = SEGMENT_COLORS.get(segment, "blue")
                    st.markdown(f"**Ação recomendada**")
                    st.info(f"💡 {ACTION_LABELS.get(action, action)}", icon=None)

                    # Razões
                    st.markdown("**Por que essa análise?**")
                    for reason in reasons:
                        st.markdown(f"- {reason}")

                    # Valor do cliente
                    value_map = {"high": "🔥 Alto", "medium": "🟡 Médio", "low": "🔵 Baixo"}
                    st.markdown(f"**Valor do cliente:** {value_map.get(value, value)}")

                except requests.exceptions.ConnectionError:
                    st.error("API offline. Rode: `uvicorn app.main:app --reload` na pasta do projeto.")
                except Exception as e:
                    st.error(f"Erro: {e}")
    else:
        st.info("Preencha os dados ao lado e clique em **Analisar cliente**.")

st.divider()

# --- Exemplo de payload JSON ---
with st.expander("Ver exemplo de payload JSON (para integração)"):
    st.code(
        """{
  "customer_id": "123",
  "orders": [
    {"date": "2026-06-10", "value": 250},
    {"date": "2026-07-12", "value": 280},
    {"date": "2026-08-14", "value": 310}
  ]
}""",
        language="json",
    )
    st.caption("Qualquer sistema pode enviar esse JSON para `POST /analyze` e receber a análise.")
