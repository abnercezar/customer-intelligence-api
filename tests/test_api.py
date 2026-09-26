import os
from datetime import date, timedelta

from fastapi.testclient import TestClient
from app.main import app, MAX_BATCH_SIZE
from app.models.predictor import _action, _risk_level
from app.models.schemas import MAX_ORDERS_PER_CUSTOMER

API_KEY = "test-key"
os.environ["API_KEY"] = API_KEY

client = TestClient(app, headers={"X-API-Key": API_KEY})
anonymous = TestClient(app)

SAMPLE_PAYLOAD = {
    "customer_id": "123",
    "orders": [
        {"date": "2026-01-10", "value": 250},
        {"date": "2026-02-12", "value": 280},
        {"date": "2026-03-14", "value": 200},
        {"date": "2026-04-20", "value": 180},
    ],
}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_health_does_not_require_api_key():
    assert anonymous.get("/health").status_code == 200


def test_analyze_returns_expected_fields():
    response = client.post("/analyze", json=SAMPLE_PAYLOAD)
    assert response.status_code == 200

    data = response.json()
    assert data["customer_id"] == "123"
    assert 0.0 <= data["churn_risk"] <= 1.0
    assert data["segment"] in {"champion", "loyal", "at_risk", "new", "potential"}
    assert data["purchase_trend"] in {"growing", "stable", "declining"}
    assert data["customer_value"] in {"high", "medium", "low"}
    assert isinstance(data["reasons"], list)
    assert len(data["reasons"]) >= 1


def test_analyze_empty_orders_returns_400():
    response = client.post("/analyze", json={"customer_id": "x", "orders": []})
    assert response.status_code == 400


def test_analyze_single_order():
    payload = {"customer_id": "new_user", "orders": [{"date": "2026-09-01", "value": 99}]}
    response = client.post("/analyze", json=payload)
    assert response.status_code == 200
    assert response.json()["segment"] == "new"


def test_missing_api_key_returns_403():
    assert anonymous.post("/analyze", json=SAMPLE_PAYLOAD).status_code == 403
    assert anonymous.post("/batch", json=[SAMPLE_PAYLOAD]).status_code == 403


def test_wrong_api_key_returns_403():
    response = anonymous.post("/analyze", json=SAMPLE_PAYLOAD, headers={"X-API-Key": "errada"})
    assert response.status_code == 403


def test_unconfigured_api_key_blocks_requests(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("API_KEYS", raising=False)
    assert anonymous.post("/analyze", json=SAMPLE_PAYLOAD).status_code == 500
    assert client.post("/analyze", json=SAMPLE_PAYLOAD).status_code == 500


def test_second_company_key_is_accepted(monkeypatch):
    monkeypatch.setenv("API_KEYS", '{"loja-ana":"chave-loja"}')
    response = anonymous.post(
        "/analyze",
        json=SAMPLE_PAYLOAD,
        headers={"X-API-Key": "chave-loja"},
    )
    assert response.status_code == 200
    assert client.post("/analyze", json=SAMPLE_PAYLOAD).status_code == 200


def test_unknown_key_stays_forbidden_when_company_keys_exist(monkeypatch):
    monkeypatch.setenv("API_KEYS", '{"loja-ana":"chave-loja"}')
    response = anonymous.post(
        "/analyze",
        json=SAMPLE_PAYLOAD,
        headers={"X-API-Key": "outra"},
    )
    assert response.status_code == 403


def test_batch_returns_one_result_per_customer():
    payload = [
        SAMPLE_PAYLOAD,
        {"customer_id": "new_user", "orders": [{"date": "2026-09-01", "value": 99}]},
    ]
    response = client.post("/batch", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert [item["customer_id"] for item in data] == ["123", "new_user"]
    assert all(item["error"] is None for item in data)
    assert data[1]["analysis"]["segment"] == "new"


def test_batch_item_error_does_not_fail_whole_batch():
    # Erros de lógica (orders vazio) são tratados por item — não derrubam o lote.
    payload = [
        SAMPLE_PAYLOAD,
        {"customer_id": "sem_pedidos", "orders": []},
    ]
    response = client.post("/batch", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data[0]["analysis"] is not None
    assert data[1]["analysis"] is None and data[1]["error"]


def test_batch_invalid_date_returns_422():
    # Data inválida é erro de schema (Pydantic) — rejeita a requisição inteira com 422.
    payload = [
        SAMPLE_PAYLOAD,
        {"customer_id": "data_invalida", "orders": [{"date": "não é data", "value": 10}]},
    ]
    response = client.post("/batch", json=payload)
    assert response.status_code == 422


def test_batch_empty_returns_400():
    assert client.post("/batch", json=[]).status_code == 400


def test_batch_over_limit_returns_400():
    payload = [SAMPLE_PAYLOAD] * (MAX_BATCH_SIZE + 1)
    assert client.post("/batch", json=payload).status_code == 400


# --- Testes de confidence ---

def test_single_order_returns_low_confidence():
    payload = {"customer_id": "novo", "orders": [{"date": "2026-09-01", "value": 200}]}
    data = client.post("/analyze", json=payload).json()
    assert data["confidence"] == "low"


def test_three_orders_returns_medium_confidence():
    payload = {
        "customer_id": "medio",
        "orders": [
            {"date": "2026-07-01", "value": 200},
            {"date": "2026-08-01", "value": 220},
            {"date": "2026-09-01", "value": 210},
        ],
    }
    data = client.post("/analyze", json=payload).json()
    assert data["confidence"] == "medium"


def test_five_orders_returns_high_confidence():
    payload = {
        "customer_id": "fiel",
        "orders": [
            {"date": "2026-05-01", "value": 300},
            {"date": "2026-06-01", "value": 310},
            {"date": "2026-07-01", "value": 320},
            {"date": "2026-08-01", "value": 315},
            {"date": "2026-09-01", "value": 330},
        ],
    }
    data = client.post("/analyze", json=payload).json()
    assert data["confidence"] == "high"


# --- Testes de baseline individual ---

def test_overdue_customer_reason_mentions_expected_interval():
    # Cliente com intervalo habitual de ~30 dias, mas 90 dias sem comprar.
    payload = {
        "customer_id": "atrasado",
        "orders": [
            {"date": "2026-01-01", "value": 200},
            {"date": "2026-02-01", "value": 200},
            {"date": "2026-03-01", "value": 200},
            {"date": "2026-04-01", "value": 200},
            # última compra há ~176 dias a partir de set/2026
        ],
    }
    data = client.post("/analyze", json=payload).json()
    reasons_text = " ".join(data["reasons"])
    # Deve mencionar o intervalo habitual, não um threshold global genérico.
    assert "habitual" in reasons_text or "esperado" in reasons_text


def test_customer_within_expected_interval_no_overdue_reason():
    # Cliente que compra a cada ~90 dias — 60 dias de recência está dentro do padrão.
    payload = {
        "customer_id": "padrao_longo",
        "orders": [
            {"date": "2026-01-01", "value": 500},
            {"date": "2026-04-01", "value": 500},
            {"date": "2026-07-01", "value": 500},
        ],
    }
    data = client.post("/analyze", json=payload).json()
    reasons_text = " ".join(data["reasons"])
    # 60 dias de recência com intervalo habitual de 90 dias — não deve alarmar.
    assert "acima do esperado" not in reasons_text


# --- Validação de entrada ---

def _error_text(response) -> str:
    return " ".join(item["msg"] for item in response.json()["detail"])


def test_negative_value_returns_422():
    payload = {"customer_id": "devolucao", "orders": [{"date": "2026-03-01", "value": -50}]}
    response = client.post("/analyze", json=payload)
    assert response.status_code == 422
    assert "Valor negativo" in _error_text(response)


def test_zero_value_is_accepted():
    payload = {"customer_id": "brinde", "orders": [{"date": "2026-03-01", "value": 0}]}
    assert client.post("/analyze", json=payload).status_code == 200


def test_future_date_returns_422():
    future = (date.today() + timedelta(days=30)).isoformat()
    payload = {"customer_id": "futuro", "orders": [{"date": future, "value": 100}]}
    response = client.post("/analyze", json=payload)
    assert response.status_code == 422
    assert "futuro" in _error_text(response)


def test_tomorrow_is_accepted_for_timezone_slack():
    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    payload = {"customer_id": "fuso", "orders": [{"date": tomorrow, "value": 100}]}
    assert client.post("/analyze", json=payload).status_code == 200


def test_too_many_orders_for_one_customer_returns_422():
    orders = [{"date": "2026-01-01", "value": 10}] * (MAX_ORDERS_PER_CUSTOMER + 1)
    response = client.post("/analyze", json={"customer_id": "gigante", "orders": orders})
    assert response.status_code == 422


def test_batch_total_orders_over_limit_returns_400(monkeypatch):
    monkeypatch.setattr("app.main.MAX_BATCH_ORDERS", 5)
    response = client.post("/batch", json=[SAMPLE_PAYLOAD, SAMPLE_PAYLOAD])  # 8 pedidos
    assert response.status_code == 400


# --- Faixa de risco e ação ---

def test_risk_level_matches_churn_score():
    data = client.post("/analyze", json=SAMPLE_PAYLOAD).json()
    assert data["risk_level"] == _risk_level(data["churn_risk"])


def test_risk_level_cuts():
    assert _risk_level(0.0) == "low"
    assert _risk_level(0.34) == "low"
    assert _risk_level(0.35) == "medium"
    assert _risk_level(0.64) == "medium"
    assert _risk_level(0.65) == "high"


def test_high_risk_established_customer_gets_retention_even_if_clustered_as_new():
    # Caso real visto na API: 4 compras, 176 dias sumido, K-Means dizia "new" → onboarding.
    assert _action("new", "high", frequency=4) == "retention"


def test_high_risk_single_order_customer_keeps_segment_action():
    # Com 1 compra não há relação estabelecida para "reter".
    assert _action("new", "high", frequency=1) == "onboarding"


def test_low_risk_does_not_trigger_retention_for_at_risk_cluster():
    assert _action("at_risk", "low", frequency=5) == "monitor"


def test_segment_action_is_kept_when_risk_agrees():
    assert _action("champion", "low", frequency=8) == "maintain_engagement"
    assert _action("at_risk", "high", frequency=5) == "retention"


def test_security_headers_hide_the_reading_from_caches():
    response = anonymous.get("/health")
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["referrer-policy"] == "no-referrer"
    assert "default-src 'none'" in response.headers["content-security-policy"]


def test_body_over_limit_returns_413(monkeypatch):
    monkeypatch.setattr("app.protection.MAX_BODY_BYTES", 20)
    response = client.post("/analyze", json=SAMPLE_PAYLOAD)
    assert response.status_code == 413
    assert "grande demais" in response.json()["detail"]


def test_customer_id_rejects_blank_and_control_characters():
    blank = client.post("/analyze", json={"customer_id": "   ", "orders": SAMPLE_PAYLOAD["orders"]})
    assert blank.status_code == 422
    broken = client.post(
        "/analyze",
        json={"customer_id": "ana\nlinha", "orders": SAMPLE_PAYLOAD["orders"]},
    )
    assert broken.status_code == 422


def test_non_finite_value_returns_422():
    payload = {"customer_id": "infinito", "orders": [{"date": "2026-03-01", "value": 1e309}]}
    response = client.post("/analyze", json=payload)
    assert response.status_code == 422
    assert "Valor inválido" in _error_text(response)


def test_internal_error_does_not_leak_details(monkeypatch):
    def boom(orders, customer_id):
        raise RuntimeError("caminho /segredo e chave interna")

    monkeypatch.setattr("app.main.predict", boom)
    one = client.post("/analyze", json=SAMPLE_PAYLOAD)
    assert one.status_code == 500
    assert "segredo" not in one.text
    batch = client.post("/batch", json=[SAMPLE_PAYLOAD])
    assert batch.status_code == 200
    assert batch.json()[0]["error"] == "Erro interno ao processar análise."
    assert "segredo" not in batch.text


def test_rate_limit_per_key_returns_429(monkeypatch):
    from app.security import _hits

    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "2")
    _hits.clear()
    assert client.post("/analyze", json=SAMPLE_PAYLOAD).status_code == 200
    assert client.post("/analyze", json=SAMPLE_PAYLOAD).status_code == 200
    blocked = client.post("/analyze", json=SAMPLE_PAYLOAD)
    assert blocked.status_code == 429
    assert blocked.headers["retry-after"] == "60"
    _hits.clear()


def test_invalid_key_flood_does_not_block_the_right_key(monkeypatch):
    from app.security import _failures, _hits

    monkeypatch.setenv("AUTH_FAILURE_LIMIT_PER_MINUTE", "2")
    _failures.clear()
    _hits.clear()
    assert anonymous.post("/analyze", json=SAMPLE_PAYLOAD, headers={"X-API-Key": "errada"}).status_code == 403
    assert anonymous.post("/analyze", json=SAMPLE_PAYLOAD, headers={"X-API-Key": "outra"}).status_code == 403
    flooded = anonymous.post("/analyze", json=SAMPLE_PAYLOAD, headers={"X-API-Key": "mais-uma"})
    assert flooded.status_code == 429
    assert client.post("/analyze", json=SAMPLE_PAYLOAD).status_code == 200
    _failures.clear()
    _hits.clear()


def test_docs_explain_the_reading_and_which_routes_need_a_key():
    schema = client.get("/openapi.json").json()
    description = schema["info"]["description"]
    assert "Authorize" in description
    assert "`GET /health`" in description
    assert "não pedem chave" in description
    assert "X-API-Key" in description
    assert "POST /analyze" in description
    assert "K-Means" not in description
    assert "floresta" not in description.lower()
    assert "empresa" in schema["components"]["securitySchemes"]["APIKeyHeader"]["description"]
    analyze = schema["paths"]["/analyze"]["post"]
    assert analyze["summary"] == "Ler um cliente"
    assert "Try it out" in analyze["description"]
    assert "pode esperar" in analyze["description"]
    assert "não cancela o lote" in schema["paths"]["/batch"]["post"]["description"]
    assert schema["paths"]["/batch"]["post"]["summary"] == "Ler vários clientes"
    assert schema["paths"]["/health"]["get"]["summary"] == "Está no ar?"
    assert schema["components"]["schemas"]["HealthStatus"]["example"] == {"status": "healthy"}
    assert schema["components"]["schemas"]["ServiceStatus"]["example"] == {"status": "ok", "version": "1.0.0"}
    assert "security" not in schema["paths"]["/health"]["get"]
    assert schema["paths"]["/analyze"]["post"]["security"]
    example = schema["components"]["schemas"]["CustomerRequest"]["example"]
    assert example["customer_id"] == "ana"
    assert example["orders"][0]["date"] == "2026-03-10"
