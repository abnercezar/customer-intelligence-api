import os

from fastapi.testclient import TestClient
from app.main import app, MAX_BATCH_SIZE

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
    monkeypatch.delenv("API_KEY")
    assert anonymous.post("/analyze", json=SAMPLE_PAYLOAD).status_code == 500
    assert client.post("/analyze", json=SAMPLE_PAYLOAD).status_code == 500


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
