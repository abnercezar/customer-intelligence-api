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
    payload = [
        SAMPLE_PAYLOAD,
        {"customer_id": "sem_pedidos", "orders": []},
        {"customer_id": "data_invalida", "orders": [{"date": "não é data", "value": 10}]},
    ]
    response = client.post("/batch", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data[0]["analysis"] is not None
    assert data[1]["analysis"] is None and data[1]["error"]
    assert data[2]["analysis"] is None and data[2]["error"]


def test_batch_empty_returns_400():
    assert client.post("/batch", json=[]).status_code == 400


def test_batch_over_limit_returns_400():
    payload = [SAMPLE_PAYLOAD] * (MAX_BATCH_SIZE + 1)
    assert client.post("/batch", json=payload).status_code == 400
