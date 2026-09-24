import pandas as pd
import pytest

from app.ml.gates import acceptance_failures
from app.ml.orders import choose_cutoffs, derive_thresholds, load_orders_csv
from app.ml import train as train_mod
from app.models import predictor


def _orders(start="2023-01-15", end="2025-06-01", customers=40) -> pd.DataFrame:
    rows = []
    start_at = pd.Timestamp(start)
    end_at = pd.Timestamp(end)
    for index in range(customers):
        customer_id = str(index)
        if index % 2 == 0:
            day = start_at
            while day <= end_at:
                rows.append({"customer_id": customer_id, "date": day, "value": 100.0 + index})
                day += pd.Timedelta(days=30)
        else:
            rows.append({"customer_id": customer_id, "date": start_at, "value": 80.0})
            rows.append({
                "customer_id": customer_id,
                "date": start_at + pd.Timedelta(days=20),
                "value": 80.0,
            })
    return pd.DataFrame(rows)


def test_choose_cutoffs_keeps_the_last_one_for_test_only():
    train_cutoffs, test_cutoff = choose_cutoffs(_orders(), horizon_days=90)
    assert test_cutoff is not None
    assert train_cutoffs
    assert all(cutoff < test_cutoff for cutoff in train_cutoffs)
    end = pd.Timestamp("2025-06-01")
    assert test_cutoff + pd.Timedelta(days=90) <= end


def test_choose_cutoffs_rejects_history_shorter_than_the_horizon():
    orders = pd.DataFrame({
        "customer_id": ["1", "1"],
        "date": pd.to_datetime(["2024-01-01", "2024-01-20"]),
        "value": [10.0, 12.0],
    })
    train_cutoffs, test_cutoff = choose_cutoffs(orders, horizon_days=90)
    assert train_cutoffs == []
    assert test_cutoff is None


def test_load_orders_csv_groups_a_day_and_accepts_portuguese_headers(tmp_path):
    path = tmp_path / "pedidos.csv"
    path.write_text(
        "cliente;data;valor\n"
        "10;2024-01-02;10.5\n"
        "10;2024-01-02;4.5\n"
        "11;2024-02-01;20\n",
        encoding="utf-8",
    )
    orders = load_orders_csv(str(path))
    assert len(orders) == 2
    row = orders.loc[orders["customer_id"] == "10"].iloc[0]
    assert row["value"] == 15
    assert row["date"] == pd.Timestamp("2024-01-02")


def test_load_orders_csv_rejects_non_positive_values(tmp_path):
    path = tmp_path / "pedidos.csv"
    path.write_text("customer_id,date,value\n1,2024-01-01,-5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="negativo"):
        load_orders_csv(str(path))


def test_derive_thresholds_keeps_high_above_medium():
    snapshot = pd.DataFrame({
        "monetary_total": [100, 200, 400, 800, 1600],
        "monetary_avg": [50, 80, 100, 200, 400],
    })
    limits = derive_thresholds(snapshot)
    assert limits["value_high"] > limits["value_medium"]
    assert limits["trend_delta"] > 0


def test_acceptance_rejects_a_measured_but_weak_result():
    report = {
        "train_customers": 1000,
        "test_customers": 400,
        "test_churn_rate": 0.4,
        "auc": 0.55,
        "accuracy": 0.60,
        "baseline": 0.60,
    }
    sizes = {name: 80 for name in ("at_risk", "champion", "loyal", "new", "potential")}
    failures = acceptance_failures(report, sizes)
    assert any("AUC" in item for item in failures)
    assert any("baseline" in item for item in failures)


def test_acceptance_passes_when_the_test_cutoff_beats_the_baseline():
    report = {
        "train_customers": 1000,
        "test_customers": 400,
        "test_churn_rate": 0.4,
        "auc": 0.80,
        "accuracy": 0.75,
        "baseline": 0.60,
    }
    sizes = {name: 80 for name in ("at_risk", "champion", "loyal", "new", "potential")}
    assert acceptance_failures(report, sizes) == []


def test_short_history_does_not_fit_or_write(tmp_path, monkeypatch):
    def explode():
        raise AssertionError("histórico curto não deve treinar")

    monkeypatch.setattr(train_mod, "build_model", explode)
    orders = pd.DataFrame({
        "customer_id": ["1", "1"],
        "date": pd.to_datetime(["2024-01-01", "2024-01-10"]),
        "value": [10.0, 10.0],
    })
    result = train_mod.train_from_orders(orders, source="curto", artifact_dir=tmp_path)
    assert result["saved"] is False
    assert any("90" in item for item in result["failures"])
    assert list(tmp_path.iterdir()) == []


def test_small_base_is_measured_and_does_not_replace_the_artifact(tmp_path, monkeypatch):
    seen = []
    real_build = train_mod.build_model

    def wrapped():
        model = real_build()
        original = model.fit

        def fit(X, y, **kwargs):
            seen.append(len(y))
            return original(X, y, **kwargs)

        model.fit = fit
        return model

    monkeypatch.setattr(train_mod, "build_model", wrapped)
    result = train_mod.train_from_orders(_orders(), source="pequena", artifact_dir=tmp_path)

    assert result["saved"] is False
    assert result["auc"] is not None
    assert result["test_customers"] > 0
    assert seen == [result["train_rows"]]
    assert result["final_fit_rows"] == result["train_rows"]
    assert list(tmp_path.iterdir()) == []


def test_predict_uses_thresholds_from_the_artifact(tmp_path, monkeypatch):
    import joblib

    path = tmp_path / "thresholds.joblib"
    joblib.dump({
        "value_high": 100.0,
        "value_medium": 40.0,
        "ticket_low": 10.0,
        "trend_delta": 1.0,
    }, path)
    monkeypatch.setattr(predictor, "THRESHOLDS_PATH", str(path))
    monkeypatch.setattr(predictor, "_thresholds", None)

    medium = predictor.predict([{"date": "2026-09-01", "value": 50}], "c")
    assert medium["customer_value"] == "medium"
    assert medium["value_medium_from"] == 40.0
    assert medium["value_high_from"] == 100.0

    low = predictor.predict([{"date": "2026-09-01", "value": 5}], "c")
    assert low["customer_value"] == "low"
    assert any("ticket médio baixo" in reason for reason in low["reasons"])
