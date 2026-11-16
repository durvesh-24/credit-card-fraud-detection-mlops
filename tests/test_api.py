"""API tests. They use a small local fixture model (see conftest.py) — no MLflow server."""

import json

import pytest
from fastapi.testclient import TestClient

from src.api import app

VALID = {"Time": 3600.0, "Amount": 42.5, "V": [0.1] * 28}


@pytest.fixture
def client(monkeypatch, tmp_path, fixture_model_dir):
    monkeypatch.setenv("MODEL_URI", fixture_model_dir)
    monkeypatch.setenv("FRAUD_THRESHOLD", "0.5")
    monkeypatch.setenv("PREDICTION_LOG_DIR", str(tmp_path / "logs"))
    with TestClient(app) as c:  # `with` runs the lifespan -> model is loaded
        yield c


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok" and r.json()["model_loaded"] is True


def test_predict_valid(client):
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert body["is_fraud"] == (body["fraud_probability"] >= 0.5)
    assert body["threshold"] == 0.5


def test_fraud_like_transaction_scores_higher(client):
    fraud_v = [0.0] * 28
    fraud_v[13], fraud_v[11] = -6.0, -5.0  # V14, V12 strongly negative like fraud in the fixture
    legit = client.post("/predict", json={**VALID, "V": [0.0] * 28}).json()["fraud_probability"]
    fraud = client.post("/predict", json={**VALID, "V": fraud_v}).json()["fraud_probability"]
    assert fraud > legit


def test_prediction_is_logged(client, tmp_path):
    client.post("/predict", json=VALID)
    files = list((tmp_path / "logs").glob("predictions-*.jsonl"))
    assert len(files) == 1
    record = json.loads(files[0].read_text().strip().splitlines()[-1])
    assert record["Amount"] == 42.5 and record["V1"] == 0.1 and record["V28"] == 0.1
    assert {"ts", "fraud_probability", "is_fraud", "model_version"} <= record.keys()


@pytest.mark.parametrize(
    "payload",
    [
        {**VALID, "V": [0.1] * 27},  # too few V values
        {**VALID, "V": [0.1] * 29},  # too many
        {**VALID, "Amount": -1.0},  # negative amount
        {**VALID, "Time": -5.0},  # negative time
        {"Time": 1.0, "V": [0.1] * 28},  # missing Amount
        {**VALID, "V": ["a"] * 28},  # wrong type
        {**VALID, "extra": 1},  # unknown field
    ],
    ids=["27_v", "29_v", "neg_amount", "neg_time", "missing_amount", "bad_type", "extra_field"],
)
def test_bad_input_returns_422(client, payload):
    assert client.post("/predict", json=payload).status_code == 422


def test_metrics_exposed_and_flagged_counter(monkeypatch, client):
    monkeypatch.setenv("FRAUD_THRESHOLD", "0.0")  # every transaction gets flagged
    client.post("/reload")

    def flagged() -> float:
        for line in client.get("/metrics").text.splitlines():
            if line.startswith("fraud_flagged_transactions_total "):
                return float(line.split()[-1])
        return 0.0

    before = flagged()
    assert client.post("/predict", json=VALID).json()["is_fraud"] is True
    assert flagged() == before + 1
    text = client.get("/metrics").text
    assert "http_requests_total" in text and "fraud_predictions_total" in text


def test_no_model_returns_503(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_URI", str(tmp_path / "does-not-exist"))
    monkeypatch.setenv("PREDICTION_LOG_DIR", str(tmp_path / "logs"))
    with TestClient(app) as c:
        assert c.get("/health").status_code == 503
        assert c.post("/predict", json=VALID).status_code == 503
