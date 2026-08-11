import pytest
from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_happy_path():
    response = client.post("/predict", json={"text": "This product is sweet die and on point"})
    assert response.status_code == 200
    data = response.json()
    assert set(data.keys()) >= {"label", "confidence", "compound_score"}
    assert data["label"] in {"positive", "negative", "neutral"}
    assert 0.0 <= data["confidence"] <= 1.0
    assert -1.0 <= data["compound_score"] <= 1.0


def test_empty_input_returns_400():
    response = client.post("/predict", json={"text": "   "})
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_predict_batch():
    payload = {"texts": ["This product is good", "This product is bad", "Maybe"]}
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data["results"], list)
    assert len(data["results"]) == 3


@pytest.mark.parametrize("bad_payload", [{"text": None}, {"text": ""}, {}])
def test_invalid_payloads(bad_payload):
    response = client.post("/predict", json=bad_payload)
    assert response.status_code in {400, 422}
