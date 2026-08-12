import pytest
from fastapi.testclient import TestClient

from main import app
import model as model_module
from model import SentimentModel


client = TestClient(app)


@pytest.fixture(autouse=True)
def disable_network_models_for_api_tests():
    """Endpoint tests exercise the stable lexicon fallback, never the Hub."""
    live_model = model_module.model
    old_state = (
        live_model.general_model, live_model.pidgin_model,
        live_model._general_attempted, live_model._pidgin_attempted, live_model.mode,
    )
    live_model.general_model = live_model.pidgin_model = None
    live_model._general_attempted = live_model._pidgin_attempted = True
    live_model._update_mode()
    yield
    (
        live_model.general_model, live_model.pidgin_model,
        live_model._general_attempted, live_model._pidgin_attempted, live_model.mode,
    ) = old_state


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_happy_path():
    response = client.post("/predict", json={"text": "This product is sweet die and on point"})
    assert response.status_code == 200
    data = response.json()
    assert set(data.keys()) >= {"label", "confidence", "compound_score", "model_used"}
    assert data["label"] in {"positive", "negative", "neutral"}
    assert 0.0 <= data["confidence"] <= 1.0
    assert -1.0 <= data["compound_score"] <= 1.0
    assert data["model_used"] in {"general", "pidgin", "lexicon_fallback"}


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


class FakeClassifier:
    def __init__(self, scores, id2label=None):
        self.scores = scores
        self.model = type("Model", (), {"config": type("Config", (), {"id2label": id2label or {}})()})()

    def __call__(self, text, **kwargs):
        return self.scores


def ensemble(general, pidgin):
    instance = SentimentModel(pipeline_factory=lambda *args, **kwargs: None)
    instance.general_model, instance.pidgin_model = general, pidgin
    instance._general_attempted = instance._pidgin_attempted = True
    instance._update_mode()
    return instance


def test_threshold_value_is_not_neutral_when_transformers_unavailable():
    instance = ensemble(None, None)
    result = instance.predict("Sabi")
    assert result.label == "positive"
    assert result.model_used == "lexicon_fallback"


def test_pidgin_model_wins_when_it_has_higher_confidence():
    instance = ensemble(
        FakeClassifier([{"label": "negative", "score": 0.60}]),
        FakeClassifier([{"label": "positive", "score": 0.91}]),
    )
    result = instance.predict("This thing mad o, I really love am")
    assert (result.label, result.model_used) == ("positive", "pidgin")


def test_general_model_wins_mixed_pidgin_english_context():
    instance = ensemble(
        FakeClassifier([{"label": "negative", "score": 0.88}]),
        FakeClassifier([{"label": "positive", "score": 0.62}]),
    )
    result = instance.predict("E sweet at first but the delivery was terrible")
    assert (result.label, result.model_used) == ("negative", "general")


def test_context_case_can_choose_pidgin_for_opposite_mad_o_meaning():
    instance = ensemble(
        FakeClassifier([{"label": "positive", "score": 0.61}]),
        FakeClassifier([{"label": "negative", "score": 0.89}]),
    )
    result = instance.predict("This seller mad o, dem spoil everything")
    assert (result.label, result.model_used) == ("negative", "pidgin")


def test_native_label_indices_are_read_from_that_models_config():
    instance = ensemble(
        FakeClassifier([{"label": "LABEL_0", "score": 0.70}], {0: "negative"}),
        None,
    )
    result = instance.predict("ordinary review")
    assert (result.label, result.model_used) == ("negative", "general")
