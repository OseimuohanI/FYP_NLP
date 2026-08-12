import httpx

import aggregation
from aggregation import build_batch_summary
from schemas import SentimentResult


def result(label, confidence, compound_score, model_used="general"):
    return SentimentResult(
        label=label,
        confidence=confidence,
        compound_score=compound_score,
        model_used=model_used,
    )


def test_rating_maps_compound_score_to_one_to_five(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    for score, expected_rating in ((1.0, 5.0), (-1.0, 1.0), (0.0, 3.0)):
        summary = build_batch_summary(["review"], [result("neutral", 0.5, score)])
        assert summary.rating == expected_rating


def test_pidgin_confidence_is_calibrated_without_mutating_results(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    item = result("positive", 1.0, 1.0, "pidgin")
    summary = build_batch_summary(["E sweet die"], [item])
    assert summary.average_confidence == 0.85
    assert item.confidence == 1.0


def test_openrouter_response_is_used_when_available(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    captured = {}

    def fake_post(url, **kwargs):
        captured.update(url=url, **kwargs)
        request = httpx.Request("POST", url)
        return httpx.Response(200, json={"choices": [{"message": {"content": "Customers are happy overall."}}]}, request=request)

    monkeypatch.setattr(aggregation.httpx, "post", fake_post)
    summary = build_batch_summary(
        ["Excellent item", "Delivery was terrible"],
        [result("positive", 0.8, 0.8), result("negative", 0.6, -0.6)],
    )
    assert summary.summary_text == "Customers are happy overall."
    assert captured["url"] == aggregation.OPENROUTER_CHAT_COMPLETIONS_URL
    assert captured["json"]["model"] == aggregation.DEFAULT_OPENROUTER_MODEL
    assert captured["timeout"] == aggregation.OPENROUTER_TIMEOUT_SECONDS
    assert "Excellent item" in captured["json"]["messages"][0]["content"]


def test_missing_key_and_failed_request_use_local_fallback(monkeypatch):
    entries = [result("positive", 0.8, 0.8)]
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    missing_key = build_batch_summary(["Great product"], entries)
    assert missing_key.summary_text == "Mostly positive sentiment (100% positive), average confidence 0.80."

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    def failing_post(*args, **kwargs):
        raise httpx.ConnectError("network unavailable")

    monkeypatch.setattr(aggregation.httpx, "post", failing_post)
    failed_request = build_batch_summary(["Great product"], entries)
    assert failed_request.summary_text == missing_key.summary_text
