"""Batch-level sentiment aggregation and optional OpenRouter dashboard summary."""

import logging
import os
from collections import Counter
from typing import Sequence

import httpx

from schemas import BatchSummary, SentimentResult

logger = logging.getLogger(__name__)

OPENROUTER_CHAT_COMPLETIONS_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_OPENROUTER_MODEL = "openrouter/free"
OPENROUTER_TIMEOUT_SECONDS = 12.0
# Starting estimate from observed Pidgin-model over-confidence. Tune this on the
# NaijaSenti held-out validation set if a more rigorous calibration is needed.
PIDGIN_CONFIDENCE_CALIBRATION_FACTOR = 0.85


def _calibrated_confidence(result: SentimentResult) -> float:
    confidence = result.confidence
    if result.model_used == "pidgin":
        confidence *= PIDGIN_CONFIDENCE_CALIBRATION_FACTOR
    return max(0.0, min(1.0, confidence))


def _fallback_summary(label_counts: Counter[str], average_confidence: float) -> str:
    total = sum(label_counts.values())
    dominant_label = max(("positive", "negative", "neutral"), key=lambda label: label_counts[label])
    positive_pct = round((label_counts["positive"] / total) * 100) if total else 0
    return (
        f"Mostly {dominant_label} sentiment ({positive_pct}% positive), "
        f"average confidence {average_confidence:.2f}."
    )


def _representative_examples(texts: Sequence[str], results: Sequence[SentimentResult]) -> list[str]:
    examples = []
    for label in ("positive", "negative"):
        candidates = (
            (index, result)
            for index, result in enumerate(results)
            if result.label == label
        )
        candidate = max(candidates, key=lambda item: _calibrated_confidence(item[1]), default=None)
        if candidate is not None:
            index, _ = candidate
            examples.append(f"{label}: {texts[index]}")
    return examples


def _llm_summary(
    *, label_counts: Counter[str], rating: float, texts: Sequence[str], results: Sequence[SentimentResult]
) -> str | None:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        logger.warning("LLM batch summary unavailable: OPENROUTER_API_KEY is not configured.")
        return None

    total = len(results)
    counts = ", ".join(f"{label_counts[label]} {label}" for label in ("positive", "negative", "neutral"))
    examples = _representative_examples(texts, results)
    prompt = (
        "Write one plain-language sentence (no markdown) summarizing overall customer sentiment "
        "for a dashboard reader. Do not merely repeat the numbers.\n"
        f"Review count and labels: {counts} out of {total} reviews.\n"
        f"Aggregate rating: {rating:.1f}/5.\n"
        f"Representative reviews:\n" + "\n".join(examples)
    )
    payload = {
        "model": os.getenv("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL),
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 60,
    }
    try:
        response = httpx.post(
            OPENROUTER_CHAT_COMPLETIONS_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=OPENROUTER_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        if not isinstance(content, str) or not content.strip():
            raise ValueError("response contained no summary text")
        return " ".join(content.split())
    except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
        logger.warning("LLM batch summary unavailable: %s", exc)
        return None


def build_batch_summary(texts: list[str], results: list[SentimentResult]) -> BatchSummary:
    """Build the complete batch aggregate without changing per-item results."""
    if len(texts) != len(results):
        raise ValueError("texts and results must have the same length")
    if not results:
        raise ValueError("cannot build a summary for an empty batch")

    average_compound_score = sum(result.compound_score for result in results) / len(results)
    rating = round(((average_compound_score + 1) / 2) * 4 + 1, 1)
    calibrated_confidences = [_calibrated_confidence(result) for result in results]
    average_confidence = round(max(0.0, min(1.0, sum(calibrated_confidences) / len(results))), 2)
    label_counts: Counter[str] = Counter(result.label for result in results)
    summary_text = _llm_summary(
        label_counts=label_counts, rating=rating, texts=texts, results=results
    ) or _fallback_summary(label_counts, average_confidence)
    return BatchSummary(
        rating=rating, average_confidence=average_confidence, summary_text=summary_text
    )
