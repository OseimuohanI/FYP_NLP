import os

import torch
from transformers import pipeline

from preprocessing import clean_text, compute_pidgin_boost, preserve_emphasis
from schemas import SentimentResult


DEFAULT_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"


class SentimentModel:
    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self.model_name = model_name
        self.model = None

    def _load_model(self):
        if self.model is not None:
            return self.model

        model_name = os.getenv("SENTIMENT_MODEL", self.model_name)
        try:
            device = 0 if torch.cuda.is_available() else -1
            self.model = pipeline(
                "sentiment-analysis",
                model=model_name,
                tokenizer=model_name,
                device=device,
                top_k=None,
                function_to_apply="softmax",
            )
        except Exception:
            self.model = False
        return self.model

    def _normalize_label(self, label: str) -> str:
        normalized = str(label).lower().strip()
        if "pos" in normalized:
            return "positive"
        if "neg" in normalized:
            return "negative"
        return "neutral"

    def _score_to_compound(self, label: str, score: float) -> float:
        if label == "positive":
            return 1.0 * score
        if label == "negative":
            return -1.0 * score
        return 0.0

    def _fallback_predict(self, cleaned: str) -> SentimentResult:
        pidgin_boost = compute_pidgin_boost(cleaned)
        emphasis_boost = preserve_emphasis(cleaned)
        score = pidgin_boost * 0.22 + emphasis_boost * 0.18

        if score > 0.4:
            label = "positive"
        elif score < -0.4:
            label = "negative"
        else:
            label = "neutral"

        confidence = min(0.95, max(0.15, 0.5 + abs(score) * 0.5))
        compound = max(-1.0, min(1.0, score))
        return SentimentResult(label=label, confidence=float(confidence), compound_score=float(compound))

    def predict(self, text: str) -> SentimentResult:
        cleaned = clean_text(text)
        if not cleaned:
            raise ValueError("text cannot be empty or whitespace-only")

        if len(cleaned.split()) <= 2:
            return SentimentResult(label="neutral", confidence=0.2, compound_score=0.0)

        model = self._load_model()
        if model is False:
            return self._fallback_predict(cleaned)

        pidgin_boost = compute_pidgin_boost(cleaned)
        emphasis_boost = preserve_emphasis(cleaned)

        raw_output = model(cleaned)[0]

        if isinstance(raw_output, list):
            scored = raw_output
        else:
            scored = [raw_output]

        best = None
        best_score = -1.0
        for item in scored:
            label = self._normalize_label(item.get("label", "neutral"))
            probability = float(item.get("score", 0.0))
            if probability > best_score:
                best = {"label": label, "score": probability}
                best_score = probability

        if best is None:
            best = {"label": "neutral", "score": 0.0}

        label = best["label"]
        confidence = max(0.0, min(1.0, float(best["score"]) + min(max(pidgin_boost, -1.0), 1.0) * 0.08 + emphasis_boost * 0.15))

        if label == "positive" and pidgin_boost < 0:
            label = "neutral" if confidence < 0.5 else label
        elif label == "negative" and pidgin_boost > 0:
            label = "neutral" if confidence < 0.5 else label

        if confidence < 0.15:
            label = "neutral"
            confidence = 0.15

        compound = self._score_to_compound(label, confidence)
        if pidgin_boost != 0:
            compound += min(max(pidgin_boost, -1.0), 1.0) * 0.25
        compound = max(-1.0, min(1.0, compound))

        return SentimentResult(label=label, confidence=float(confidence), compound_score=float(compound))


model = SentimentModel()


def predict(text: str) -> SentimentResult:
    return model.predict(text)
