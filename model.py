"""
model.py — replaces FYP_NLP/model.py

Changes from the original:
1. Model load failures are now logged (previously silently swallowed via
   `except Exception: self.model = False`), and the active mode
   ("transformer" vs "lexicon_fallback") is tracked and exposed so you can
   verify at a glance which engine actually served a given response.
2. Removed the hard "<=2 words -> always neutral" rule, which was
   overriding your own Pidgin lexicon for single-word reviews (e.g.
   "wahala" never reached compute_pidgin_boost before). Short text now
   uses the lexicon signal directly instead of being discarded.
3. `confidence` and `compound_score` are no longer the same number wearing
   two names. `confidence` reflects the transformer's own classification
   probability; `compound_score` is a separate intensity measure that
   blends the model output with lexicon/emphasis signal.
4. Added `truncation=True, max_length=512` to the pipeline call so long
   reviews don't error out or behave unpredictably.
5. Added an explicit `load()` method so the model can be preloaded at
   FastAPI startup instead of on the first incoming request (avoids a
   slow, demo-unfriendly cold start on the first real call).
6. Fallback predictor now also uses the English lexicon, not just Pidgin +
   emphasis, so it isn't blind to plain English reviews when it's active.
"""

import logging
import os

import torch
from transformers import pipeline

from preprocessing import (
    clean_text,
    compute_english_lexicon_boost,
    compute_pidgin_boost,
    preserve_emphasis,
)
from schemas import SentimentResult

logger = logging.getLogger("sentiment_model")

DEFAULT_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"


def _clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


class SentimentModel:
    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self.model_name = model_name
        self.model = None
        self.mode = "unloaded"  # "transformer" | "lexicon_fallback" | "unloaded"

    def load(self) -> None:
        """Explicit preload — call this at app startup, not lazily on first request."""
        self._load_model()

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
            self.mode = "transformer"
            logger.info("Sentiment model loaded: %s (device=%s)", model_name, device)
        except Exception:
            self.model = False
            self.mode = "lexicon_fallback"
            logger.exception(
                "Failed to load sentiment transformer '%s' — falling back to "
                "lexicon-only scoring. Predictions from this point on are NOT "
                "using the transformer model.",
                model_name,
            )
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
            return score
        if label == "negative":
            return -score
        return 0.0

    def _lexicon_only_label(self, pidgin_boost: float, lexicon_boost: float):
        combined = pidgin_boost + lexicon_boost
        if combined > 0.3:
            return "positive", combined
        if combined < -0.3:
            return "negative", combined
        return "neutral", combined

    def _fallback_predict(
        self, cleaned: str, pidgin_boost: float, emphasis_boost: float, lexicon_boost: float
    ) -> SentimentResult:
        """Used when the transformer failed to load. Lexicon + emphasis only."""
        label, combined = self._lexicon_only_label(pidgin_boost, lexicon_boost)
        combined += emphasis_boost * 0.3 if label != "neutral" else 0.0
        confidence = min(0.85, max(0.15, 0.4 + abs(combined) * 0.4))
        compound = _clamp(combined)
        return SentimentResult(label=label, confidence=float(confidence), compound_score=float(compound))

    def predict(self, text: str) -> SentimentResult:
        cleaned = clean_text(text)
        if not cleaned:
            raise ValueError("text cannot be empty or whitespace-only")

        pidgin_boost = compute_pidgin_boost(cleaned)
        lexicon_boost = compute_english_lexicon_boost(cleaned)
        emphasis_boost = preserve_emphasis(cleaned)
        word_count = len(cleaned.split())

        # Very short text: transformer context is too thin to trust on its
        # own, and running it through the model wastes a call for a case
        # the lexicon already answers directly (e.g. "wahala", "sabi").
        if word_count <= 2:
            label, combined = self._lexicon_only_label(pidgin_boost, lexicon_boost)
            confidence = 0.4 if label != "neutral" else 0.3
            compound = _clamp(combined)
            return SentimentResult(label=label, confidence=confidence, compound_score=float(compound))

        model = self._load_model()
        if model is False:
            return self._fallback_predict(cleaned, pidgin_boost, emphasis_boost, lexicon_boost)

        raw_output = model(cleaned, truncation=True, max_length=512)[0]
        scored = raw_output if isinstance(raw_output, list) else [raw_output]

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
        # `confidence` is the transformer's own classification probability —
        # left un-blended with lexicon signal so it means one consistent thing.
        confidence = float(best["score"])

        # If the model is unsure AND the lexicon strongly disagrees with it,
        # let the lexicon override the label. A confident model call is
        # trusted over the lexicon; an unsure one isn't.
        lexical_signal = pidgin_boost + lexicon_boost
        if confidence < 0.55:
            if lexical_signal > 0.5:
                label = "positive"
            elif lexical_signal < -0.5:
                label = "negative"

        # `compound_score` is a separate intensity measure: starts from the
        # model's own label+probability, then nudged by lexicon/emphasis
        # signal rather than reusing `confidence` as its magnitude.
        compound = self._score_to_compound(label, confidence)
        compound += _clamp(pidgin_boost) * 0.2 + _clamp(lexicon_boost) * 0.1 + emphasis_boost * 0.1
        compound = _clamp(compound)

        return SentimentResult(label=label, confidence=confidence, compound_score=float(compound))


model = SentimentModel()


def predict(text: str) -> SentimentResult:
    return model.predict(text)


def get_status() -> dict:
    """Used by /health to report which engine is actually active."""
    return {"mode": model.mode, "model_name": model.model_name}
