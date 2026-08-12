"""Sentiment inference with marker-based general/Pidgin model routing."""

import logging
import os
from typing import Any, Callable, Optional

import torch
from transformers import pipeline

from preprocessing import (
    clean_text,
    compute_english_lexicon_boost,
    compute_pidgin_boost,
    is_pidgin_leaning,
    preserve_emphasis,
)
from schemas import SentimentResult

logger = logging.getLogger("sentiment_model")

DEFAULT_GENERAL_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
DEFAULT_PIDGIN_MODEL = "Davlan/naija-twitter-sentiment-afriberta-large"
PIDGIN_ID_TO_LABEL = {"0": "positive", "1": "neutral", "2": "negative"}


def _clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


class SentimentModel:
    """Route Pidgin-marked text to the Pidgin model and all other text to general."""

    def __init__(
        self,
        general_model_name: str = DEFAULT_GENERAL_MODEL,
        pidgin_model_name: str = DEFAULT_PIDGIN_MODEL,
        pipeline_factory: Callable[..., Any] = pipeline,
    ) -> None:
        self.general_model_name = general_model_name
        self.pidgin_model_name = pidgin_model_name
        self._pipeline_factory = pipeline_factory
        self.general_model: Optional[Any] = None
        self.pidgin_model: Optional[Any] = None
        self._general_attempted = False
        self._pidgin_attempted = False
        self.mode = "unloaded"

    @property
    def model_name(self) -> str:
        """Backward-compatible health field describing configured models."""
        return f"{self.general_model_name}; {self.pidgin_model_name}"

    def load(self) -> None:
        """Preload each model independently; either one may remain usable."""
        self._load_general_model()
        self._load_pidgin_model()
        self._update_mode()

    def _load_pipeline(self, model_name: str, model_kind: str) -> Optional[Any]:
        try:
            device = 0 if torch.cuda.is_available() else -1
            loaded = self._pipeline_factory(
                "sentiment-analysis", model=model_name, tokenizer=model_name,
                device=device, top_k=None, function_to_apply="softmax",
            )
            logger.info("%s sentiment model loaded: %s (device=%s)", model_kind, model_name, device)
            return loaded
        except Exception:
            logger.exception("Failed to load %s sentiment model '%s'", model_kind, model_name)
            return None

    def _load_general_model(self) -> Optional[Any]:
        if not self._general_attempted:
            self._general_attempted = True
            name = os.getenv("SENTIMENT_GENERAL_MODEL", self.general_model_name)
            self.general_model = self._load_pipeline(name, "general")
        return self.general_model

    def _load_pidgin_model(self) -> Optional[Any]:
        if not self._pidgin_attempted:
            self._pidgin_attempted = True
            name = os.getenv("SENTIMENT_PIDGIN_MODEL", self.pidgin_model_name)
            self.pidgin_model = self._load_pipeline(name, "Pidgin")
        return self.pidgin_model

    def _update_mode(self) -> None:
        if self.general_model is not None and self.pidgin_model is not None:
            self.mode = "both_loaded"
        elif self.general_model is not None:
            self.mode = "general_only"
        elif self.pidgin_model is not None:
            self.mode = "pidgin_only"
        elif self._general_attempted and self._pidgin_attempted:
            self.mode = "lexicon_fallback"
        else:
            self.mode = "unloaded"

    def _normalize_label(
        self, label: str, classifier: Optional[Any] = None, model_used: str = "general"
    ) -> str:
        """Map native labels (including LABEL_0) to the API's three labels."""
        raw = str(label).lower().strip()
        if "pos" in raw:
            return "positive"
        if "neg" in raw:
            return "negative"
        if "neu" in raw:
            return "neutral"

        # Applied based on the model card's documented usage example — confirm
        # against the actual model.config.id2label output once the diagnostic
        # script has been run, and adjust if they don't match.
        if model_used == "pidgin" and (raw.startswith("label_") or raw.isdigit()):
            index = raw.rsplit("_", 1)[-1]
            return PIDGIN_ID_TO_LABEL.get(index, "neutral")

        # General-model pipelines sometimes return LABEL_n. Consult that
        # model's config because its label indices are not universal.
        if classifier is not None and raw.startswith("label_"):
            index = raw.rsplit("_", 1)[-1]
            id2label = getattr(getattr(classifier, "model", None), "config", None)
            id2label = getattr(id2label, "id2label", {}) or {}
            configured = id2label.get(index, id2label.get(int(index), ""))
            if configured and str(configured).lower() != raw:
                return self._normalize_label(str(configured), model_used=model_used)
        return "neutral"

    def _best_prediction(self, classifier: Any, text: str, model_used: str) -> dict[str, float | str]:
        raw_output = classifier(text, truncation=True, max_length=512)
        # top_k=None gives a list of label dictionaries for one input. Be
        # defensive about pipeline-version output nesting.
        if isinstance(raw_output, list) and raw_output and isinstance(raw_output[0], list):
            raw_output = raw_output[0]
        scored = raw_output if isinstance(raw_output, list) else [raw_output]
        best = max(
            (
                {"label": self._normalize_label(item.get("label", "neutral"), classifier, model_used),
                 "score": float(item.get("score", 0.0))}
                for item in scored
                if isinstance(item, dict)
            ),
            key=lambda item: float(item["score"]),
            default={"label": "neutral", "score": 0.0},
        )
        return best

    def _lexicon_only_label(self, pidgin_boost: float, lexicon_boost: float) -> tuple[str, float]:
        combined = pidgin_boost + lexicon_boost
        if combined >= 0.3:
            return "positive", combined
        if combined <= -0.3:
            return "negative", combined
        return "neutral", combined

    def _fallback_predict(self, cleaned: str) -> SentimentResult:
        """Last resort only: lexicon scoring when neither transformer loaded."""
        pidgin_boost = compute_pidgin_boost(cleaned)
        english_boost = compute_english_lexicon_boost(cleaned)
        label, combined = self._lexicon_only_label(pidgin_boost, english_boost)
        if label != "neutral":
            combined += preserve_emphasis(cleaned) * 0.3
        confidence = min(0.85, max(0.15, 0.4 + abs(combined) * 0.4))
        return SentimentResult(
            label=label, confidence=float(confidence), compound_score=float(_clamp(combined)),
            model_used="lexicon_fallback",
        )

    def predict(self, text: str) -> SentimentResult:
        cleaned = clean_text(text)
        if not cleaned:
            raise ValueError("text cannot be empty or whitespace-only")

        pidgin_leaning = is_pidgin_leaning(cleaned)
        general = self._load_general_model()
        pidgin = self._load_pidgin_model()
        self._update_mode()

        # Make one inference call: the marker router selects the preferred
        # model, then uses the other loaded model only if the preferred one is
        # unavailable.
        if pidgin_leaning and pidgin is not None:
            model_used, classifier = "pidgin", pidgin
        elif not pidgin_leaning and general is not None:
            model_used, classifier = "general", general
        elif general is not None:
            logger.warning("Pidgin route selected but Pidgin model is unavailable; using general model.")
            model_used, classifier = "general", general
        elif pidgin is not None:
            logger.warning("General route selected but general model is unavailable; using Pidgin model.")
            model_used, classifier = "pidgin", pidgin
        else:
            return self._fallback_predict(cleaned)

        best = self._best_prediction(classifier, cleaned, model_used)
        label = str(best["label"])
        confidence = float(best["score"])
        return SentimentResult(
            label=label, confidence=confidence,
            compound_score=float(self._score_to_compound(label, confidence)), model_used=model_used,
        )

    @staticmethod
    def _score_to_compound(label: str, score: float) -> float:
        return score if label == "positive" else -score if label == "negative" else 0.0


model = SentimentModel()


def predict(text: str) -> SentimentResult:
    return model.predict(text)


def get_status() -> dict:
    return {"mode": model.mode, "model_name": model.model_name}
