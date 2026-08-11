"""
schemas.py — replaces FYP_NLP/schemas.py

Changes from the original:
1. BatchPredictRequest.texts now has a max_length cap (default 100). The
   original had no upper bound, so Laravel sending a very large batch in
   one synchronous call could block a request for a long time with no
   queue behind it. If you need larger batches, chunk them on the Laravel
   side rather than raising this limit too far.
2. HealthResponse now includes `mode`, so /health can tell you whether the
   transformer or the lexicon fallback is actually serving predictions —
   this pairs with the logging added in model.py.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

MAX_BATCH_SIZE = 100


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1)

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("text cannot be empty or whitespace-only")
        return value.strip()


class BatchPredictRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=MAX_BATCH_SIZE)

    @field_validator("texts")
    @classmethod
    def texts_must_not_be_blank(cls, value: List[str]) -> List[str]:
        cleaned = []
        for item in value:
            if item is None or not str(item).strip():
                raise ValueError("texts cannot contain empty or whitespace-only entries")
            cleaned.append(str(item).strip())
        return cleaned


class SentimentResult(BaseModel):
    label: Literal["positive", "negative", "neutral"]
    confidence: float = Field(ge=0.0, le=1.0)
    compound_score: float = Field(ge=-1.0, le=1.0)


class BatchPredictResponse(BaseModel):
    results: List[SentimentResult]


class HealthResponse(BaseModel):
    status: str
    model: Optional[str] = None
    mode: Optional[str] = None  # "transformer" | "lexicon_fallback" | "unloaded"
