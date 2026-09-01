"""
schemas.py
"""

from datetime import date
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

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
    # `model_used` is part of the public response contract, not Pydantic internals.
    model_config = ConfigDict(protected_namespaces=())
    label: Literal["positive", "negative", "neutral"]
    confidence: float = Field(ge=0.0, le=1.0)
    compound_score: float = Field(ge=-1.0, le=1.0)
    model_used: Literal["general", "pidgin", "lexicon_fallback"]


class BatchSummary(BaseModel):
    rating: float = Field(ge=1.0, le=5.0)
    average_confidence: float = Field(ge=0.0, le=1.0)
    summary_text: str


class BatchPredictResponse(BaseModel):
    results: List[SentimentResult]
    summary: BatchSummary


class HealthResponse(BaseModel):
    status: str
    model: Optional[str] = None
    mode: Optional[str] = None  # both_loaded | general_only | pidgin_only | lexicon_fallback | unloaded


class WeeklyDataPoint(BaseModel):
    period_start: date
    avg_compound_score: float
    avg_rating: float | None = None
    review_count: int


class ForecastRequest(BaseModel):
    history: list[WeeklyDataPoint]
    horizon_weeks: int = Field(default=4, ge=1, le=12)


class SeriesForecast(BaseModel):
    trend_direction: Literal["increasing", "decreasing", "stable"]
    projected_values: list[float]
    projected_period_starts: list[date]


class ForecastResponse(BaseModel):
    insufficient_data: bool
    weeks_of_history: int
    minimum_weeks_required: int
    sentiment: SeriesForecast | None = None
    rating: SeriesForecast | None = None
    volume: SeriesForecast | None = None
