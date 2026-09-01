"""
Weekly trend forecasting for business review history.
"""

from __future__ import annotations

import logging
import warnings
from datetime import date, timedelta
from typing import Iterable

import pandas as pd
# statsmodels can emit convergence warnings on short or unusual series; log
# them rather than silencing them so callers can judge forecast confidence.
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from schemas import ForecastRequest, ForecastResponse, SeriesForecast

logger = logging.getLogger("forecasting")

MIN_WEEKS_FOR_FORECAST = 8
# Tunable dead zone for classifying a fitted trend as flat rather than noisy.
TREND_DIRECTION_DEAD_ZONE = 0.01


def _project_period_starts(last_period_start: date, horizon: int) -> list[date]:
    return [last_period_start + timedelta(weeks=offset) for offset in range(1, horizon + 1)]


def _classify_trend(trend_value: float, latest_level: float) -> str:
    scale = max(abs(latest_level), 1.0)
    normalized_trend = trend_value / scale
    if normalized_trend > TREND_DIRECTION_DEAD_ZONE:
        return "increasing"
    if normalized_trend < -TREND_DIRECTION_DEAD_ZONE:
        return "decreasing"
    return "stable"


def _fit_forecast(history: list[float], horizon: int) -> tuple[list[float], float]:
    series = pd.Series(history, dtype="float64")
    model = ExponentialSmoothing(series, trend="add", seasonal=None)

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        fit = model.fit(optimized=True)

    for caught_warning in caught_warnings:
        logger.warning("Forecast fit warning: %s", caught_warning.message)

    projected_values = [float(value) for value in fit.forecast(horizon)]
    trend_state = getattr(fit, "trend", None)
    if trend_state is not None and len(trend_state):
        trend_value = float(trend_state.iloc[-1])
    elif horizon > 1:
        trend_value = float(projected_values[1] - projected_values[0])
    else:
        trend_value = 0.0

    return projected_values, trend_value


def forecast_series(history: list[float], horizon: int, last_period_start: date) -> SeriesForecast:
	if len(history) < 2:
		raise ValueError("history must contain at least two weekly values")

	projected_values, trend_value = _fit_forecast(history, horizon)
	latest_level = float(history[-1])
	return SeriesForecast(
		trend_direction=_classify_trend(trend_value, latest_level),
		projected_values=projected_values,
		projected_period_starts=_project_period_starts(last_period_start, horizon),
	)


def build_forecast(payload: ForecastRequest) -> ForecastResponse:
	weeks_of_history = len(payload.history)
	if weeks_of_history < MIN_WEEKS_FOR_FORECAST:
		return ForecastResponse(
			insufficient_data=True,
			weeks_of_history=weeks_of_history,
			minimum_weeks_required=MIN_WEEKS_FOR_FORECAST,
		)

	last_period_start = payload.history[-1].period_start
	sentiment_history = [point.avg_compound_score for point in payload.history]
	volume_history = [float(point.review_count) for point in payload.history]
	rating_history = [point.avg_rating for point in payload.history]
	rating_available = all(value is not None for value in rating_history)

	sentiment_forecast = forecast_series(sentiment_history, payload.horizon_weeks, last_period_start)
	volume_forecast = forecast_series(volume_history, payload.horizon_weeks, last_period_start)
	rating_forecast = (
		forecast_series([float(value) for value in rating_history if value is not None], payload.horizon_weeks, last_period_start)
		if rating_available
		else None
	)

	return ForecastResponse(
		insufficient_data=False,
		weeks_of_history=weeks_of_history,
		minimum_weeks_required=MIN_WEEKS_FOR_FORECAST,
		sentiment=sentiment_forecast,
		rating=rating_forecast,
		volume=volume_forecast,
	)