from datetime import date, timedelta

from forecasting import MIN_WEEKS_FOR_FORECAST, build_forecast, forecast_series
from schemas import ForecastRequest, WeeklyDataPoint


def build_history(compound_values, rating_values=None, volume_values=None):
    rating_values = rating_values if rating_values is not None else [3.5] * len(compound_values)
    volume_values = volume_values if volume_values is not None else [10] * len(compound_values)
    history = []
    for index, compound in enumerate(compound_values):
        history.append(
            WeeklyDataPoint(
                period_start=date(2026, 1, 5) + timedelta(weeks=index),
                avg_compound_score=compound,
                avg_rating=rating_values[index],
                review_count=volume_values[index],
            )
        )
    return history


def test_forecast_series_classifies_clear_trends():
    horizon = 4
    last_period_start = date(2026, 3, 2)

    increasing = forecast_series([0.1 * i for i in range(8)], horizon, last_period_start)
    decreasing = forecast_series([1.0 - 0.1 * i for i in range(8)], horizon, last_period_start)
    stable = forecast_series([0.25] * 8, horizon, last_period_start)

    assert increasing.trend_direction == "increasing"
    assert decreasing.trend_direction == "decreasing"
    assert stable.trend_direction == "stable"


def test_minimum_data_guard_returns_insufficient_data():
    payload = ForecastRequest(
        history=build_history([0.1 * i for i in range(MIN_WEEKS_FOR_FORECAST - 1)]),
        horizon_weeks=4,
    )

    response = build_forecast(payload)

    assert response.insufficient_data is True
    assert response.weeks_of_history == MIN_WEEKS_FOR_FORECAST - 1
    assert response.minimum_weeks_required == MIN_WEEKS_FOR_FORECAST
    assert response.sentiment is None
    assert response.rating is None
    assert response.volume is None


def test_null_rating_skips_rating_forecast_but_keeps_other_series():
    history = []
    for index in range(MIN_WEEKS_FOR_FORECAST):
        history.append(
            WeeklyDataPoint(
                period_start=date(2026, 1, 5) + timedelta(weeks=index),
                avg_compound_score=0.05 * index,
                avg_rating=None if index == 3 else 3.0 + 0.1 * index,
                review_count=20 + index,
            )
        )

    response = build_forecast(ForecastRequest(history=history, horizon_weeks=4))

    assert response.insufficient_data is False
    assert response.sentiment is not None
    assert response.volume is not None
    assert response.rating is None