"""
Evaluate weekly forecasts against a holdout tail.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from forecasting import build_forecast
from schemas import ForecastRequest, WeeklyDataPoint


def load_history(path: Path) -> list[WeeklyDataPoint]:
    raw = json.loads(path.read_text())
    if isinstance(raw, dict):
        raw = raw.get("history", raw)
    return [WeeklyDataPoint.model_validate(item) for item in raw]


def mae(actual: list[float], predicted: list[float]) -> float:
    return sum(abs(observed - forecast) for observed, forecast in zip(actual, predicted)) / len(actual)


def rmse(actual: list[float], predicted: list[float]) -> float:
    return math.sqrt(
        sum((observed - forecast) ** 2 for observed, forecast in zip(actual, predicted)) / len(actual)
    )


def evaluate_series(name: str, actual: list[float], predicted: list[float]) -> str:
    return f"{name}: MAE={mae(actual, predicted):.4f}, RMSE={rmse(actual, predicted):.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate weekly forecast accuracy on a held-out tail.")
    parser.add_argument("--input", required=True, type=Path, help="Path to a JSON file containing weekly history.")
    parser.add_argument("--holdout-weeks", type=int, default=4, help="Number of trailing weeks to hold out.")
    args = parser.parse_args()

    history = load_history(args.input)
    if len(history) <= args.holdout_weeks:
        raise SystemExit("history must contain more weeks than the holdout horizon")

    train_history = history[:-args.holdout_weeks]
    holdout_history = history[-args.holdout_weeks:]
    response = build_forecast(ForecastRequest(history=train_history, horizon_weeks=args.holdout_weeks))
    if response.insufficient_data:
        raise SystemExit("training history is shorter than the minimum required forecast window")

    print(f"Held out {args.holdout_weeks} weeks from {len(history)} total weeks")
    print(evaluate_series(
        "sentiment",
        [point.avg_compound_score for point in holdout_history],
        response.sentiment.projected_values if response.sentiment else [],
    ))

    if all(point.avg_rating is not None for point in holdout_history) and response.rating is not None:
        print(evaluate_series(
            "rating",
            [float(point.avg_rating) for point in holdout_history],
            response.rating.projected_values,
        ))
    else:
        print("rating: skipped (missing values in the holdout or training history)")

    print(evaluate_series(
        "volume",
        [float(point.review_count) for point in holdout_history],
        response.volume.projected_values if response.volume else [],
    ))


if __name__ == "__main__":
    main()