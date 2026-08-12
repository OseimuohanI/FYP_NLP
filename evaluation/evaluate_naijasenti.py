"""Evaluate the API against the checked-in NaijaSenti Pidgin test sample.

Usage: python evaluation/evaluate_naijasenti.py
Set SENTIMENT_API_URL to point at a deployed service; it defaults to the local
FastAPI server. Results are written to evaluation/results/ (gitignored).
"""

import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
DATA_FILES = sorted((ROOT / "data").glob("naijasenti_pcm_test_*.json"))
API_URL = os.getenv("SENTIMENT_API_URL", "http://127.0.0.1:8000/predict/batch")
BATCH_SIZE = 50
LABELS = ("positive", "neutral", "negative")
DATASET_LABELS = {0: "positive", 1: "neutral", 2: "negative"}


def load_examples() -> list[dict]:
    examples = []
    for file in DATA_FILES:
        payload = json.loads(file.read_text(encoding="utf-8"))
        for item in payload["rows"]:
            row = item["row"]
            examples.append({"text": row["tweet"], "label": DATASET_LABELS[row["label"]]})
    if not examples:
        raise FileNotFoundError("No local NaijaSenti PCM sample files found in evaluation/data")
    return examples


def predict_batch(texts: list[str]) -> list[dict]:
    request = Request(
        API_URL, data=json.dumps({"texts": texts}).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urlopen(request, timeout=120) as response:  # nosec B310 - endpoint is operator configured
        return json.loads(response.read().decode("utf-8"))["results"]


def class_metrics(expected: list[str], predicted: list[str]) -> dict:
    metrics = {}
    for label in LABELS:
        tp = sum(e == label and p == label for e, p in zip(expected, predicted))
        fp = sum(e != label and p == label for e, p in zip(expected, predicted))
        fn = sum(e == label and p != label for e, p in zip(expected, predicted))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        metrics[label] = {
            "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "support": sum(e == label for e in expected),
        }
    return metrics


def main() -> None:
    examples = load_examples()
    results = []
    for start in range(0, len(examples), BATCH_SIZE):
        results.extend(predict_batch([item["text"] for item in examples[start:start + BATCH_SIZE]]))

    expected = [item["label"] for item in examples]
    predicted = [item["label"] for item in results]
    by_model = defaultdict(lambda: {"total": 0, "correct": 0})
    for truth, result in zip(expected, results):
        bucket = by_model[result["model_used"]]
        bucket["total"] += 1
        bucket["correct"] += result["label"] == truth
    for bucket in by_model.values():
        bucket["accuracy"] = bucket["correct"] / bucket["total"] if bucket["total"] else 0.0

    report = {
        "dataset": "HausaNLP/NaijaSenti-Twitter", "config": "pcm", "split": "test",
        "sample_size": len(examples), "api_url": API_URL,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "accuracy": sum(e == p for e, p in zip(expected, predicted)) / len(expected),
        "per_class": class_metrics(expected, predicted),
        "by_model_used": dict(sorted(by_model.items())),
        "prediction_counts": dict(Counter(predicted)),
    }
    output_dir = ROOT / "results"
    output_dir.mkdir(exist_ok=True)
    output = output_dir / "naijasenti_pcm_evaluation.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved report: {output}")


if __name__ == "__main__":
    main()
