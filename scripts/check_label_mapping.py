"""Print the Pidgin model's configured label mapping for manual verification.

Run this only after rebuilding the service image:
    python scripts/check_label_mapping.py

The model card's documented usage example expects:
    0: positive, 1: neutral, 2: negative
Compare that mapping with the printed ``model.config.id2label`` and report any
difference before changing the explicit override in ``model.py``.
"""

from transformers import AutoModelForSequenceClassification


PIDGIN_MODEL = "Davlan/naija-twitter-sentiment-afriberta-large"
EXPECTED_ID2LABEL = {0: "positive", 1: "neutral", 2: "negative"}


def main() -> None:
    model = AutoModelForSequenceClassification.from_pretrained(PIDGIN_MODEL)
    print(f"Model: {PIDGIN_MODEL}")
    print(f"Expected model-card mapping: {EXPECTED_ID2LABEL}")
    print(f"Configured model.config.id2label: {model.config.id2label}")


if __name__ == "__main__":
    main()
