# NaijaSenti evaluation

`data/` contains a reproducible 300-item sample from the `pcm` test split of
`HausaNLP/NaijaSenti-Twitter` (row offsets 0–299). The source dataset uses the
CC-BY-NC-SA-4.0 licence; retain its attribution and licence if redistributing
the sample.

Start the service, then run `python evaluation/evaluate_naijasenti.py`. It
reports overall accuracy, precision/recall/F1 by class, and accuracy grouped by
the winning `model_used`, writing the same report to `evaluation/results/`.

The sample is a held-out evaluation fixture, never a training input. Pidgin
sentiment has documented annotation ambiguity, so report results with that
limitation rather than treating a single score as definitive.
