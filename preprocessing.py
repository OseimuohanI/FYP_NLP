"""
preprocessing.py — replaces FYP_NLP/preprocessing.py

Changes from the original:
1. compute_pidgin_boost now uses word-boundary regex matching instead of
   plain substring `in` checks, and masks matched phrases before checking
   shorter ones so overlapping entries (e.g. "no be" inside "e no be") are
   not double-counted.
2. Added an English sentiment lexicon (compute_english_lexicon_boost) so
   the fallback predictor isn't blind to plain English reviews.
3. Corrected four PIDGIN_LEXICON values using baseline test evidence and
   added the separate "no wahala" phrase. The lexicon remains available only
   for the no-transformer fallback, not as a transformer override.
4. Removed a dead/redundant regex in normalize_elongated_words (the
   generic elongation collapse already handled "goooood" -> "good"
   before the specific pattern ever had a chance to match).
"""

import html
import re
import unicodedata

# Baseline-test evidence corrected wahala, shine your eye, no gree and mad o.
# Context-dependent entries (especially no gree and mad o) are lower-confidence
# signals, not settled ground truth; validate them against real examples. This
# lexicon is fallback-only now that transformer predictions use a dual-model
# confidence ensemble, so it must never override a transformer prediction.
PIDGIN_LEXICON = {
    "no wahala": 0.5,
    "no gree": -0.3,  # lower-confidence correction; validate on real examples
    "wahala": -0.4,
    "sweet die": 0.9,
    "on point": 0.9,
    "shine your eye": -0.4,
    "chop": 0.4,
    "how body": 0.7,
    "sabi": 0.3,
    "abeg": 0.2,
    "make we": 0.1,
    "no be": -0.2,
    "bad market": -0.9,
    "suffering": -0.8,
    "e no be": -0.5,
    "e don spoil": -0.8,
    "mad o": 0.0,  # context-dependent slang; intentionally near-neutral
    "shey": 0.1,
    "naija": 0.2,
}

# Small general-purpose English sentiment lexicon. This exists so the
# fallback predictor (used when the transformer fails to load) and the
# short-text path aren't blind to plain English reviews that contain no
# Pidgin at all. Not exhaustive — extend as you see false negatives during
# testing.
ENGLISH_LEXICON = {
    "good": 0.5, "great": 0.7, "excellent": 0.9, "amazing": 0.9,
    "love": 0.8, "perfect": 0.9, "best": 0.7, "fast": 0.4,
    "recommend": 0.6, "satisfied": 0.6, "happy": 0.6, "quality": 0.4,
    "nice": 0.5, "awesome": 0.8, "smooth": 0.4, "reliable": 0.5,
    "bad": -0.5, "terrible": -0.9, "worst": -0.9, "awful": -0.9,
    "poor": -0.5, "slow": -0.4, "disappointed": -0.7, "disappointing": -0.7,
    "broken": -0.6, "scam": -0.9, "fake": -0.8, "refund": -0.3,
    "waste": -0.7, "horrible": -0.9, "never": -0.3, "avoid": -0.7,
    "delay": -0.4, "delayed": -0.4, "late": -0.3, "damaged": -0.6,
    "faulty": -0.6, "rude": -0.6,
}


def clean_text(text: str) -> str:
    if text is None:
        return ""
    text = html.unescape(text)
    text = text.replace("&nbsp;", " ")
    text = text.strip()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"@[A-Za-z0-9_]+", " ", text)
    text = re.sub(r"#(\w+)", r" \1 ", text)
    text = replace_emojis_with_sentiment(text)
    text = text.replace("&", " and ")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[\u2018\u2019]", "'", text)
    text = re.sub(r"[\u201C\u201D]", '"', text)
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = normalize_unicode(text)
    text = normalize_elongated_words(text)
    text = normalize_repeated_punctuation(text)
    text = collapse_whitespace(text)
    text = text.strip()
    return text


def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def normalize_elongated_words(text: str) -> str:
    # Collapses any run of 3+ identical chars down to 2 (e.g. "goooood" -> "good").
    # This alone handles the elongation case; no separate word-specific
    # pattern is needed on top of it.
    text = re.sub(r"(\w)\1{2,}", r"\1\1", text)
    return text


def replace_emojis_with_sentiment(text: str) -> str:
    emoji_map = {
        "😊": " happy ", "😃": " happy ", "😄": " happy ", "😁": " happy ",
        "😞": " sad ", "😡": " angry ", "😠": " angry ", "😍": " love ",
        "🤩": " excited ", "😭": " crying ", "😩": " upset ", "😒": " disappointed ",
        "🙌": " great ", "🔥": " great ", "⚠️": " warning ", "✅": " positive ",
        "❌": " negative ",
    }
    for emoji, replacement in emoji_map.items():
        text = text.replace(emoji, replacement)
    return text


def normalize_repeated_punctuation(text: str) -> str:
    text = re.sub(r"[!?]{3,}", "!!!", text)
    text = re.sub(r"[.]{3,}", "...", text)
    text = re.sub(r"[-_]{2,}", " ", text)
    return text


def collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _word_boundary_matches(phrase: str, text: str) -> list:
    """Find whole-phrase matches (not substrings of other words)."""
    pattern = r"(?<!\w)" + re.escape(phrase) + r"(?!\w)"
    return re.findall(pattern, text)


def compute_pidgin_boost(text: str) -> float:
    """
    Sums polarity for matched Pidgin phrases, using word-boundary matching
    and masking matched spans so overlapping entries (e.g. "no be" being a
    substring of "e no be") are not double-counted. Longer phrases are
    checked first so they "claim" their text before shorter phrases look
    at what's left.
    """
    lowered = f" {text.lower()} "
    working = lowered
    score = 0.0
    for phrase in sorted(PIDGIN_LEXICON, key=len, reverse=True):
        value = PIDGIN_LEXICON[phrase]
        matches = _word_boundary_matches(phrase, working)
        if matches:
            score += value * len(matches)
            pattern = r"(?<!\w)" + re.escape(phrase) + r"(?!\w)"
            working = re.sub(pattern, " ", working)
    return score


def compute_english_lexicon_boost(text: str) -> float:
    """Same idea as compute_pidgin_boost but for plain English sentiment words."""
    lowered = f" {text.lower()} "
    working = lowered
    score = 0.0
    for word in sorted(ENGLISH_LEXICON, key=len, reverse=True):
        value = ENGLISH_LEXICON[word]
        matches = _word_boundary_matches(word, working)
        if matches:
            score += value * len(matches)
            pattern = r"(?<!\w)" + re.escape(word) + r"(?!\w)"
            working = re.sub(pattern, " ", working)
    return score


def preserve_emphasis(text: str) -> float:
    exclamation_signal = text.count("!") * 0.05
    uppercase_ratio = sum(1 for ch in text if ch.isupper()) / max(len(text), 1)
    caps_signal = min(uppercase_ratio * 0.3, 0.3)
    return exclamation_signal + caps_signal
