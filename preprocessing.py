import html
import re
import unicodedata


PIDGIN_LEXICON = {
    "no gree": 0.9,
    "wahala": 0.8,
    "sweet die": 0.9,
    "on point": 0.9,
    "shine your eye": 0.7,
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
    "mad o": -0.7,
    "shey": 0.1,
    "naija": 0.2,
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
    text = unicodedata.normalize("NFKC", text)
    return text


def normalize_elongated_words(text: str) -> str:
    text = re.sub(r"(\w)\1{2,}", r"\1\1", text)
    text = re.sub(r"(goooo+od|Goooo+od)", "good", text, flags=re.IGNORECASE)
    return text


def replace_emojis_with_sentiment(text: str) -> str:
    emoji_map = {
        "😊": " happy ",
        "😃": " happy ",
        "😄": " happy ",
        "😁": " happy ",
        "😞": " sad ",
        "😡": " angry ",
        "😠": " angry ",
        "😍": " love ",
        "🤩": " excited ",
        "😭": " crying ",
        "😩": " upset ",
        "😒": " disappointed ",
        "🙌": " great ",
        "🔥": " great ",
        "⚠️": " warning ",
        "✅": " positive ",
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


def compute_pidgin_boost(text: str) -> float:
    lowered = text.lower()
    score = 0.0
    for phrase, value in PIDGIN_LEXICON.items():
        if phrase in lowered:
            score += value
    return score


def preserve_emphasis(text: str) -> float:
    exclamation_signal = text.count("!") * 0.05
    uppercase_ratio = sum(1 for ch in text if ch.isupper()) / max(len(text), 1)
    caps_signal = min(uppercase_ratio * 0.3, 0.3)
    return exclamation_signal + caps_signal
