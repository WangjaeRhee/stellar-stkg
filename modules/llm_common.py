"""
Shared LLM call settings and response-cleaning helpers.

- LLM_OPTIONS: deterministic decoding options passed to every ollama
  client.chat call (extraction, normalization, judging).
- clean_normalizer_response: turns a free-form normalizer reply into a
  single clean value, or falls back to the original value when the reply
  looks broken (explanation text, CJK characters, far too long).
"""

import re

# Deterministic decoding for every Ollama call in the framework.
LLM_OPTIONS = {"temperature": 0, "seed": 42}

# Prefixes that small models like to prepend to the answer.
_PREFIX_RE = re.compile(
    r"^\s*(normalized|normalised|country|answer|output|result|entity|"
    r"relation|location)\s*[:：]\s*",
    re.IGNORECASE,
)

# CJK Unified Ideographs, Hiragana/Katakana, Hangul.
_CJK_RE = re.compile(r"[一-鿿㐀-䶿぀-ヿ가-힯]")

# Hard limits for a normalized value.
MAX_NORMALIZED_LEN = 80
MAX_GROWTH_FACTOR = 3
MIN_GROWTH_ALLOWANCE = 30  # short inputs (e.g. "UN") may legitimately expand


def contains_cjk(text):
    return bool(_CJK_RE.search(text or ""))


def first_nonempty_line(text):
    for line in (text or "").splitlines():
        if line.strip():
            return line.strip()
    return ""


def clean_normalizer_response(raw, original):
    """
    Reduce a normalizer reply to one value.

    Returns (value, info) where info is None when the reply was used as-is
    (after trimming) or a dict describing why the original was kept:
        {"reason": <str>, "raw": <raw reply>, "candidate": <cleaned line>}
    """
    line = first_nonempty_line(raw)
    candidate = _PREFIX_RE.sub("", line)
    # strip markdown emphasis / code ticks / wrapping quotes / trailing punct
    candidate = candidate.strip().strip("*`").strip()
    candidate = candidate.strip('"\'“”‘’').strip()
    candidate = candidate.rstrip(".,;:").strip()
    # a second pass in case the prefix was inside the quotes
    candidate = _PREFIX_RE.sub("", candidate).strip().strip('"\'“”‘’').strip()

    reason = None
    if not candidate:
        reason = "empty"
    elif contains_cjk(candidate):
        reason = "cjk"
    elif len(candidate) > MAX_NORMALIZED_LEN:
        reason = "too_long_abs"
    elif len(candidate) > max(MAX_GROWTH_FACTOR * len(original or ""),
                              MIN_GROWTH_ALLOWANCE):
        reason = "too_long_rel"

    if reason:
        return original, {"reason": reason, "raw": raw, "candidate": candidate}
    return candidate, None
