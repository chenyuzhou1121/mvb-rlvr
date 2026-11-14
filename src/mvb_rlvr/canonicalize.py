"""Normalize multilingual answer text and compare numeric or unit values."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import re
import unicodedata

ANSWER_MARKERS = (
    "answer is",
    "the answer is",
    "final answer",
    "\u7b54\u6848\u662f",
    "\u7b54\u6848\u4e3a",
    "\u7b54\u3048\u306f",
    "\u7b54\u3048:",
    "\u7b54\u3048\uff1a",
    "\u89e3\u306f",
)

KANJI_DIGITS = {
    "\u96f6": 0,
    "\u3007": 0,
    "\u4e00": 1,
    "\u4e8c": 2,
    "\u4e24": 2,
    "\u5169": 2,
    "\u4e09": 3,
    "\u56db": 4,
    "\u4e94": 5,
    "\u516d": 6,
    "\u4e03": 7,
    "\u516b": 8,
    "\u4e5d": 9,
}

KANJI_UNITS = {"\u5341": 10, "\u767e": 100, "\u5343": 1000, "\u4e07": 10000, "\u842c": 10000}
NUMERIC_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:\s*/\s*[-+]?\d+(?:\.\d+)?)?\s*%?")
UNIT_RE = re.compile(r"([-+]?\d+(?:\.\d+)?)\s*(mm|cm|m|km)\b", re.IGNORECASE)
CJK_MAGNITUDE_RE = re.compile(r"([-+]?\d+(?:\.\d+)?)\s*([\u5341\u767e\u5343\u4e07\u842c])")


@dataclass(frozen=True)
class CanonicalAnswer:
    """Store normalized text, rational numbers, meter values, and trace tags."""

    original: str
    text: str
    numbers: tuple[Fraction, ...]
    unit_values: tuple[Fraction, ...]
    trace: tuple[str, ...]


def normalize_width(text: str) -> str:
    """Apply Unicode NFKC normalization, including fullwidth ASCII variants."""
    return unicodedata.normalize("NFKC", text)


def kanji_number_to_int(text: str) -> int | None:
    """Parse supported CJK numeral symbols using ten-thousand sections.

    Return None for empty text or characters outside KANJI_DIGITS and
    KANJI_UNITS. Units below ten thousand accumulate within each section.
    """
    stripped = text.strip()
    if not stripped or any(ch not in KANJI_DIGITS and ch not in KANJI_UNITS for ch in stripped):
        return None
    total = 0
    section = 0
    number = 0
    for ch in stripped:
        if ch in KANJI_DIGITS:
            number = KANJI_DIGITS[ch]
        elif ch in ("\u4e07", "\u842c"):
            # Close the current section before starting the next one.
            section += number
            if section == 0:
                section = 1
            total += section * 10000
            section = 0
            number = 0
        else:
            unit = KANJI_UNITS[ch]
            if number == 0:
                number = 1
            section += number * unit
            number = 0
    return total + section + number


def replace_kanji_numbers(text: str) -> tuple[str, bool]:
    """Replace runs of supported CJK numerals; return text and a change flag."""
    changed = False

    def repl(match: re.Match[str]) -> str:
        nonlocal changed
        value = kanji_number_to_int(match.group(0))
        if value is None:
            return match.group(0)
        changed = True
        return str(value)

    numeral_pattern = (
        r"[\u96f6\u3007\u4e00\u4e8c\u4e24\u5169\u4e09\u56db"
        r"\u4e94\u516d\u4e03\u516b\u4e5d\u5341\u767e\u5343\u4e07\u842c]+"
    )
    return re.sub(numeral_pattern, repl, text), changed


def replace_cjk_magnitude_numbers(text: str) -> tuple[str, bool]:
    """Expand Arabic numbers followed by a CJK unit; return text and a flag."""
    changed = False

    def repl(match: re.Match[str]) -> str:
        nonlocal changed
        number = Fraction(match.group(1))
        factor = KANJI_UNITS[match.group(2)]
        value = number * factor
        changed = True
        if value.denominator == 1:
            return str(value.numerator)
        return f"{float(value):.12g}"

    return CJK_MAGNITUDE_RE.sub(repl, text), changed


def strip_answer_markers(text: str) -> tuple[str, bool]:
    """Strip through the last occurrence of the first matching marker type."""
    lowered = text.lower()
    for marker in ANSWER_MARKERS:
        idx = lowered.rfind(marker)
        if idx >= 0:
            return text[idx + len(marker) :].strip(" :\uff1a\u3002.\n\t"), True
    return text, False


def _parse_fraction(token: str) -> Fraction | None:
    cleaned = token.strip().replace(" ", "").replace(",", "")
    is_percent = cleaned.endswith("%")
    if is_percent:
        cleaned = cleaned[:-1]
    try:
        if "/" in cleaned:
            left, right = cleaned.split("/", 1)
            value = Fraction(left) / Fraction(right)
        else:
            value = Fraction(cleaned)
    except (ValueError, ZeroDivisionError):
        return None
    if is_percent:
        value /= 100
    return value


def extract_numbers(text: str) -> tuple[Fraction, ...]:
    """Read signed numbers, fractions, and percentages as rational values.

    Preserve match order and duplicates; skip tokens that cannot be parsed,
    including fractions with a zero denominator. Percentages are divided by 100.
    """
    values: list[Fraction] = []
    for match in NUMERIC_RE.finditer(text):
        value = _parse_fraction(match.group(0))
        if value is not None:
            values.append(value)
    return tuple(values)


def extract_unit_values(text: str) -> tuple[Fraction, ...]:
    """Return numeric mm, cm, m, or km measurements converted to meters."""
    factors = {"mm": Fraction(1, 1000), "cm": Fraction(1, 100), "m": Fraction(1), "km": Fraction(1000)}
    values: list[Fraction] = []
    for match in UNIT_RE.finditer(text):
        number = _parse_fraction(match.group(1))
        if number is None:
            continue
        values.append(number * factors[match.group(2).lower()])
    return tuple(values)


def canonicalize_answer(answer: str) -> CanonicalAnswer:
    """Return normalized answer text, extracted values, and applied steps."""
    trace: list[str] = []
    text = answer.strip()
    normalized = normalize_width(text)
    if normalized != text:
        trace.append("unicode_nfkc")
    text = normalized
    text, changed = strip_answer_markers(text)
    if changed:
        trace.append("strip_answer_marker")
    # Expand Arabic-number magnitude suffixes before replacing CJK numerals.
    text, changed = replace_cjk_magnitude_numbers(text)
    if changed:
        trace.append("cjk_magnitude_to_arabic")
    text, changed = replace_kanji_numbers(text)
    if changed:
        trace.append("kanji_number_to_arabic")
    if "," in text:
        trace.append("remove_commas")
    text = text.replace(",", "")
    collapsed = re.sub(r"\s+", " ", text).strip().lower()
    if collapsed != text:
        trace.append("collapse_whitespace_lower")
    text = collapsed.strip("\u3002.;\uff1b")
    return CanonicalAnswer(
        original=answer,
        text=text,
        numbers=extract_numbers(text),
        unit_values=extract_unit_values(text),
        trace=tuple(trace),
    )


def answers_equivalent(gold: str, candidate: str) -> tuple[bool, CanonicalAnswer, CanonicalAnswer, list[str]]:
    """Return equivalence, both canonical forms, and a combined trace.

    Try raw text, normalized text, sets of meter values, and sets of numbers
    in that order. Set comparisons ignore value order and multiplicity.
    """
    gold_c = canonicalize_answer(gold)
    cand_c = canonicalize_answer(candidate)
    trace = list(dict.fromkeys([*gold_c.trace, *cand_c.trace]))
    if gold.strip() == candidate.strip():
        return True, gold_c, cand_c, [*trace, "raw_exact_match"]
    if gold_c.text == cand_c.text:
        return True, gold_c, cand_c, [*trace, "normalized_text_match"]
    if gold_c.unit_values and cand_c.unit_values and set(gold_c.unit_values) == set(cand_c.unit_values):
        return True, gold_c, cand_c, [*trace, "unit_equivalence"]
    if gold_c.numbers and cand_c.numbers and set(gold_c.numbers) == set(cand_c.numbers):
        return True, gold_c, cand_c, [*trace, "numeric_equivalence"]
    return False, gold_c, cand_c, trace
