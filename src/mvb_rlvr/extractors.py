"""Extract final answers using boxed expressions and multilingual markers."""

from __future__ import annotations

import re

BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
FINAL_PATTERNS = (
    re.compile(r"(?:final answer|answer is|the answer is)\s*[:\uff1a]?\s*(.+)$", re.IGNORECASE),
    re.compile(r"(?:\u7b54\u6848\u662f|\u7b54\u6848\u4e3a)\s*[:\uff1a]?\s*(.+)$"),
    re.compile(r"(?:\u7b54\u3048\u306f|\u7b54\u3048)\s*[:\uff1a]?\s*(.+)$"),
)


def extract_final_answer(text: str) -> tuple[str, str]:
    """Return an answer and extraction-method tag from a completion.

    Prefer the last nonnested boxed expression, then a marked answer scanning
    lines backward, then the stripped final line. Empty input yields empty text.
    """
    stripped = text.strip()
    boxed = BOXED_RE.findall(stripped)
    if boxed:
        return boxed[-1].strip(), "boxed_regex"
    for line in reversed(stripped.splitlines()):
        cleaned = line.strip()
        if not cleaned:
            continue
        for pattern in FINAL_PATTERNS:
            match = pattern.search(cleaned)
            if match:
                return match.group(1).strip(" .\u3002"), "final_marker_regex"
    return stripped.splitlines()[-1].strip() if stripped else "", "last_line"
