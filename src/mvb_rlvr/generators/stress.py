"""Generate labeled answer-format variants from seed math problems."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Iterable

from mvb_rlvr.canonicalize import canonicalize_answer
from mvb_rlvr.extractors import extract_final_answer
from mvb_rlvr.schema import BenchmarkRecord

FULLWIDTH_TRANS = str.maketrans(
    "0123456789.-+/%", "\uff10\uff11\uff12\uff13\uff14\uff15\uff16\uff17\uff18\uff19\uff0e\uff0d\uff0b\uff0f\uff05"
)
DIGIT_TO_CN = {
    "0": "\u96f6",
    "1": "\u4e00",
    "2": "\u4e8c",
    "3": "\u4e09",
    "4": "\u56db",
    "5": "\u4e94",
    "6": "\u516d",
    "7": "\u4e03",
    "8": "\u516b",
    "9": "\u4e5d",
}


@dataclass(frozen=True)
class SeedProblem:
    """Store a reference problem and optional wrong-answer override.

    Use the same source_dataset and problem_id across aligned translations.
    gold_answer is answer text; metadata is copied to generated records.
    """

    problem_id: str
    language: str
    source_dataset: str
    problem_text: str
    gold_answer: str
    incorrect_answer: str | None = None
    metadata: dict[str, object] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "SeedProblem":
        """Read the five required seed fields and optional answer or metadata."""
        return cls(
            problem_id=str(data["problem_id"]),
            language=str(data["language"]),
            source_dataset=str(data["source_dataset"]),
            problem_text=str(data["problem_text"]),
            gold_answer=str(data["gold_answer"]),
            incorrect_answer=str(data["incorrect_answer"]) if data.get("incorrect_answer") is not None else None,
            metadata=data.get("metadata") if isinstance(data.get("metadata"), dict) else None,
        )


def _stable_id(*parts: str) -> str:
    joined = "::".join(parts)
    digest = hashlib.sha1(joined.encode("utf-8")).hexdigest()[:12]
    return f"rec_{digest}"


def _fullwidth(answer: str) -> str:
    return answer.translate(FULLWIDTH_TRANS)


def _digit_kanji(answer: str) -> str:
    return "".join(DIGIT_TO_CN.get(ch, ch) for ch in answer)


def _with_explanation(answer: str, language: str) -> str:
    if language == "cn":
        return f"\u7b54\u6848\u662f\uff1a{answer}\u3002"
    if language == "jp":
        return f"\u7b54\u3048\u306f {answer} \u3067\u3059\u3002"
    return f"The answer is {answer}."


def _mixed_script(answer: str, language: str) -> str:
    if language == "cn":
        return f"final answer \u662f {answer}"
    if language == "jp":
        return f"final answer \u306f {answer}"
    return f"\u7b54\u6848: {answer}"


def _near_miss(answer: str, fallback: str | None) -> str:
    if fallback:
        return fallback
    canon = canonicalize_answer(answer)
    if canon.numbers:
        value = canon.numbers[-1] + 1
        if value.denominator == 1:
            return str(value.numerator)
        return f"{value.numerator}/{value.denominator}"
    return f"{answer}x"


def _percent_equivalent(answer: str) -> str | None:
    # Multiply by 100 because the emitted percent sign scales the value back.
    canon = canonicalize_answer(answer)
    if len(canon.numbers) != 1:
        return None
    percent = canon.numbers[0] * 100
    if percent.denominator == 1:
        return f"{percent.numerator}%"
    return f"{float(percent):.4g}%"


def _decimal_equivalent(answer: str) -> str | None:
    if "/" not in answer:
        return None
    canon = canonicalize_answer(answer)
    if len(canon.numbers) != 1:
        return None
    return f"{float(canon.numbers[0]):.8g}"


def _fraction_equivalent(answer: str) -> str | None:
    canon = canonicalize_answer(answer)
    if len(canon.numbers) != 1:
        return None
    value = canon.numbers[0]
    if value.denominator == 1:
        return None
    return f"{value.numerator}/{value.denominator}"


def _raw_completion(candidate: str, language: str) -> str:
    if language == "cn":
        return f"\u6211\u4eec\u9010\u6b65\u8ba1\u7b97\u540e\u5f97\u5230\u6700\u7ec8\u7ed3\u679c\u3002\n\u7b54\u6848\u662f\uff1a{candidate}"
    if language == "jp":
        return f"\u9806\u756a\u306b\u8a08\u7b97\u3059\u308b\u3068\u6700\u7d42\u7d50\u679c\u304c\u5f97\u3089\u308c\u307e\u3059\u3002\n\u7b54\u3048\u306f {candidate}"
    return f"After solving step by step, we obtain the result.\nFinal answer: {candidate}"


def generate_for_seed(seed: SeedProblem, stress_types: Iterable[str]) -> list[BenchmarkRecord]:
    """Return requested variants with labels, extracted answers, and stable IDs.

    Iterate stress_types in order. Names absent from this seed's variant map
    are skipped, including numeric transformations that cannot be constructed.
    """
    records: list[BenchmarkRecord] = []
    variants: dict[str, tuple[str, str, str]] = {
        "correct_plain": (seed.gold_answer, "correct", "accept"),
        "fullwidth_digits": (_fullwidth(seed.gold_answer), "correct", "accept"),
        "answer_with_explanation": (_with_explanation(seed.gold_answer, seed.language), "correct", "accept"),
        "mixed_script_answer": (_mixed_script(seed.gold_answer, seed.language), "correct", "accept"),
        "near_miss_numeric": (_near_miss(seed.gold_answer, seed.incorrect_answer), "incorrect", "reject"),
    }
    percent = _percent_equivalent(seed.gold_answer)
    if percent is not None:
        variants["percent_equivalent"] = (percent, "correct", "accept")
    decimal = _decimal_equivalent(seed.gold_answer)
    if decimal is not None:
        variants["decimal_equivalent"] = (decimal, "correct", "accept")
    fraction = _fraction_equivalent(seed.gold_answer)
    if fraction is not None:
        variants["fraction_equivalent"] = (fraction, "correct", "accept")
    if re.search(r"\d", seed.gold_answer):
        variants["kanji_digits"] = (_digit_kanji(seed.gold_answer), "correct", "accept")

    for stress_type in stress_types:
        if stress_type not in variants:
            continue
        candidate, label, expected = variants[stress_type]
        raw_completion = _raw_completion(candidate, seed.language)
        extracted, extraction_method = extract_final_answer(raw_completion)
        records.append(
            BenchmarkRecord(
                record_id=_stable_id(seed.problem_id, seed.language, stress_type, candidate),
                problem_id=seed.problem_id,
                language=seed.language,
                source_dataset=seed.source_dataset,
                problem_text=seed.problem_text,
                gold_answer_canonical=seed.gold_answer,
                candidate_answer=candidate,
                candidate_label=label,
                transformation_type=stress_type,
                expected_verdict=expected,
                raw_completion=raw_completion,
                extracted_answer=extracted,
                extraction_method=extraction_method,
                rollout_source="human_stress",
                cot_language=seed.language,
                trusted_reward=1.0 if expected == "accept" else 0.0,
                metadata={"generator": "answer_stress", **(seed.metadata or {})},
            )
        )
    return records


def generate_records(seed_rows: Iterable[dict[str, object]], stress_types: Iterable[str]) -> list[BenchmarkRecord]:
    """Apply stress_types to each seed row and concatenate the records."""
    records: list[BenchmarkRecord] = []
    for row in seed_rows:
        records.extend(generate_for_seed(SeedProblem.from_dict(row), stress_types))
    return records
