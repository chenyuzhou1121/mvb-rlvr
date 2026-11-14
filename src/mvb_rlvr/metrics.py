"""Aggregate verifier decisions by language, dataset, and transformation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import random
from typing import Any, Iterable

ACCEPT = "accept"
REJECT = "reject"


@dataclass(frozen=True)
class Confusion:
    """Store binary decision counts plus ambiguous and error outcomes."""

    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0
    ambiguous: int = 0
    error: int = 0

    @property
    def n(self) -> int:
        """Return the count of all outcomes, including ambiguity and errors."""
        return self.tp + self.tn + self.fp + self.fn + self.ambiguous + self.error

    @property
    def comparable_n(self) -> int:
        """Return the number of binary accept/reject comparisons."""
        return self.tp + self.tn + self.fp + self.fn

    @property
    def error_count(self) -> int:
        """Return false positives, false negatives, and verifier errors."""
        return self.fp + self.fn + self.error

    def to_metrics(self) -> dict[str, float | int]:
        """Return counts and rates, using zero for undefined denominators.

        error_rate divides error_count by comparable_n. Other binary rates use
        the corresponding confusion counts; ambiguous outcomes are separate.
        """
        precision = self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0
        recall = self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        fpr = self.fp / (self.fp + self.tn) if self.fp + self.tn else 0.0
        fnr = self.fn / (self.fn + self.tp) if self.fn + self.tp else 0.0
        tnr = self.tn / (self.tn + self.fp) if self.tn + self.fp else 0.0
        balanced = (recall + tnr) / 2 if self.comparable_n else 0.0
        error_rate = self.error_count / self.comparable_n if self.comparable_n else 0.0
        return {
            "n": self.n,
            "comparable_n": self.comparable_n,
            "tp": self.tp,
            "tn": self.tn,
            "fp": self.fp,
            "fn": self.fn,
            "ambiguous": self.ambiguous,
            "error": self.error,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "fpr": fpr,
            "fnr": fnr,
            "balanced_accuracy": balanced,
            "error_rate": error_rate,
        }


def _confusion(records: Iterable[dict[str, Any]], verifier: str) -> Confusion:
    # Nonbinary expectations and explicit ambiguity take precedence over errors.
    counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "ambiguous": 0, "error": 0}
    for record in records:
        expected = record.get("expected_verdict")
        output = record.get("verifier_outputs", {}).get(verifier, {})
        verdict = output.get("verdict", "error")
        if expected not in (ACCEPT, REJECT) or verdict == "ambiguous":
            counts["ambiguous"] += 1
        elif verdict == "error":
            counts["error"] += 1
        elif expected == ACCEPT and verdict == ACCEPT:
            counts["tp"] += 1
        elif expected == REJECT and verdict == REJECT:
            counts["tn"] += 1
        elif expected == REJECT and verdict == ACCEPT:
            counts["fp"] += 1
        elif expected == ACCEPT and verdict == REJECT:
            counts["fn"] += 1
        else:
            counts["error"] += 1
    return Confusion(**counts)


def _group_key(record: dict[str, Any], fields: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(str(record.get(field, "")) for field in fields)


def _bootstrap_error_ci(errors: list[int], iterations: int = 200, seed: int = 42) -> dict[str, float]:
    if not errors:
        return {"low": 0.0, "high": 0.0}
    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(iterations):
        sample = [errors[rng.randrange(len(errors))] for _ in errors]
        values.append(sum(sample) / len(sample))
    values.sort()
    low_idx = int(0.025 * (len(values) - 1))
    high_idx = int(0.975 * (len(values) - 1))
    return {"low": values[low_idx], "high": values[high_idx]}


def summarize_verifier(
    records: list[dict[str, Any]],
    verifier: str,
    group_fields: tuple[str, ...] = ("language",),
) -> list[dict[str, Any]]:
    """Group verdict records and return metrics with bootstrap error intervals.

    Rows need expected_verdict and verifier_outputs keyed by verifier name.
    Interval samples include every binary expectation and treat any differing
    verdict, including ambiguity or a missing output, as an error.
    """
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[_group_key(record, group_fields)].append(record)

    rows: list[dict[str, Any]] = []
    for key, items in sorted(groups.items()):
        confusion = _confusion(items, verifier)
        row: dict[str, Any] = {field: value for field, value in zip(group_fields, key)}
        row["verifier"] = verifier
        row.update(confusion.to_metrics())
        errors: list[int] = []
        for item in items:
            if item.get("expected_verdict") not in (ACCEPT, REJECT):
                continue
            output = item.get("verifier_outputs", {}).get(verifier, {})
            errors.append(1 if output.get("verdict") != item.get("expected_verdict") else 0)
        ci = _bootstrap_error_ci(errors)
        row["error_rate_ci_low"] = ci["low"]
        row["error_rate_ci_high"] = ci["high"]
        rows.append(row)
    return rows


def compute_vlb(language_rows: list[dict[str, Any]], reference_language: str = "en") -> dict[str, Any]:
    """Return each verifier's max-minus-min language error rate and offsets.

    Read language, verifier, and error_rate from grouped metric rows. Reference
    offsets use a zero baseline when reference_language is absent.
    """
    by_verifier: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in language_rows:
        by_verifier[row["verifier"]].append(row)
    output: dict[str, Any] = {}
    for verifier, rows in by_verifier.items():
        rates = {row["language"]: row["error_rate"] for row in rows if "language" in row}
        if not rates:
            continue
        output[verifier] = {
            "vlb": max(rates.values()) - min(rates.values()),
            "error_rate_by_language": rates,
            "directional_bias_vs_reference": {
                lang: rate - rates.get(reference_language, 0.0) for lang, rate in rates.items()
            },
        }
    return output


def equivalent_transformation_variance(records: list[dict[str, Any]], verifier: str) -> list[dict[str, Any]]:
    """Report verdict variation across accept-labeled variants of each answer.

    Group by problem_id, language, and gold answer; emit groups with two or more
    rows, including verdict counts and a flag for any disagreement.
    """
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("expected_verdict") == ACCEPT:
            key = (
                str(record.get("problem_id", "")),
                str(record.get("language", "")),
                str(record.get("gold_answer_canonical", "")),
            )
            groups[key].append(record)
    rows: list[dict[str, Any]] = []
    for (problem_id, language, gold), items in sorted(groups.items()):
        verdicts = [
            item.get("verifier_outputs", {}).get(verifier, {}).get("verdict", "error")
            for item in items
        ]
        if len(verdicts) < 2:
            continue
        rows.append(
            {
                "verifier": verifier,
                "problem_id": problem_id,
                "language": language,
                "gold_answer_canonical": gold,
                "n_equivalent_variants": len(verdicts),
                "verdicts": dict(sorted((v, verdicts.count(v)) for v in set(verdicts))),
                "has_variance": len(set(verdicts)) > 1,
            }
        )
    return rows


def consensus_error_rate(records: list[dict[str, Any]], threshold: float = 0.5) -> dict[str, Any]:
    """Return the fraction of high-proxy rows whose trusted reward is below threshold.

    Only rows containing both reward values are eligible. Return zero when no
    eligible proxy reward reaches the threshold.
    """
    eligible = [
        record
        for record in records
        if record.get("proxy_reward") is not None and record.get("trusted_reward") is not None
    ]
    high_proxy = [record for record in eligible if float(record["proxy_reward"]) >= threshold]
    if not high_proxy:
        return {"eligible_n": len(eligible), "high_proxy_n": 0, "consensus_error_rate": 0.0}
    wrong = [record for record in high_proxy if float(record["trusted_reward"]) < threshold]
    return {
        "eligible_n": len(eligible),
        "high_proxy_n": len(high_proxy),
        "wrong_high_proxy_n": len(wrong),
        "consensus_error_rate": len(wrong) / len(high_proxy),
    }


def summarize(records: list[dict[str, Any]], verifiers: list[str]) -> dict[str, Any]:
    """Return grouped metrics, language gaps, variant variance, and reward errors."""
    language_rows: list[dict[str, Any]] = []
    transformation_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    variance_rows: list[dict[str, Any]] = []
    for verifier in verifiers:
        language_rows.extend(summarize_verifier(records, verifier, ("language",)))
        transformation_rows.extend(
            summarize_verifier(records, verifier, ("language", "transformation_type"))
        )
        source_rows.extend(summarize_verifier(records, verifier, ("source_dataset", "language")))
        variance_rows.extend(equivalent_transformation_variance(records, verifier))
    return {
        "by_language": language_rows,
        "by_transformation": transformation_rows,
        "by_source_language": source_rows,
        "vlb": compute_vlb(language_rows),
        "equivalent_transformation_variance": variance_rows,
        "consensus_error_rate": consensus_error_rate(records),
    }
