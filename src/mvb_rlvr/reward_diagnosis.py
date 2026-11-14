"""Compare verifier rewards with canonical equivalence and summarize errors."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import fields
import random
import re
from typing import Any, Iterable

from mvb_rlvr.canonicalize import CanonicalAnswer, answers_equivalent
from mvb_rlvr.schema import BenchmarkRecord, VerifierVerdict
from mvb_rlvr.verifiers.rule import build_verifier

ACCEPT = "accept"
REJECT = "reject"
REFUSAL_RE = re.compile(r"(\u65e0\u6cd5|\u7121\u6cd5|\u3067\u304d\u307e\u305b\u3093|cannot|can't|unable|not enough)", re.IGNORECASE)


def _record_from_row(row: dict[str, Any]) -> BenchmarkRecord:
    allowed = {field.name for field in fields(BenchmarkRecord)}
    return BenchmarkRecord.from_dict({key: value for key, value in row.items() if key in allowed})


def _numbers_to_strings(answer: CanonicalAnswer) -> list[str]:
    return [str(value) for value in answer.numbers]


def _reward_from_verdict(verdict: str) -> float | None:
    if verdict == ACCEPT:
        return 1.0
    if verdict == REJECT:
        return 0.0
    return None


def reward_error_type(trusted_reward: float, proxy_reward: float | None) -> str:
    """Classify binary rewards as tp/tn/fp/fn, missing proxy, or ambiguous."""
    if proxy_reward is None:
        return "proxy_error"
    if trusted_reward == 1.0 and proxy_reward == 1.0:
        return "tp"
    if trusted_reward == 0.0 and proxy_reward == 0.0:
        return "tn"
    if trusted_reward == 0.0 and proxy_reward == 1.0:
        return "fp"
    if trusted_reward == 1.0 and proxy_reward == 0.0:
        return "fn"
    return "ambiguous"


def extraction_quality(row: dict[str, Any]) -> dict[str, Any]:
    """Return extraction method, emptiness, refusal-pattern, and length fields."""
    candidate = str(row.get("candidate_answer") or row.get("extracted_answer") or "")
    return {
        "extraction_method": row.get("extraction_method", "unknown"),
        "candidate_empty": candidate.strip() == "",
        "refusal_like": bool(REFUSAL_RE.search(candidate)),
        "candidate_length": len(candidate),
    }


def diagnose_records(
    rows: Iterable[dict[str, Any]],
    verifier_names: list[str],
    primary_verifier: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return annotated records and one detail row per record-verifier pair.

    Input rows follow BenchmarkRecord. Recompute trusted_reward and labels with
    answers_equivalent, replacing existing labels and verifier outputs. Convert
    accept/reject verdicts to 1.0/0.0; other verdicts map to None. Set proxy_reward
    from primary_verifier, defaulting to the first name in verifier_names.
    """
    verifiers = [build_verifier(name) for name in verifier_names]
    primary = primary_verifier or (verifier_names[0] if verifier_names else None)
    annotated: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []

    for row in rows:
        record = _record_from_row(row)
        equivalent, gold, candidate, trace = answers_equivalent(
            record.gold_answer_canonical,
            record.candidate_answer,
        )
        trusted_reward = 1.0 if equivalent else 0.0
        trusted_label = "correct" if equivalent else "incorrect"
        expected_verdict = ACCEPT if equivalent else REJECT
        out = dict(row)
        out["trusted_reward"] = trusted_reward
        out["candidate_label"] = trusted_label
        out["expected_verdict"] = expected_verdict
        out["trusted_equivalence"] = {
            "equivalent": equivalent,
            "gold_text": gold.text,
            "candidate_text": candidate.text,
            "gold_numbers": _numbers_to_strings(gold),
            "candidate_numbers": _numbers_to_strings(candidate),
            "trace": trace,
        }
        out["extraction_quality"] = extraction_quality(out)
        out["verifier_outputs"] = {}
        out["proxy_rewards"] = {}
        out["reward_error_types"] = {}

        for verifier in verifiers:
            verdict: VerifierVerdict = verifier.verify(record)
            proxy_reward = _reward_from_verdict(verdict.verdict)
            error_type = reward_error_type(trusted_reward, proxy_reward)
            out["verifier_outputs"][verdict.verifier_name] = verdict.to_dict()
            out["proxy_rewards"][verdict.verifier_name] = proxy_reward
            out["reward_error_types"][verdict.verifier_name] = error_type
            if verdict.verifier_name == primary:
                out["proxy_reward"] = proxy_reward
            details.append(
                {
                    "record_id": record.record_id,
                    "problem_id": record.problem_id,
                    "language": record.language,
                    "source_dataset": record.source_dataset,
                    "model_name": record.model_name,
                    "rollout_source": record.rollout_source,
                    "cot_language": record.cot_language,
                    "verifier": verdict.verifier_name,
                    "trusted_reward": trusted_reward,
                    "proxy_reward": proxy_reward,
                    "reward_error_type": error_type,
                    "candidate_answer": record.candidate_answer,
                    "gold_answer_canonical": record.gold_answer_canonical,
                    "extraction_method": out["extraction_quality"]["extraction_method"],
                    "refusal_like": out["extraction_quality"]["refusal_like"],
                    "equivalence_trace": trace,
                    "verdict": verdict.verdict,
                    "normalized_gold": verdict.normalized_gold,
                    "normalized_candidate": verdict.normalized_candidate,
                    "rationale": verdict.rationale,
                }
            )
        annotated.append(out)

    return annotated, details


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _bootstrap_ci(values: list[float], iterations: int = 500, seed: int = 42) -> dict[str, float]:
    if not values:
        return {"low": 0.0, "high": 0.0}
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(iterations):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(sum(sample) / len(sample))
    means.sort()
    low_idx = int(0.025 * (len(means) - 1))
    high_idx = int(0.975 * (len(means) - 1))
    return {"low": means[low_idx], "high": means[high_idx]}


def _phi(tp: int, tn: int, fp: int, fn: int) -> float:
    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if denom <= 0:
        return 0.0
    return ((tp * tn) - (fp * fn)) / (denom ** 0.5)


def summarize_group(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate diagnostic detail rows into reward, error, and language metrics.

    Binary comparable rows determine confusion rates; zero denominators yield
    zero. Expected policy pressure is mean(proxy - trusted) over available proxy
    values. Bootstrap intervals resample row values with a fixed random seed.
    """
    tp = sum(1 for row in items if row["reward_error_type"] == "tp")
    tn = sum(1 for row in items if row["reward_error_type"] == "tn")
    fp = sum(1 for row in items if row["reward_error_type"] == "fp")
    fn = sum(1 for row in items if row["reward_error_type"] == "fn")
    proxy_error = sum(1 for row in items if row["reward_error_type"] == "proxy_error")
    comparable = tp + tn + fp + fn
    trusted_correct = tp + fn
    trusted_incorrect = tn + fp
    reward_errors = [
        1.0 if row["reward_error_type"] in ("fp", "fn") else 0.0
        for row in items if row["reward_error_type"] != "proxy_error"
    ]
    fn_values = [
        1.0 if row["reward_error_type"] == "fn" else 0.0
        for row in items
        if row["trusted_reward"] == 1.0 and row["proxy_reward"] is not None
    ]
    fp_values = [
        1.0 if row["reward_error_type"] == "fp" else 0.0
        for row in items
        if row["trusted_reward"] == 0.0 and row["proxy_reward"] is not None
    ]
    pressure_values = [
        float(row["proxy_reward"]) - float(row["trusted_reward"])
        for row in items
        if row["proxy_reward"] is not None
    ]
    cot_language_counts = Counter(str(row.get("cot_language") or "unknown") for row in items)
    cot_known = [row for row in items if row.get("cot_language")]
    cot_drift = sum(1 for row in cot_known if str(row.get("cot_language")) != str(row.get("language")))
    return {
        "n": len(items),
        "comparable_n": comparable,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "proxy_error": proxy_error,
        "trusted_correct": trusted_correct,
        "trusted_incorrect": trusted_incorrect,
        "trusted_accuracy": _safe_div(trusted_correct, comparable),
        "proxy_accept_rate": _safe_div(tp + fp, comparable),
        "reward_accuracy": _safe_div(tp + tn, comparable),
        "reward_error_rate": _safe_div(fp + fn, comparable),
        "false_negative_mass": _safe_div(fn, trusted_correct),
        "false_positive_mass": _safe_div(fp, trusted_incorrect),
        "precision": _safe_div(tp, tp + fp),
        "recall": _safe_div(tp, tp + fn),
        "expected_policy_pressure": sum(pressure_values) / len(pressure_values) if pressure_values else 0.0,
        "phi_correlation": _phi(tp, tn, fp, fn),
        "cot_language_counts": dict(sorted(cot_language_counts.items())),
        "cot_language_drift_rate": _safe_div(cot_drift, len(cot_known)),
        "reward_error_rate_ci": _bootstrap_ci(reward_errors),
        "false_negative_mass_ci": _bootstrap_ci(fn_values),
        "false_positive_mass_ci": _bootstrap_ci(fp_values),
    }


def _group_rows(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row.get(key, "")) for key in keys)].append(row)
    output: list[dict[str, Any]] = []
    for key, items in sorted(groups.items()):
        result = {name: value for name, value in zip(keys, key)}
        result.update(summarize_group(items))
        output.append(result)
    return output


def extraction_stats(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return per-language extraction method, empty, refusal, and fallback counts."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("language", ""))].append(row)
    output: list[dict[str, Any]] = []
    for language, items in sorted(groups.items()):
        methods = Counter(str(row.get("extraction_method", "unknown")) for row in items)
        empty = sum(1 for row in items if not str(row.get("candidate_answer") or row.get("extracted_answer") or "").strip())
        refusal = sum(1 for row in items if REFUSAL_RE.search(str(row.get("candidate_answer") or row.get("extracted_answer") or "")))
        fallback = sum(1 for row in items if str(row.get("extraction_method", "")) == "last_line")
        output.append(
            {
                "language": language,
                "n": len(items),
                "method_counts": dict(sorted(methods.items())),
                "empty_count": empty,
                "empty_rate": _safe_div(empty, len(items)),
                "refusal_like_count": refusal,
                "refusal_like_rate": _safe_div(refusal, len(items)),
                "last_line_count": fallback,
                "last_line_rate": _safe_div(fallback, len(items)),
            }
        )
    return output


def compute_vlb(rows: list[dict[str, Any]], reference_language: str = "en") -> dict[str, Any]:
    """Return per-verifier language gaps for reward errors and conditional errors.

    Input rows contain language-level summarize_group results and a verifier
    name. Reference offsets use zero when the reference language is absent.
    """
    by_verifier: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_verifier[str(row["verifier"])].append(row)
    output: dict[str, Any] = {}
    for verifier, items in by_verifier.items():
        rates = {row["language"]: row["reward_error_rate"] for row in items if row.get("language")}
        fn_rates = {row["language"]: row["false_negative_mass"] for row in items if row.get("language")}
        fp_rates = {row["language"]: row["false_positive_mass"] for row in items if row.get("language")}
        if not rates:
            continue
        output[verifier] = {
            "vlb_reward_error_rate": max(rates.values()) - min(rates.values()),
            "vlb_false_negative_mass": max(fn_rates.values()) - min(fn_rates.values()) if fn_rates else 0.0,
            "vlb_false_positive_mass": max(fp_rates.values()) - min(fp_rates.values()) if fp_rates else 0.0,
            "reward_error_rate_by_language": rates,
            "false_negative_mass_by_language": fn_rates,
            "false_positive_mass_by_language": fp_rates,
            "directional_reward_error_vs_reference": {
                language: value - rates.get(reference_language, 0.0)
                for language, value in rates.items()
            },
        }
    return output


def summarize_diagnosis(
    original_rows: list[dict[str, Any]],
    detail_rows: list[dict[str, Any]],
    verifier_names: list[str],
    primary_verifier: str,
) -> dict[str, Any]:
    """Summarize annotated rows and details by language and source dataset.

    original_rows supplies record counts and extraction statistics; detail_rows
    supplies verifier metrics and up to twenty examples per binary error type.
    """
    by_language = _group_rows(detail_rows, ("verifier", "language"))
    by_source_language = _group_rows(detail_rows, ("verifier", "source_dataset", "language"))
    return {
        "n_records": len(original_rows),
        "verifiers": verifier_names,
        "primary_verifier": primary_verifier,
        "extraction_stats_by_language": extraction_stats(original_rows),
        "by_language": by_language,
        "by_source_language": by_source_language,
        "vlb": compute_vlb(by_language),
        "error_examples": {
            "false_negative": [row for row in detail_rows if row["reward_error_type"] == "fn"][:20],
            "false_positive": [row for row in detail_rows if row["reward_error_type"] == "fp"][:20],
        },
        "reference_equivalence_function": "answers_equivalent",
    }
