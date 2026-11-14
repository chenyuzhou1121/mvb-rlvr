"""Reward answer agreement against cached samples from other languages."""

from __future__ import annotations

from collections.abc import Callable
import json
import logging
from math import sqrt
from pathlib import Path
from typing import Any

from mvb_rlvr.canonicalize import answers_equivalent
from mvb_rlvr.extractors import extract_final_answer
from mvb_rlvr.self_consistency import canonical_cluster_key, detect_response_language
from mvb_rlvr.training_rewards import completion_to_text


DEFAULT_LANGUAGES = ("en", "jp", "cn")


def _as_list(value: Any, n: int, default: str = "") -> list[str]:
    """Broadcast scalar or per-prompt values to contiguous completion groups."""
    if isinstance(value, list):
        items = [str(item) for item in value]
        if len(items) == n:
            return items
        if len(items) == 1:
            return items * n
        if items and n % len(items) == 0:
            repeats = n // len(items)
            return [item for item in items for _ in range(repeats)]
        raise ValueError(f"cannot align list of length {len(items)} to completions length {n}")
    if value is None:
        return [default for _ in range(n)]
    return [str(value) for _ in range(n)]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if not values:
        return 0.0
    avg = _mean(values)
    return sqrt(sum((value - avg) ** 2 for value in values) / len(values))


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _length_penalty(
    *,
    enabled: bool,
    completion_tokens: int | None,
    completion_chars: int,
    target_tokens: int | None,
    target_chars: int | None,
    weight: float,
    max_penalty: float,
) -> tuple[float, str, int | None]:
    # Prefer token counts when available; otherwise use the character target.
    if not enabled:
        return 0.0, "disabled", None
    if completion_tokens is not None and target_tokens and target_tokens > 0:
        observed = completion_tokens
        target = target_tokens
        unit = "tokens"
    elif target_chars and target_chars > 0:
        observed = completion_chars
        target = target_chars
        unit = "chars"
    else:
        return 0.0, "unconfigured", None
    excess_ratio = max(0.0, (observed - target) / target)
    return min(max_penalty, excess_ratio * weight), unit, observed


def problem_cache_key(source_dataset: str, problem_id: str) -> str:
    """Join dataset and problem identifiers; replace an empty dataset by unknown."""
    source = source_dataset or "unknown"
    return f"{source}::{problem_id}"


def load_support_cache(path: str | Path, support_model: str | None = None) -> tuple[str, dict[str, Any]]:
    """Return the chosen model name and its problem cache from a JSON file.

    Use support_model if supplied, otherwise default_model or the first model.
    Require a nonempty models map and a problems key in the chosen entry.
    """
    cache_path = Path(path)
    with cache_path.open(encoding="utf-8") as fh:
        cache = json.load(fh)
    models = cache.get("models")
    if not isinstance(models, dict) or not models:
        raise ValueError(f"support cache has no models: {cache_path}")
    if support_model:
        model_key = support_model
    else:
        model_key = str(cache.get("default_model") or next(iter(models)))
    if model_key not in models:
        raise ValueError(f"support model {model_key!r} not found in support cache {cache_path}")
    model_cache = models[model_key]
    if not isinstance(model_cache, dict) or "problems" not in model_cache:
        raise ValueError(f"support cache model {model_key!r} is missing problems")
    return model_key, model_cache


def find_problem_support(
    model_cache: dict[str, Any],
    *,
    source_dataset: str,
    problem_id: str,
) -> tuple[str | None, dict[str, Any] | None]:
    """Find dataset-qualified support, then a bare problem ID; return key/data."""
    problems = model_cache.get("problems", {})
    key = problem_cache_key(source_dataset, problem_id)
    if key in problems:
        return key, problems[key]
    if problem_id in problems:
        return problem_id, problems[problem_id]
    return None, None


def crosslingual_support_score(
    problem_support: dict[str, Any] | None,
    *,
    cluster_key: str,
    language: str,
    languages: tuple[str, ...] = DEFAULT_LANGUAGES,
) -> dict[str, Any]:
    """Return support fractions and a reward for one answer cluster.

    Exclude the target language and languages with no samples. Reward equals
    mean cluster fraction times the fraction of considered languages with a
    positive count. Missing support or failed extraction yields zero reward.
    """
    if not problem_support or cluster_key == "EXTRACTION_FAILED":
        return {
            "reward": 0.0,
            "leave_one_out_consensus": None,
            "other_language_coverage": None,
            "other_support_languages": 0,
            "support_languages_considered": 0,
            "support_fractions": {},
            "missing_support": problem_support is None,
        }

    support_by_language = problem_support.get("languages", {})
    fractions: dict[str, float] = {}
    other_support_languages = 0
    for current_language in languages:
        if current_language == language:
            continue
        info = support_by_language.get(current_language, {})
        total = int(info.get("total") or 0)
        if total <= 0:
            continue
        counts = info.get("cluster_counts", {})
        count = int(counts.get(cluster_key, 0))
        fraction = count / total
        fractions[current_language] = fraction
        if fraction > 0:
            other_support_languages += 1

    considered = len(fractions)
    if considered == 0:
        return {
            "reward": 0.0,
            "leave_one_out_consensus": None,
            "other_language_coverage": None,
            "other_support_languages": 0,
            "support_languages_considered": 0,
            "support_fractions": fractions,
            "missing_support": False,
        }

    # Each available language contributes equally, independent of sample count.
    leave_one_out = _mean(list(fractions.values()))
    coverage = other_support_languages / considered
    return {
        "reward": leave_one_out * coverage,
        "leave_one_out_consensus": leave_one_out,
        "other_language_coverage": coverage,
        "other_support_languages": other_support_languages,
        "support_languages_considered": considered,
        "support_fractions": fractions,
        "missing_support": False,
    }


def build_crosslingual_consensus_reward_fn(
    *,
    support_cache_path: str | Path,
    support_model: str | None = None,
    languages: tuple[str, ...] = DEFAULT_LANGUAGES,
    audit_path: str | Path | None = None,
    logger: logging.Logger | None = None,
    high_reward_threshold: float = 0.75,
    audit_label: str = "train",
    reward_clip_min: float = 0.0,
    reward_clip_max: float = 1.0,
    length_penalty_enabled: bool = False,
    length_target_tokens: int | None = None,
    length_target_chars: int | None = None,
    length_penalty_weight: float = 0.2,
    length_penalty_max: float = 0.2,
    language_consistency_enabled: bool = False,
    language_penalty_weight: float = 0.05,
    language_penalty_max: float = 0.1,
) -> Callable[..., list[float]]:
    """Load fixed support and return a per-completion crosslingual callback.

    support_cache_path is the JSON produced by build_support_cache; support_model
    selects its model. languages lists the tags considered for support. The
    callback accepts prompts, completions, gold_answer_canonical or gold_answer,
    language, source_dataset, problem_id, and optional completion_ids. Scalars
    broadcast; shorter column lists repeat items contiguously to match the batch.
    Prompts are accepted for trainer compatibility. Gold affects audits only.

    Base reward comes from crosslingual_support_score. Enabled length penalties
    use token counts and length_target_tokens, otherwise length_target_chars,
    scaling excess / target by weight up to the cap. Enabled language penalties
    subtract the smaller weight and cap for detected language mismatches. Clip
    final rewards to reward_clip_min and reward_clip_max.

    audit_path appends JSONL, logger reports batches, audit_label names entries,
    and high_reward_threshold selects the high-reward audit subset.
    """
    if reward_clip_min > reward_clip_max:
        raise ValueError(
            f"reward_clip_min must be <= reward_clip_max, got {reward_clip_min} > {reward_clip_max}"
        )
    model_key, model_cache = load_support_cache(support_cache_path, support_model=support_model)
    path = Path(audit_path) if audit_path else None
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
    batch_index = 0

    def reward_fn(
        prompts: list[Any],
        completions: list[Any],
        gold_answer: list[str] | None = None,
        gold_answer_canonical: list[str] | None = None,
        language: list[str] | None = None,
        source_dataset: list[str] | None = None,
        problem_id: list[str] | None = None,
        **kwargs: Any,
    ) -> list[float]:
        """Return rewards in completion order using aligned identifying columns."""
        nonlocal batch_index
        n = len(completions)
        if n == 0:
            return []
        gold_values = gold_answer_canonical if gold_answer_canonical is not None else gold_answer
        golds = _as_list(gold_values, n)
        languages_aligned = _as_list(language, n, "unknown")
        sources = _as_list(source_dataset, n, "unknown")
        problem_ids = _as_list(problem_id, n, "unknown")
        completion_ids = kwargs.get("completion_ids")
        if isinstance(completion_ids, list) and len(completion_ids) == n:
            completion_id_lengths = [
                len(item) if hasattr(item, "__len__") else None for item in completion_ids
            ]
        else:
            completion_id_lengths = [None] * n

        rewards: list[float] = []
        audit_rows: list[dict[str, Any]] = []
        for idx, completion in enumerate(completions):
            text = completion_to_text(completion)
            extracted, extraction_method = extract_final_answer(text)
            extracted = extracted.strip()
            if not extracted:
                cluster_key, cluster_id, canonical_info = "EXTRACTION_FAILED", "EXTRACTION_FAILED", {
                    "canonical_text": "",
                    "canonical_numbers": [],
                    "canonical_unit_values": [],
                    "canonical_trace": [],
                }
                extraction_failed = True
            else:
                cluster_key, cluster_id, canonical_info = canonical_cluster_key(extracted)
                extraction_failed = cluster_key == "EXTRACTION_FAILED"

            support_key, problem_support = find_problem_support(
                model_cache,
                source_dataset=sources[idx],
                problem_id=problem_ids[idx],
            )
            support = crosslingual_support_score(
                problem_support,
                cluster_key=cluster_key,
                language=languages_aligned[idx],
                languages=languages,
            )
            base_reward = float(support["reward"])
            length_penalty, length_penalty_unit, length_observed = _length_penalty(
                enabled=length_penalty_enabled,
                completion_tokens=completion_id_lengths[idx],
                completion_chars=len(text),
                target_tokens=length_target_tokens,
                target_chars=length_target_chars,
                weight=length_penalty_weight,
                max_penalty=length_penalty_max,
            )
            detected_language = detect_response_language(text)
            expected_language = languages_aligned[idx]
            language_mismatch = (
                language_consistency_enabled
                and detected_language is not None
                and expected_language in set(languages)
                and detected_language != expected_language
            )
            language_penalty = min(language_penalty_weight, language_penalty_max) if language_mismatch else 0.0
            reward_before_clip = base_reward - length_penalty - language_penalty
            reward = _clip(reward_before_clip, reward_clip_min, reward_clip_max)
            rewards.append(reward)

            trusted_correct: bool | None = None
            gold = golds[idx]
            if gold and not extraction_failed:
                try:
                    trusted_correct = answers_equivalent(gold, extracted)[0]
                except Exception:
                    trusted_correct = None
            high_wrong: bool | None = None
            if trusted_correct is not None and reward >= high_reward_threshold:
                high_wrong = not trusted_correct

            audit_rows.append(
                {
                    "audit_label": audit_label,
                    "batch_index": batch_index,
                    "idx": idx,
                    "support_model": model_key,
                    "support_problem_key": support_key,
                    "problem_id": problem_ids[idx],
                    "language": languages_aligned[idx],
                    "source_dataset": sources[idx],
                    "cluster_id": cluster_id,
                    "cluster_key": cluster_key,
                    "extracted_answer": extracted,
                    "extraction_method": extraction_method,
                    "extraction_failed": extraction_failed,
                    **canonical_info,
                    "support_fractions": support["support_fractions"],
                    "xlingual_leave_one_out_consensus": support["leave_one_out_consensus"],
                    "xlingual_other_language_coverage": support["other_language_coverage"],
                    "xlingual_other_support_languages": support["other_support_languages"],
                    "xlingual_support_languages_considered": support["support_languages_considered"],
                    "missing_support": support["missing_support"],
                    "base_reward": base_reward,
                    "length_penalty": length_penalty,
                    "length_penalty_unit": length_penalty_unit,
                    "length_observed": length_observed,
                    "length_target_tokens": length_target_tokens,
                    "length_target_chars": length_target_chars,
                    "language_penalty": language_penalty,
                    "language_consistency_expected": expected_language,
                    "language_consistency_detected": detected_language,
                    "language_mismatch": language_mismatch,
                    "reward_before_clip": reward_before_clip,
                    "reward_clip_min": reward_clip_min,
                    "reward_clip_max": reward_clip_max,
                    "reward_was_clipped": reward != reward_before_clip,
                    "reward": reward,
                    "trusted_correct_if_gold_available": trusted_correct,
                    "high_crosslingual_reward_trusted_wrong": high_wrong,
                    "completion_chars": len(text),
                    "completion_tokens": completion_id_lengths[idx],
                }
            )

        if logger:
            high_reward_rows = [
                row for row in audit_rows if float(row.get("reward", 0.0) or 0.0) >= high_reward_threshold
            ]
            high_wrong_count = sum(
                1 for row in high_reward_rows if row["high_crosslingual_reward_trusted_wrong"] is True
            )
            missing_support_count = sum(1 for row in audit_rows if row["missing_support"])
            extraction_failed_count = sum(1 for row in audit_rows if row["extraction_failed"])
            logger.info(
                "reward_batch mode=crosslingual_consensus label=%s batch_index=%d support_model=%s "
                "n=%d accepted=%d mean_reward=%.6f reward_std=%.6f missing_support=%d extraction_failures=%d "
                "mean_base_reward=%.6f mean_length_penalty=%.6f mean_language_penalty=%.6f "
                "high_crosslingual_reward=%d high_crosslingual_reward_trusted_wrong_rate=%.6f",
                audit_label,
                batch_index,
                model_key,
                n,
                len(high_reward_rows),
                _mean(rewards),
                _std(rewards),
                missing_support_count,
                extraction_failed_count,
                _mean([float(row["base_reward"]) for row in audit_rows]),
                _mean([float(row["length_penalty"]) for row in audit_rows]),
                _mean([float(row["language_penalty"]) for row in audit_rows]),
                len(high_reward_rows),
                high_wrong_count / len(high_reward_rows) if high_reward_rows else 0.0,
            )
        if path:
            with path.open("a", encoding="utf-8") as fh:
                for row in audit_rows:
                    fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        batch_index += 1
        return rewards

    return reward_fn
