"""Reward within-prompt answer agreement with optional completion penalties."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from fractions import Fraction
import hashlib
import json
import logging
from math import sqrt
from pathlib import Path
import re
from typing import Any

from mvb_rlvr.canonicalize import answers_equivalent, canonicalize_answer
from mvb_rlvr.extractors import extract_final_answer
from mvb_rlvr.training_rewards import completion_to_text


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


def _prompt_hash(prompt: Any) -> str:
    try:
        payload = json.dumps(prompt, ensure_ascii=False, sort_keys=True)
    except TypeError:
        payload = str(prompt)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _fraction_tuple(values: tuple[Fraction, ...]) -> tuple[str, ...]:
    return tuple(sorted(f"{value.numerator}/{value.denominator}" for value in values))


def canonical_cluster_key(answer: str) -> tuple[str, str, dict[str, Any]]:
    """Return a cluster key, short digest, and canonicalization details.

    Prefer sorted meter values, then sorted numbers, then normalized text.
    Numeric keys retain repeated values. Empty normalized text returns
    EXTRACTION_FAILED with an empty digest.
    """
    canonical = canonicalize_answer(answer)
    if not canonical.text:
        return "EXTRACTION_FAILED", "", {
            "canonical_text": canonical.text,
            "canonical_numbers": [],
            "canonical_unit_values": [],
            "canonical_trace": list(canonical.trace),
        }
    if canonical.unit_values:
        values = _fraction_tuple(canonical.unit_values)
        key = "unit:" + "|".join(values)
    elif canonical.numbers:
        values = _fraction_tuple(canonical.numbers)
        key = "num:" + "|".join(values)
    else:
        key = "text:" + canonical.text
    return key, hashlib.sha1(key.encode("utf-8")).hexdigest()[:12], {
        "canonical_text": canonical.text,
        "canonical_numbers": list(_fraction_tuple(canonical.numbers)),
        "canonical_unit_values": list(_fraction_tuple(canonical.unit_values)),
        "canonical_trace": list(canonical.trace),
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if not values:
        return 0.0
    avg = _mean(values)
    return sqrt(sum((value - avg) ** 2 for value in values) / len(values))


EN_WORD_RE = re.compile(r"\b(the|and|is|are|answer|final|therefore|because|equals?)\b", re.IGNORECASE)


def detect_response_language(text: str) -> str | None:
    """Infer jp from kana, cn from Han density, or en from ASCII letters.

    Return None if no branch matches. The heuristic examines completion text,
    including any reasoning and final answer.
    """
    jp = sum(1 for ch in text if "\u3040" <= ch <= "\u30ff")
    han = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    ascii_letters = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    if jp > 0:
        return "jp"
    if han > max(jp, ascii_letters / 4):
        return "cn"
    if ascii_letters >= 3 or EN_WORD_RE.search(text):
        return "en"
    return None


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
    penalty = min(max_penalty, excess_ratio * weight)
    return penalty, unit, observed


def build_self_consistency_reward_fn(
    *,
    num_generations: int,
    audit_path: str | Path | None = None,
    logger: logging.Logger | None = None,
    high_reward_threshold: float = 0.75,
    audit_label: str = "train",
    strict_group_alignment: bool = True,
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
    """Return a callback scoring each answer by its within-group cluster share.

    Completions must form contiguous groups of num_generations per prompt.
    The callback accepts prompts, completions, optional gold_answer_canonical
    (preferred over gold_answer), language, source_dataset, and problem_id.
    Columns broadcast as scalars or repeat each per-prompt item contiguously.
    strict_group_alignment checks identical problem IDs and prompt hashes.

    Base reward is cluster_size / num_generations, or zero for failed extraction.
    Enabled length penalties use completion_ids lengths with length_target_tokens,
    otherwise length_target_chars; excess / target is weighted and capped.
    Enabled language penalties subtract the smaller configured weight and cap
    when detected and expected languages differ. Clip the result to the supplied
    reward bounds. Gold answers affect audit fields only.

    audit_path appends JSONL; logger reports batches. audit_label names entries,
    and high_reward_threshold selects the high-reward audit subset.
    """
    if num_generations <= 0:
        raise ValueError(f"num_generations must be positive, got {num_generations}")
    if reward_clip_min > reward_clip_max:
        raise ValueError(
            f"reward_clip_min must be <= reward_clip_max, got {reward_clip_min} > {reward_clip_max}"
        )
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
        """Return aligned rewards; accept optional completion_ids for lengths."""
        nonlocal batch_index
        n = len(completions)
        if n == 0:
            return []
        if n % num_generations != 0:
            raise ValueError(
                f"completion count must be divisible by num_generations: n={n}, "
                f"num_generations={num_generations}"
            )
        prompts_aligned = _as_list(prompts, n)
        gold_values = gold_answer_canonical if gold_answer_canonical is not None else gold_answer
        golds = _as_list(gold_values, n)
        languages = _as_list(language, n, "unknown")
        sources = _as_list(source_dataset, n, "unknown")
        problem_ids = _as_list(problem_id, n, "unknown")
        completion_ids = kwargs.get("completion_ids")
        completion_id_lengths: list[int | None]
        if isinstance(completion_ids, list) and len(completion_ids) == n:
            completion_id_lengths = [len(item) if hasattr(item, "__len__") else None for item in completion_ids]
        else:
            completion_id_lengths = [None] * n

        rewards: list[float] = [0.0] * n
        audit_rows: list[dict[str, Any]] = []
        group_summaries: list[dict[str, Any]] = []

        for group_index, start in enumerate(range(0, n, num_generations)):
            end = start + num_generations
            group_problem_ids = problem_ids[start:end]
            group_prompt_hashes = [_prompt_hash(prompt) for prompt in prompts_aligned[start:end]]
            if strict_group_alignment:
                if len(set(group_problem_ids)) > 1:
                    raise ValueError(
                        "self-consistency group has multiple problem_id values: "
                        f"group_index={group_index} values={sorted(set(group_problem_ids))}"
                    )
                if len(set(group_prompt_hashes)) > 1:
                    raise ValueError(
                        "self-consistency group has multiple prompt hashes: "
                        f"group_index={group_index} values={sorted(set(group_prompt_hashes))}"
                    )

            extracted_items: list[dict[str, Any]] = []
            valid_keys: list[str] = []
            for offset, completion in enumerate(completions[start:end]):
                idx = start + offset
                text = completion_to_text(completion)
                extracted, method = extract_final_answer(text)
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
                if not extraction_failed:
                    valid_keys.append(cluster_key)
                extracted_items.append(
                    {
                        "idx": idx,
                        "group_offset": offset,
                        "completion_text": text,
                        "completion_chars": len(text),
                        "completion_tokens": completion_id_lengths[idx],
                        "extracted_answer": extracted,
                        "extraction_method": method,
                        "extraction_failed": extraction_failed,
                        "cluster_key": cluster_key,
                        "cluster_id": cluster_id,
                        **canonical_info,
                    }
                )

            # Failed extractions stay in the denominator but receive no votes.
            cluster_counts = Counter(valid_keys)
            largest_cluster_size = max(cluster_counts.values(), default=0)
            largest_cluster_fraction = largest_cluster_size / num_generations
            cluster_count = len(cluster_counts)
            extraction_failures = sum(1 for item in extracted_items if item["extraction_failed"])

            for item in extracted_items:
                idx = int(item["idx"])
                cluster_size = 0 if item["extraction_failed"] else int(cluster_counts[item["cluster_key"]])
                base_reward = cluster_size / num_generations if cluster_size else 0.0
                length_penalty, length_penalty_unit, length_observed = _length_penalty(
                    enabled=length_penalty_enabled,
                    completion_tokens=item["completion_tokens"],
                    completion_chars=item["completion_chars"],
                    target_tokens=length_target_tokens,
                    target_chars=length_target_chars,
                    weight=length_penalty_weight,
                    max_penalty=length_penalty_max,
                )
                detected_language = detect_response_language(str(item["completion_text"]))
                expected_language = languages[idx]
                language_mismatch = (
                    language_consistency_enabled
                    and detected_language is not None
                    and expected_language in {"en", "jp", "cn"}
                    and detected_language != expected_language
                )
                language_penalty = min(language_penalty_weight, language_penalty_max) if language_mismatch else 0.0
                reward_before_clip = base_reward - length_penalty - language_penalty
                reward = _clip(reward_before_clip, reward_clip_min, reward_clip_max)
                reward_was_clipped = reward != reward_before_clip
                rewards[idx] = reward

                trusted_correct: bool | None = None
                gold = golds[idx]
                if gold and not item["extraction_failed"]:
                    try:
                        trusted_correct = answers_equivalent(gold, str(item["extracted_answer"]))[0]
                    except Exception:
                        trusted_correct = None
                high_wrong: bool | None = None
                if trusted_correct is not None and reward >= high_reward_threshold:
                    high_wrong = not trusted_correct

                audit_rows.append(
                    {
                        "audit_label": audit_label,
                        "batch_index": batch_index,
                        "group_index": group_index,
                        "idx": idx,
                        "group_offset": item["group_offset"],
                        "problem_id": problem_ids[idx],
                        "language": languages[idx],
                        "source_dataset": sources[idx],
                        "prompt_hash": group_prompt_hashes[idx - start],
                        "group_size": num_generations,
                        "cluster_id": item["cluster_id"],
                        "cluster_size": cluster_size,
                        "cluster_count": cluster_count,
                        "largest_cluster_fraction": largest_cluster_fraction,
                        "extracted_answer": item["extracted_answer"],
                        "extraction_method": item["extraction_method"],
                        "extraction_failed": item["extraction_failed"],
                        "canonical_text": item["canonical_text"],
                        "canonical_numbers": item["canonical_numbers"],
                        "canonical_unit_values": item["canonical_unit_values"],
                        "canonical_trace": item["canonical_trace"],
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
                        "reward_was_clipped": reward_was_clipped,
                        "reward": reward,
                        "trusted_correct_if_gold_available": trusted_correct,
                        "high_self_reward_trusted_wrong": high_wrong,
                        "completion_chars": item["completion_chars"],
                        "completion_tokens": item["completion_tokens"],
                    }
                )

            group_rewards = rewards[start:end]
            group_summaries.append(
                {
                    "group_index": group_index,
                    "cluster_count": cluster_count,
                    "largest_cluster_fraction": largest_cluster_fraction,
                    "extraction_failures": extraction_failures,
                    "mean_reward": _mean(group_rewards),
                    "reward_std": _std(group_rewards),
                }
            )

        if logger:
            extraction_failure_count = sum(int(row["extraction_failures"]) for row in group_summaries)
            clipped_count = sum(1 for row in audit_rows if row["reward_was_clipped"])
            language_mismatch_count = sum(1 for row in audit_rows if row["language_mismatch"])
            high_self_reward_rows = [
                row for row in audit_rows if float(row.get("reward", 0.0) or 0.0) >= high_reward_threshold
            ]
            high_wrong_count = sum(1 for row in high_self_reward_rows if row["high_self_reward_trusted_wrong"] is True)
            high_wrong_rate = high_wrong_count / len(high_self_reward_rows) if high_self_reward_rows else 0.0
            logger.info(
                "reward_batch mode=self_consistency label=%s batch_index=%d n=%d accepted=%d "
                "mean_reward=%.6f reward_std=%.6f groups=%d num_generations=%d mean_cluster_count=%.6f "
                "mean_largest_cluster_fraction=%.6f extraction_failures=%d mean_base_reward=%.6f "
                "mean_length_penalty=%.6f mean_language_penalty=%.6f clipped=%d language_mismatches=%d "
                "high_self_reward=%d high_self_reward_trusted_wrong_rate=%.6f",
                audit_label,
                batch_index,
                n,
                n - extraction_failure_count,
                _mean(rewards),
                _std(rewards),
                len(group_summaries),
                num_generations,
                _mean([float(row["cluster_count"]) for row in group_summaries]),
                _mean([float(row["largest_cluster_fraction"]) for row in group_summaries]),
                extraction_failure_count,
                _mean([float(row["base_reward"]) for row in audit_rows]),
                _mean([float(row["length_penalty"]) for row in audit_rows]),
                _mean([float(row["language_penalty"]) for row in audit_rows]),
                clipped_count,
                language_mismatch_count,
                len(high_self_reward_rows),
                high_wrong_rate,
            )
        if path:
            with path.open("a", encoding="utf-8") as fh:
                for row in audit_rows:
                    fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        batch_index += 1
        return rewards

    return reward_fn
