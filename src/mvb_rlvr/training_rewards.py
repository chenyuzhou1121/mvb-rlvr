"""Adapt answer verifiers to batched training reward callbacks."""

from __future__ import annotations

from collections.abc import Callable
import json
import logging
from pathlib import Path
from typing import Any

from mvb_rlvr.extractors import extract_final_answer
from mvb_rlvr.schema import BenchmarkRecord
from mvb_rlvr.verifiers.rule import build_verifier


def completion_to_text(completion: Any) -> str:
    """Read text from a string, content dictionary, or final chat message.

    Lists ending in a dictionary use that last message's content; other lists
    join their string representations with newlines. Other inputs become str.
    """
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    if isinstance(completion, list):
        if completion and isinstance(completion[-1], dict):
            return str(completion[-1].get("content", ""))
        return "\n".join(str(item) for item in completion)
    return str(completion)


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


def build_rule_reward_fn(
    verifier_name: str = "rule_exact_match",
    audit_path: str | Path | None = None,
    logger: logging.Logger | None = None,
) -> Callable[..., list[float]]:
    """Return a callback awarding 1.0 for accept and 0.0 for other verdicts.

    verifier_name selects a built-in verifier. audit_path appends per-completion
    JSONL details; logger receives batch totals. The callback takes prompts,
    completions, and optional gold_answer_canonical (preferred over gold_answer),
    language, source_dataset, and problem_id columns. Scalars broadcast; shorter
    lists repeat each item contiguously when their length divides the batch.
    Prompts are accepted for trainer compatibility and do not affect rewards.
    """
    verifier = build_verifier(verifier_name)
    path = Path(audit_path) if audit_path else None
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)

    def reward_fn(
        prompts: list[Any],
        completions: list[Any],
        gold_answer: list[str] | None = None,
        gold_answer_canonical: list[str] | None = None,
        language: list[str] | None = None,
        source_dataset: list[str] | None = None,
        problem_id: list[str] | None = None,
        **_: Any,
    ) -> list[float]:
        """Return one reward per completion in input order using aligned columns."""
        gold_values = gold_answer_canonical if gold_answer_canonical is not None else gold_answer
        n = len(completions)
        golds = _as_list(gold_values, n)
        languages = _as_list(language, n, "unknown")
        sources = _as_list(source_dataset, n, "unknown")
        problem_ids = _as_list(problem_id, n, "unknown")
        rewards: list[float] = []
        audit_rows: list[dict[str, Any]] = []
        for idx, completion in enumerate(completions):
            text = completion_to_text(completion)
            extracted, method = extract_final_answer(text)
            record = BenchmarkRecord(
                record_id=f"train_reward_{idx}",
                problem_id=problem_ids[idx],
                language=languages[idx],
                source_dataset=sources[idx],
                problem_text="",
                gold_answer_canonical=golds[idx],
                candidate_answer=extracted,
                candidate_label="ambiguous",
                transformation_type="grpo_completion",
                expected_verdict="ambiguous",
                raw_completion=text,
                extracted_answer=extracted,
                extraction_method=method,
                rollout_source="grpo_train",
            )
            verdict = verifier.verify(record)
            reward = 1.0 if verdict.verdict == "accept" else 0.0
            rewards.append(reward)
            audit_rows.append(
                {
                    "idx": idx,
                    "problem_id": problem_ids[idx],
                    "language": languages[idx],
                    "source_dataset": sources[idx],
                    "verifier": verifier_name,
                    "gold_answer": golds[idx],
                    "extracted_answer": extracted,
                    "extraction_method": method,
                    "verdict": verdict.verdict,
                    "reward": reward,
                    "normalized_gold": verdict.normalized_gold,
                    "normalized_candidate": verdict.normalized_candidate,
                    "normalization_trace": verdict.normalization_trace,
                    "verdict_error": verdict.error,
                }
            )
        if logger:
            accepted = sum(rewards)
            logger.info(
                "reward_batch verifier=%s n=%d accepted=%d mean_reward=%.6f",
                verifier_name,
                len(rewards),
                int(accepted),
                accepted / len(rewards) if rewards else 0.0,
            )
        if path:
            with path.open("a", encoding="utf-8") as fh:
                for row in audit_rows:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        return rewards

    return reward_fn
