"""Select answer clusters and build per-model multilingual support caches."""

from __future__ import annotations

from collections import Counter, defaultdict
from math import isfinite
from typing import Any

from mvb_rlvr.crosslingual_consensus import problem_cache_key
from mvb_rlvr.self_consistency import canonical_cluster_key


STRATEGIES = ("local_majority", "target_crosslingual", "union_crosslingual")


def answer_cluster(row: dict[str, Any]) -> str:
    """Return the canonical cluster of extracted text, falling back to candidate."""
    answer = str(row.get("extracted_answer") or row.get("candidate_answer") or "").strip()
    return canonical_cluster_key(answer)[0]


def select_cluster(
    target_rows: list[dict[str, Any]],
    support_rows: list[dict[str, Any]],
    *,
    strategy: str = "target_crosslingual",
    support_weight: float = 1.0,
) -> tuple[str, dict[str, Any]]:
    """Return a winning cluster and its score, counts, and borrowed flag.

    local_majority ranks target counts only. target_crosslingual adds weighted
    support counts to target candidates; union_crosslingual also admits support
    candidates. Break ties by target count, support count for crosslingual
    strategies, and first appearance, preferring target order. Ignore failed
    extraction clusters; return NO_CANDIDATE if none remain.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown selection strategy: {strategy}")
    if not isfinite(support_weight) or support_weight < 0:
        raise ValueError("support_weight must be finite and nonnegative")
    target_counts = Counter(answer_cluster(row) for row in target_rows)
    support_counts = Counter(answer_cluster(row) for row in support_rows)
    candidates = set(target_counts)
    if strategy == "union_crosslingual":
        candidates.update(support_counts)
    candidates.discard("EXTRACTION_FAILED")
    if not candidates:
        return "NO_CANDIDATE", {
            "score": 0.0, "target_count": 0, "support_count": 0, "borrowed": False
        }
    target_order = {key: index for index, key in enumerate(dict.fromkeys(
        answer_cluster(row) for row in target_rows
    ))}
    support_order = {key: index for index, key in enumerate(dict.fromkeys(
        answer_cluster(row) for row in support_rows
    ))}
    weight = 0.0 if strategy == "local_majority" else support_weight

    def rank(key: str) -> tuple[float, int, int, int]:
        order = target_order.get(key, len(target_order) + support_order.get(key, 0))
        return (
            target_counts[key] + weight * support_counts[key],
            target_counts[key],
            0 if strategy == "local_majority" else support_counts[key],
            -order,
        )

    selected = max(candidates, key=rank)
    return selected, {
        "score": rank(selected)[0],
        "target_count": target_counts[selected],
        "support_count": support_counts[selected],
        "borrowed": selected not in target_counts,
    }


def select_answers(
    rows: list[dict[str, Any]],
    *,
    strategy: str = "target_crosslingual",
    support_weight: float = 1.0,
) -> list[dict[str, Any]]:
    """Select one answer per model, dataset, problem, and target language.

    Other languages of the same group supply support. Rows need source_dataset,
    problem_id, language, and answer text; absent model_name becomes default.
    Add correctness flags when every group row has a trusted_reward value.
    """
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row.get("model_name") or "default"),
            str(row["source_dataset"]),
            str(row["problem_id"]),
        )
        groups[key].append(row)
    output: list[dict[str, Any]] = []
    for (model, source, problem), group in sorted(groups.items()):
        for language in sorted({str(row["language"]) for row in group}):
            target = [row for row in group if str(row["language"]) == language]
            support = [row for row in group if str(row["language"]) != language]
            selected, details = select_cluster(
                target, support, strategy=strategy, support_weight=support_weight
            )
            eligible = target + support if strategy == "union_crosslingual" else target
            selected_rows = [row for row in eligible if answer_cluster(row) == selected]
            item: dict[str, Any] = {
                "model_name": model,
                "source_dataset": source,
                "problem_id": problem,
                "language": language,
                "strategy": strategy,
                "support_weight": support_weight,
                "selected_cluster": selected,
                "selected_answer": str(
                    selected_rows[0].get("extracted_answer")
                    or selected_rows[0].get("candidate_answer") or ""
                ).strip() if selected_rows else None,
                **details,
            }
            if all(row.get("trusted_reward") is not None for row in group):
                item.update({
                    "selected_trusted_correct": any(
                        float(row["trusted_reward"]) >= 1.0 for row in selected_rows
                    ),
                    "target_contains_trusted_correct": any(
                        float(row["trusted_reward"]) >= 1.0 for row in target
                    ),
                    "union_contains_trusted_correct": any(
                        float(row["trusted_reward"]) >= 1.0 for row in group
                    ),
                })
            output.append(item)
    return output


def build_support_cache(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Count answer clusters by model, aligned problem, and language.

    Require the same identifying fields as select_answers. Return models and
    a default_model chosen from the first valid row. Language totals and
    cluster_counts include only valid answers.
    """
    models: dict[str, Any] = {}
    for row in rows:
        cluster = answer_cluster(row)
        if cluster == "EXTRACTION_FAILED":
            continue
        model = str(row.get("model_name") or "default")
        model_cache = models.setdefault(model, {"problems": {}})
        key = problem_cache_key(str(row["source_dataset"]), str(row["problem_id"]))
        problem = model_cache["problems"].setdefault(key, {"languages": {}})
        language = problem["languages"].setdefault(
            str(row["language"]), {"total": 0, "cluster_counts": {}}
        )
        language["total"] += 1
        counts = language["cluster_counts"]
        counts[cluster] = counts.get(cluster, 0) + 1
    return {"default_model": next(iter(models), None), "models": models}
