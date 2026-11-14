"""Sample chat completions from seed problems and save benchmark records."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from mvb_rlvr.extractors import extract_final_answer
from mvb_rlvr.generators.stress import SeedProblem
from mvb_rlvr.io import ensure_dir, read_jsonl
from mvb_rlvr.prompts import build_math_messages
from mvb_rlvr.schema import BenchmarkRecord
from mvb_rlvr.verifiers.llm_judge import OpenAICompatibleJudgeVerifier


def detect_cot_language(text: str, fallback: str) -> str:
    """Infer jp, cn, or en from script counts; otherwise return fallback."""
    jp = sum(1 for ch in text if "\u3040" <= ch <= "\u30ff")
    cn = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    ascii_letters = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    if jp > 0:
        return "jp"
    if cn > max(jp, ascii_letters / 4):
        return "cn"
    if ascii_letters:
        return "en"
    return fallback


def sample_rollouts(config: dict[str, Any], root: Path, logger: logging.Logger) -> None:
    """Write sampled BenchmarkRecord JSONL and log API response metadata.

    Require seed_path, output_path, base_url, and model in config; resolve file
    paths against root. api_key_env names the credential environment variable.
    Sampling is sequential and flushes each record; opening the output replaces
    existing contents, so an interrupted run retains only completed records.
    """
    base_url = str(config["base_url"])
    model = str(config["model"])
    api_key = os.environ.get(str(config.get("api_key_env", "MODEL_API_KEY")), "")
    client = OpenAICompatibleJudgeVerifier(
        timeout_seconds=float(config.get("timeout_seconds", 120)),
        max_retries=int(config.get("max_retries", 3)),
    )
    samples = int(config.get("samples_per_problem", 4))
    max_tokens = int(config.get("max_tokens", 4096))
    if samples <= 0 or max_tokens <= 0:
        raise ValueError("samples_per_problem and max_tokens must be positive")
    seeds = [SeedProblem.from_dict(row) for row in read_jsonl(root / config["seed_path"])]
    if not seeds:
        raise ValueError("seed input is empty")
    destination = root / config["output_path"]
    ensure_dir(destination.parent)
    temperature = float(config.get("temperature", 0.7))
    completed = 0
    truncated = 0
    with destination.open("w", encoding="utf-8") as handle:
        for seed_index, seed in enumerate(seeds):
            messages = build_math_messages(
                seed.problem_text, seed.language, config.get("reasoning_mode", "direct")
            )
            for sample_index in range(samples):
                payload = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                response = client._post_json(base_url, api_key, payload)
                completion = client._extract_content(response)
                choice = response["choices"][0]
                finish_reason = choice.get("finish_reason")
                response_model = response.get("model")
                usage = response.get("usage", {})
                extracted, method = extract_final_answer(completion)
                rollout_id = f"rollout_{seed_index}_{sample_index}"
                record = BenchmarkRecord(
                    record_id=rollout_id,
                    problem_id=seed.problem_id,
                    language=seed.language,
                    source_dataset=seed.source_dataset,
                    problem_text=seed.problem_text,
                    gold_answer_canonical=seed.gold_answer,
                    candidate_answer=extracted,
                    candidate_label="ambiguous",
                    transformation_type="model_rollout",
                    expected_verdict="ambiguous",
                    raw_completion=completion,
                    extracted_answer=extracted,
                    extraction_method=method,
                    rollout_source="model_api",
                    rollout_id=rollout_id,
                    model_name=str(response_model or model),
                    sampling_temperature=temperature,
                    cot_language=detect_cot_language(completion, seed.language),
                    metadata={
                        "requested_model": model,
                        "response_model": response_model,
                        "finish_reason": finish_reason,
                        "usage": usage,
                        "reasoning_mode": config.get("reasoning_mode", "direct"),
                    },
                )
                handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
                handle.flush()
                completed += 1
                truncated += int(finish_reason == "length")
                logger.info(
                    "sample_complete record_id=%s response_model=%s finish_reason=%s usage=%s",
                    rollout_id, response_model, finish_reason, usage,
                )
    logger.info(
        "sampling_complete records=%d truncated=%d truncation_rate=%.6f",
        completed, truncated, truncated / completed,
    )
