# mvb-rlvr

This package audits multilingual verifier bias in reinforcement learning with verifiable rewards (RLVR).
It provides answer-format stress tests, rollout diagnosis of language-conditioned reward errors, and label-free target-local cross-lingual answer selection.
Python reward callbacks provide rule-based, self-consistency, and cached crosslingual rewards.

Paper: *Multilingual Verifier Bias in RLVR: Benchmark, Rollout Diagnosis, and the Cross-Lingual Selection Bottleneck*. PRICAI 2026; [arXiv:2608.20362](https://arxiv.org/abs/2608.20362).

## Installation

Python 3.10 or newer is required. Default commands use the standard library.
Install from the repository root:

```sh
python3 -m pip install -e .
```

The optional `math_verify` and `minerva_is_equiv` verifiers use math-verify, SymPy, and the antlr4 runtime:

```sh
python3 -m pip install -e '.[public-verifiers]'
```

## Quick start

Create a UTF-8 seed JSONL file with one object per line, then run the bundled benchmark config:

```sh
mkdir -p data
cat > data/seeds.jsonl <<'EOF'
{"problem_id":"addition","language":"en","source_dataset":"example","problem_text":"What is 2 + 2?","gold_answer":"4"}
EOF
python3 -m mvb_rlvr benchmark --config configs/benchmark.json
```

All five seed fields shown above are required; answers are strings. Prompt language tags are `en`, `jp`, and `cn`.
Optional `incorrect_answer` overrides the generated near-miss answer; optional `metadata` is an object copied to stress records.
Translations must share `source_dataset` and `problem_id` to supply crosslingual support.
The benchmark writes candidates and verdicts as JSONL, a JSON summary, four grouped CSV tables, and a log under `outputs/`.
Use `--operation generate`, `--operation verify`, or `--operation summarize` to run one stage; the default is `all`.

## Sampling and answer selection

In `configs/rollout_sampling.json`, set `base_url` and `model` to your chat API and set the environment variable named by `api_key_env` outside the config.
The API accepts chat-completion requests and returns text in the first choice's `message.content`.

```sh
python3 -m mvb_rlvr sample --config configs/rollout_sampling.json
python3 -m mvb_rlvr diagnose --input outputs/rollouts.jsonl --output outputs/diagnosed.jsonl --summary outputs/diagnosis.json
python3 -m mvb_rlvr select --input outputs/diagnosed.jsonl --output outputs/selected.jsonl --strategy target_crosslingual --support-weight 1.0
python3 -m mvb_rlvr support-cache --input outputs/rollouts.jsonl --output outputs/support_cache.json
```

Sampling writes `BenchmarkRecord` JSONL, including raw completions, extracted answers, model names, and response metadata.
Required record fields are `record_id`, `problem_id`, `language`, `source_dataset`, `problem_text`, `gold_answer_canonical`, `candidate_answer`, `candidate_label`, `transformation_type`, and `expected_verdict`; optional fields are in `schema.py`.
Labels are `correct`, `incorrect`, or `ambiguous`; verdicts are `accept`, `reject`, `ambiguous`, or `error`.
Diagnosis replaces labels and `trusted_reward` with `answers_equivalent` results, attaches verifier rewards, and writes a grouped JSON summary.
Use `--verifiers` with comma-separated names and `--primary-verifier` to choose the verifier supplying `proxy_reward`; the primary must be listed.
Selection emits one JSONL row per model, dataset, problem, and language. `local_majority` counts target answers; `target_crosslingual` adds weighted other-language counts; `union_crosslingual` also admits other-language candidates.
Support caches contain per-model problem maps with per-language totals and cluster counts.
All commands accept `--root` (default `.`). Config and data paths resolve against this root, including paths inside configs. Diagnose, select, and support-cache accept `--log-file` (default `outputs/run.log`).

## Configuration

Both configs use `seed_path` for the input JSONL and `log_file` for file logging.
Outputs replace existing files; logs append. Sampling flushes each completed record.

| Benchmark key | Meaning |
| --- | --- |
| `seed_path` | Seed JSONL, initially `data/seeds.jsonl`. |
| `candidate_path` | Generated `BenchmarkRecord` JSONL destination and verification input. |
| `verdict_path` | Verified JSONL destination and summary input. |
| `summary_path` | Grouped metrics JSON destination. |
| `csv_dir` | Directory for language, transformation, source-language, and variance tables. |
| `log_file` | Progress log destination. |
| `progress_every` | Verification log interval in records, clamped to at least one. |
| `verifiers` | Names or objects with a `name` field and verifier constructor options. |
| `stress_types` | Ordered variant names; the bundled list enumerates supported transformations. Unavailable variants are skipped per seed. |

| Sampling key | Meaning |
| --- | --- |
| `seed_path` | Seed JSONL, initially `data/seeds.jsonl`. |
| `output_path` | Sampled record JSONL destination. |
| `log_file` | Progress and response metadata log destination. |
| `base_url` | Chat API root, such as `https://api.example.com/v1`, or full chat/completions URL. |
| `api_key_env` | Environment variable holding the API key; default `MODEL_API_KEY`. |
| `model` | Requested API model identifier. |
| `samples_per_problem` | Positive number of completions per seed; default 4. |
| `temperature` | Sampling temperature sent to the API; default 0.7. |
| `max_tokens` | Positive completion token budget; default 4096. |
| `reasoning_mode` | `direct` adds `/no_think` and requests a final line; `reasoning` requests reasoning and a boxed answer. |
| `timeout_seconds` | Timeout per API request; default 120. |
| `max_retries` | Total request attempts, including the first; default 3, with at least one attempt. |

Verifier names are `rule_exact_match`, `rule_normalized_numeric`, `hybrid_rule_exact_then_numeric`, `math_verify`, `minerva_is_equiv`, and `llm_judge_openai_compatible`.
Only the API judge accepts constructor options. A benchmark entry can be `{"name":"llm_judge_openai_compatible","model":"model-name"}`.
The judge reads `LLM_JUDGE_BASE_URL`, `LLM_JUDGE_API_KEY`, and `LLM_JUDGE_MODEL` by default.
`base_url_env`, `api_key_env`, and `model_env` override those variable names; explicit `base_url`, `api_key`, and `model` values take precedence. Keep credentials in the environment.
`output_name` or its alias `verifier_name` changes the output key, with `output_name` taking precedence; output keys must be distinct within a benchmark.
Judge request options are `timeout_seconds` (30), `max_retries` (3 total attempts), `temperature` (0), `seed` (42; null omits it), `max_tokens` (256), and `response_format` (true; false omits the JSON-object request).

## Python interfaces and layout

`training_rewards.build_rule_reward_fn`, `self_consistency.build_self_consistency_reward_fn`, and `crosslingual_consensus.build_crosslingual_consensus_reward_fn` return callbacks accepting `prompts`, `completions`, and aligned metadata columns.
Self-consistency uses contiguous groups of `num_generations`; crosslingual rewards load a support-cache JSON file. Their docstrings describe grouping, penalties, audit output, and reward clipping.

```text
configs/                 Benchmark and sampling JSON configurations
src/mvb_rlvr/cli.py       Command definitions and workflow dispatch
src/mvb_rlvr/schema.py    Seed-independent record and verdict formats
src/mvb_rlvr/generators/  Seed parsing and answer stress transformations
src/mvb_rlvr/verifiers/   Rule, symbolic, and chat API verifiers
src/mvb_rlvr/             Canonicalization, extraction, prompts, I/O, metrics,
                         sampling, diagnosis, selection, and reward callbacks
pyproject.toml           Package metadata, dependencies, and console entry point
```

## License

MIT. See [LICENSE](LICENSE).
