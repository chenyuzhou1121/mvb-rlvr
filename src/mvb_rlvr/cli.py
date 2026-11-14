"""Provide commands for benchmark generation, sampling, diagnosis, and selection."""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Any

from mvb_rlvr.generators.stress import generate_records
from mvb_rlvr.io import ensure_dir, read_json, read_jsonl, write_json, write_jsonl
from mvb_rlvr.metrics import summarize
from mvb_rlvr.reward_diagnosis import diagnose_records, summarize_diagnosis
from mvb_rlvr.sampling import sample_rollouts
from mvb_rlvr.selection import STRATEGIES, build_support_cache, select_answers
from mvb_rlvr.schema import BenchmarkRecord
from mvb_rlvr.verifiers.rule import build_verifier, verifier_names_from_specs


def setup_logging(log_file: Path) -> logging.Logger:
    """Configure INFO logging to a UTF-8 file and the console; return the logger."""
    ensure_dir(log_file.parent)
    logger = logging.getLogger("mvb_rlvr")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_config(path: Path) -> dict[str, Any]:
    """Read a JSON configuration object or raise if its root is another type."""
    config = read_json(path)
    if not isinstance(config, dict):
        raise ValueError("config must be a JSON object")
    return config


def generate_candidates(config: dict[str, Any], root: Path, logger: logging.Logger) -> Path:
    """Read seeds, write configured stress variants, and return candidate_path.

    Require seed_path, candidate_path, and stress_types; resolve paths from root.
    """
    seed_path = _resolve(root, config["seed_path"])
    output_path = _resolve(root, config["candidate_path"])
    stress_types = config["stress_types"]
    logger.info("generate_start seed_path=%s output_path=%s stress_types=%s", seed_path, output_path, stress_types)
    seed_rows = read_jsonl(seed_path)
    records = generate_records(seed_rows, stress_types)
    write_jsonl(output_path, [record.to_dict() for record in records])
    logger.info("generate_done records=%d", len(records))
    return output_path


def verify_candidates(config: dict[str, Any], root: Path, logger: logging.Logger) -> Path:
    """Apply configured verifiers and return the written verdict_path.

    Require candidate_path, verdict_path, and verifiers. progress_every controls
    logging frequency. Paths are relative to root unless absolute.
    """
    candidate_path = _resolve(root, config["candidate_path"])
    output_path = _resolve(root, config["verdict_path"])
    verifier_specs = config["verifiers"]
    verifiers = [build_verifier(spec) for spec in verifier_specs]
    verifier_names = [verifier.name for verifier in verifiers]
    progress_every = max(1, int(config.get("progress_every", 50)))
    logger.info(
        "verify_start candidate_path=%s output_path=%s verifiers=%s progress_every=%d",
        candidate_path,
        output_path,
        verifier_names,
        progress_every,
    )
    output_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(read_jsonl(candidate_path), start=1):
        record = BenchmarkRecord.from_dict(row)
        out = record.to_dict()
        out["verifier_outputs"] = {}
        for verifier in verifiers:
            verdict = verifier.verify(record)
            out["verifier_outputs"][verdict.verifier_name] = verdict.to_dict()
        output_rows.append(out)
        if idx % progress_every == 0:
            logger.info("verify_progress records=%d", idx)
    write_jsonl(output_path, output_rows)
    logger.info("verify_done records=%d", len(output_rows))
    return output_path


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_verdicts(config: dict[str, Any], root: Path, logger: logging.Logger) -> Path:
    """Write summary JSON and four grouped CSV tables; return summary_path.

    Require verdict_path, summary_path, csv_dir, and verifiers. Resolve paths
    from root; verifier specifications determine the output names to summarize.
    """
    verdict_path = _resolve(root, config["verdict_path"])
    summary_path = _resolve(root, config["summary_path"])
    csv_dir = _resolve(root, config["csv_dir"])
    verifier_names = verifier_names_from_specs(config["verifiers"])
    logger.info("summarize_start verdict_path=%s summary_path=%s", verdict_path, summary_path)
    rows = read_jsonl(verdict_path)
    summary = summarize(rows, verifier_names)
    write_json(summary_path, summary)
    _write_csv(csv_dir / "by_language.csv", summary["by_language"])
    _write_csv(csv_dir / "by_transformation.csv", summary["by_transformation"])
    _write_csv(csv_dir / "by_source_language.csv", summary["by_source_language"])
    _write_csv(csv_dir / "equivalent_transformation_variance.csv", summary["equivalent_transformation_variance"])
    logger.info("summarize_done summary_path=%s", summary_path)
    return summary_path


def run(args: argparse.Namespace) -> int:
    """Dispatch parsed CLI arguments and return zero on success, one on failure.

    Resolve config, input, output, and log paths against args.root. Configuration
    loading and logger setup errors propagate before command dispatch.
    """
    root = Path(args.root).resolve()
    if args.command in {"benchmark", "sample"}:
        config = load_config(_resolve(root, args.config))
        log_path = _resolve(root, config.get("log_file", "outputs/run.log"))
    else:
        config = {}
        log_path = _resolve(root, args.log_file)
    logger = setup_logging(log_path)
    logger.info("command_start command=%s", args.command)
    try:
        if args.command == "benchmark":
            operations = {
                "generate": generate_candidates,
                "verify": verify_candidates,
                "summarize": summarize_verdicts,
            }
            selected = operations if args.operation == "all" else [args.operation]
            for operation in selected:
                operations[operation](config, root, logger)
        elif args.command == "sample":
            sample_rollouts(config, root, logger)
        else:
            rows = read_jsonl(_resolve(root, args.input))
            if args.command == "diagnose":
                names = [name.strip() for name in args.verifiers.split(",") if name.strip()]
                if not names:
                    raise ValueError("at least one verifier is required")
                primary = args.primary_verifier or names[0]
                if primary not in names:
                    raise ValueError("primary_verifier must be included in verifiers")
                annotated, details = diagnose_records(rows, names, primary)
                write_jsonl(_resolve(root, args.output), annotated)
                write_json(
                    _resolve(root, args.summary),
                    summarize_diagnosis(annotated, details, names, primary),
                )
            elif args.command == "select":
                selected = select_answers(
                    rows, strategy=args.strategy, support_weight=args.support_weight
                )
                write_jsonl(_resolve(root, args.output), selected)
            elif args.command == "support-cache":
                write_json(_resolve(root, args.output), build_support_cache(rows))
            logger.info("input_records=%d", len(rows))
    except Exception:
        logger.exception("command_failed command=%s", args.command)
        return 1
    logger.info("command_complete command=%s", args.command)
    return 0


def main() -> int:
    """Parse command-line arguments and return the selected command's exit code."""
    parser = argparse.ArgumentParser(
        description="Benchmark multilingual verifiers, diagnose rollouts, and select answers.",
        epilog=(
            "Seed JSONL fields: problem_id, language (en, jp, cn), source_dataset, "
            "problem_text, gold_answer; optional incorrect_answer. Record JSONL fields "
            "are defined by BenchmarkRecord in schema.py."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    benchmark = commands.add_parser("benchmark", help="Generate, verify, and summarize answer variants.")
    benchmark.add_argument("--config", required=True, help="Benchmark JSON configuration path.")
    benchmark.add_argument(
        "--operation", choices=["generate", "verify", "summarize", "all"], default="all",
        help="Operation to run; all runs generation, verification, and summary.",
    )
    sample = commands.add_parser("sample", help="Sample model rollouts from seed JSONL through a model API.")
    sample.add_argument("--config", required=True, help="Sampling JSON configuration path.")
    diagnose = commands.add_parser("diagnose", help="Compare proxy rewards with canonical answer equivalence.")
    diagnose.add_argument("--input", required=True, help="BenchmarkRecord JSONL path, such as sample output.")
    diagnose.add_argument("--output", required=True, help="Annotated record JSONL destination.")
    diagnose.add_argument("--summary", required=True, help="Grouped diagnostic JSON destination.")
    diagnose.add_argument(
        "--verifiers", default="rule_exact_match,rule_normalized_numeric",
        help=("Comma-separated verifier names: rule_exact_match, rule_normalized_numeric, "
              "hybrid_rule_exact_then_numeric, math_verify, minerva_is_equiv, "
              "llm_judge_openai_compatible."),
    )
    diagnose.add_argument("--primary-verifier", help="Verifier used for the proxy_reward field; defaults to first.")
    select = commands.add_parser("select", help="Select answer clusters within each model and aligned problem.")
    select.add_argument("--input", required=True, help="Rollout or diagnosed record JSONL path.")
    select.add_argument("--output", required=True, help="Selected answers and optional correctness JSONL destination.")
    select.add_argument("--strategy", choices=STRATEGIES, default="target_crosslingual", help="Candidate pool and ranking rule.")
    select.add_argument("--support-weight", type=float, default=1.0, help="Weight on other-language cluster counts.")
    cache = commands.add_parser("support-cache", help="Build a support cache for the crosslingual reward callback.")
    cache.add_argument("--input", required=True, help="Support rollout record JSONL path.")
    cache.add_argument("--output", required=True, help="Support cache JSON destination.")
    for command in (benchmark, sample, diagnose, select, cache):
        command.add_argument("--root", default=".", help="Base directory for relative configuration and data paths.")
    for command in (diagnose, select, cache):
        command.add_argument("--log-file", default="outputs/run.log", help="Progress log destination.")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
