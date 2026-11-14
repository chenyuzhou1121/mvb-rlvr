"""Define serializable benchmark inputs and verifier outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Literal

Language = Literal["en", "jp", "cn"]
CandidateLabel = Literal["correct", "incorrect", "ambiguous"]
Verdict = Literal["accept", "reject", "ambiguous", "error"]


@dataclass(frozen=True)
class BenchmarkRecord:
    """Store one candidate, its reference answer, and optional rollout metadata.

    The first ten fields are required in record JSONL. Labels use ``correct``,
    ``incorrect``, or ``ambiguous``; expected verdicts use ``accept``, ``reject``,
    ``ambiguous``, or ``error``. Language tags used by prompts are en, jp, and cn.
    Annotations and types describe the format without validating field values.
    """

    record_id: str
    problem_id: str
    language: str
    source_dataset: str
    problem_text: str
    gold_answer_canonical: str
    candidate_answer: str
    candidate_label: CandidateLabel
    transformation_type: str
    expected_verdict: Verdict
    raw_completion: str | None = None
    extracted_answer: str | None = None
    extraction_method: str = "provided_candidate"
    rollout_source: str = "human_stress"
    rollout_id: str | None = None
    model_name: str | None = None
    checkpoint_step: str | None = None
    sampling_temperature: float | None = None
    cot_language: str | None = None
    proxy_reward: float | None = None
    trusted_reward: float | None = None
    human_audit_note: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return all record fields as a dictionary, including optional values."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BenchmarkRecord":
        """Require core fields, ignore extra keys, and default optional fields."""
        required = {
            "record_id",
            "problem_id",
            "language",
            "source_dataset",
            "problem_text",
            "gold_answer_canonical",
            "candidate_answer",
            "candidate_label",
            "transformation_type",
            "expected_verdict",
        }
        missing = sorted(required.difference(data))
        if missing:
            raise ValueError(f"benchmark record missing required fields: {missing}")
        allowed = {item.name for item in fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in allowed})


@dataclass(frozen=True)
class VerifierVerdict:
    """Store a decision, confidence, and optional normalization or error details.

    Decisions use accept, reject, ambiguous, or error. Confidence is a score
    supplied by the verifier; normalized values and traces describe its path.
    """

    verifier_name: str
    verdict: Verdict
    confidence: float
    normalized_gold: str | None = None
    normalized_candidate: str | None = None
    matched_answer: str | None = None
    normalization_trace: list[str] = field(default_factory=list)
    rationale: str | None = None
    error: str | None = None
    raw_output: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return all verdict fields as a dictionary."""
        return asdict(self)


def attach_verdict(record: BenchmarkRecord, verdict: VerifierVerdict) -> dict[str, Any]:
    """Return a record dictionary with one verdict keyed by verifier name."""
    data = record.to_dict()
    outputs = data.setdefault("verifier_outputs", {})
    outputs[verdict.verifier_name] = verdict.to_dict()
    return data
