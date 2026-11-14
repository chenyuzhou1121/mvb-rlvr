"""Build exact, normalized, symbolic, and API-backed answer verifiers."""

from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
import re
import signal
from typing import Any, Iterator, Union

from mvb_rlvr.canonicalize import answers_equivalent, canonicalize_answer
from mvb_rlvr.schema import BenchmarkRecord, VerifierVerdict
from mvb_rlvr.verifiers.base import Verifier


class ExactMatchVerifier(Verifier):
    """Accept candidates whose stripped text equals the reference text."""

    name = "rule_exact_match"

    def verify(self, record: BenchmarkRecord) -> VerifierVerdict:
        """Return accept or reject with the stripped answer strings."""
        verdict = "accept" if record.candidate_answer.strip() == record.gold_answer_canonical.strip() else "reject"
        return VerifierVerdict(
            verifier_name=self.name,
            verdict=verdict,
            confidence=1.0,
            normalized_gold=record.gold_answer_canonical.strip(),
            normalized_candidate=record.candidate_answer.strip(),
            matched_answer=record.candidate_answer if verdict == "accept" else None,
            normalization_trace=["raw_exact_match"] if verdict == "accept" else [],
        )


class NormalizedRuleVerifier(Verifier):
    """Compare candidates through multilingual text, numeric, and unit rules."""

    name = "rule_normalized_numeric"

    def verify(self, record: BenchmarkRecord) -> VerifierVerdict:
        """Return the equivalence verdict and normalized answer details."""
        try:
            equivalent, gold, candidate, trace = answers_equivalent(
                record.gold_answer_canonical, record.candidate_answer
            )
        except Exception as exc:
            gold = canonicalize_answer(record.gold_answer_canonical)
            candidate = canonicalize_answer(record.candidate_answer)
            return VerifierVerdict(
                verifier_name=self.name,
                verdict="error",
                confidence=0.0,
                normalized_gold=gold.text,
                normalized_candidate=candidate.text,
                error=repr(exc),
            )
        return VerifierVerdict(
            verifier_name=self.name,
            verdict="accept" if equivalent else "reject",
            confidence=1.0,
            normalized_gold=gold.text,
            normalized_candidate=candidate.text,
            matched_answer=record.candidate_answer if equivalent else None,
            normalization_trace=trace,
        )


class HybridRuleVerifier(Verifier):
    """Accept exact matches first, then use normalized answer equivalence."""

    name = "hybrid_rule_exact_then_numeric"

    def __init__(self) -> None:
        """Create the exact and normalized verifiers used in sequence."""
        self.exact = ExactMatchVerifier()
        self.normalized = NormalizedRuleVerifier()

    def verify(self, record: BenchmarkRecord) -> VerifierVerdict:
        """Return the first acceptance or the normalized fallback verdict."""
        exact = self.exact.verify(record)
        if exact.verdict == "accept":
            return VerifierVerdict(
                verifier_name=self.name,
                verdict="accept",
                confidence=exact.confidence,
                normalized_gold=exact.normalized_gold,
                normalized_candidate=exact.normalized_candidate,
                matched_answer=exact.matched_answer,
                normalization_trace=["hybrid_exact_accept", *exact.normalization_trace],
            )

        fallback = self.normalized.verify(record)
        return VerifierVerdict(
            verifier_name=self.name,
            verdict=fallback.verdict,
            confidence=fallback.confidence,
            normalized_gold=fallback.normalized_gold,
            normalized_candidate=fallback.normalized_candidate,
            matched_answer=fallback.matched_answer,
            error=fallback.error,
            normalization_trace=["hybrid_exact_reject", *fallback.normalization_trace],
        )


def _render_parsed_answer(parsed: object) -> str:
    return repr(parsed)


class MathVerifyVerifier(Verifier):
    """Compare answers using the optional math_verify parser and verifier."""

    name = "math_verify"

    def __init__(self) -> None:
        """Load optional dependencies or raise an installation error."""
        try:
            from math_verify import parse, verify
        except ImportError as exc:
            raise RuntimeError(
                "math_verify requires the optional public verifier dependencies; "
                "install with `pip install '.[public-verifiers]'`."
            ) from exc
        self._parse = parse
        self._verify = verify

    @lru_cache(maxsize=65536)
    def _parse_answer(self, answer: str) -> object:
        return self._parse(answer)

    def verify(self, record: BenchmarkRecord) -> VerifierVerdict:
        """Parse both answers and return a comparison or parsing error."""
        try:
            gold = self._parse_answer(record.gold_answer_canonical)
            candidate = self._parse_answer(record.candidate_answer)
            equivalent = bool(gold and candidate and self._verify(gold, candidate))
        except Exception as exc:
            return VerifierVerdict(
                verifier_name=self.name,
                verdict="error",
                confidence=0.0,
                normalized_gold=record.gold_answer_canonical.strip(),
                normalized_candidate=record.candidate_answer.strip(),
                error=repr(exc),
                normalization_trace=["math_verify_default_parse_error"],
            )
        return VerifierVerdict(
            verifier_name=self.name,
            verdict="accept" if equivalent else "reject",
            confidence=1.0,
            normalized_gold=_render_parsed_answer(gold),
            normalized_candidate=_render_parsed_answer(candidate),
            matched_answer=record.candidate_answer if equivalent else None,
            normalization_trace=["math_verify_default_parse", "math_verify_strict"],
        )


_MINERVA_SUBSTITUTIONS = (
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    (r"\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (r",\text{and}", ","),
    (r"\text{and}", ","),
    (r"\text{m}", r"\text{}"),
)
_MINERVA_REMOVED_EXPRESSIONS = (
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "ft",
    "hours",
    "km",
    "units",
    r"\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    r"\text{s}",
    r"\text{.}",
    "\\text{\ns}",
    r"\text{}^2",
    r"\text{}^3",
    "\\text{\n}",
    r"\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    r"\dots",
)


def _minerva_normalize_final_answer(answer: str) -> str:
    # Keep the final equality term, then apply literal and LaTeX replacements.
    normalized = answer.split("=")[-1]
    for before, after in _MINERVA_SUBSTITUTIONS:
        normalized = normalized.replace(before, after)
    for expression in _MINERVA_REMOVED_EXPRESSIONS:
        normalized = normalized.replace(expression, "")
    normalized = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", r"$\3$", normalized)
    normalized = re.sub(r"(\\text\{)(.*?)(\})", r"\2", normalized)
    normalized = re.sub(r"(\\textbf\{)(.*?)(\})", r"\2", normalized)
    normalized = re.sub(r"(\\overline\{)(.*?)(\})", r"\2", normalized)
    normalized = re.sub(r"(\\boxed\{)(.*)(\})", r"\2", normalized)
    normalized = re.sub(r"(frac)([^{])(.)", r"frac{\2}{\3}", normalized)
    normalized = re.sub(r"(sqrt)([^{])", r"sqrt{\2}", normalized)
    normalized = normalized.replace("$", "")
    if normalized.replace(",", "").isdigit():
        normalized = normalized.replace(",", "")
    return normalized


def _fix_fracs(answer: str) -> str:
    # Add braces to compact fractions while preserving an incomplete suffix.
    pieces = answer.split(r"\frac")
    output = pieces[0]
    for piece in pieces[1:]:
        output += r"\frac"
        if not piece or piece[0] == "{":
            output += piece
            continue
        if len(piece) < 2:
            return answer
        numerator, denominator = piece[0], piece[1]
        suffix = piece[2:]
        if denominator == "{":
            output += "{" + numerator + "}" + denominator + suffix
        else:
            output += "{" + numerator + "}{" + denominator + "}" + suffix
    return output


def _fix_integer_slash_fraction(answer: str) -> str:
    pieces = answer.split("/")
    if len(pieces) != 2:
        return answer
    try:
        numerator, denominator = (int(piece) for piece in pieces)
    except ValueError:
        return answer
    if answer != f"{numerator}/{denominator}":
        return answer
    return rf"\frac{{{numerator}}}{{{denominator}}}"


def _fix_sqrt(answer: str) -> str:
    if r"\sqrt" not in answer:
        return answer
    pieces = answer.split(r"\sqrt")
    output = pieces[0]
    for piece in pieces[1:]:
        if not piece:
            return answer
        output += r"\sqrt"
        output += piece if piece[0] == "{" else "{" + piece[0] + "}" + piece[1:]
    return output


def _hendrycks_strip_string(answer: str) -> str:
    normalized = answer.replace("\n", "").replace(r"\!", "").replace(r"\\", "\\")
    normalized = normalized.replace("tfrac", "frac").replace("dfrac", "frac")
    normalized = normalized.replace(r"\left", "").replace(r"\right", "")
    normalized = normalized.replace(r"^{\circ}", "").replace(r"^\circ", "")
    normalized = normalized.replace(r"\$", "")
    if r"\text{ " in normalized:
        pieces = normalized.split(r"\text{ ")
        if len(pieces) == 2:
            normalized = pieces[0]
    normalized = normalized.replace(r"\%", "")
    normalized = normalized.replace(" .", " 0.").replace("{.", "{0.")
    if not normalized:
        return normalized
    if normalized[0] == ".":
        normalized = "0" + normalized
    pieces = normalized.split("=")
    if len(pieces) == 2 and len(pieces[0]) <= 2:
        normalized = pieces[1]
    normalized = _fix_sqrt(normalized).replace(" ", "")
    normalized = _fix_fracs(normalized)
    if normalized == "0.5":
        normalized = r"\frac{1}{2}"
    return _fix_integer_slash_fraction(normalized)


class _MinervaTimeout(TimeoutError):
    pass


@contextmanager
def _minerva_timeout(seconds: int) -> Iterator[None]:
    """Bound symbolic comparison with SIGALRM where the platform supports it."""
    if not hasattr(signal, "SIGALRM"):
        yield
        return
    previous_handler = signal.getsignal(signal.SIGALRM)

    def handle_timeout(_signum: int, _frame: object) -> None:
        raise _MinervaTimeout(f"symbolic comparison exceeded {seconds}s")

    signal.signal(signal.SIGALRM, handle_timeout)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)


class MinervaIsEquivVerifier(Verifier):
    """Normalize answer strings, then compare their symbolic difference."""

    name = "minerva_is_equiv"

    def __init__(self) -> None:
        """Load SymPy LaTeX parsing and require its antlr4 runtime."""
        try:
            import sympy
            from sympy.parsing.latex import parse_latex
        except ImportError as exc:
            raise RuntimeError(
                "minerva_is_equiv requires the optional public verifier dependencies; "
                "install with `pip install '.[public-verifiers]'`."
            ) from exc
        try:
            import antlr4
        except ImportError as exc:
            raise RuntimeError(
                "minerva_is_equiv requires antlr4 for SymPy LaTeX parsing; "
                "install with `pip install '.[public-verifiers]'`."
            ) from exc
        self._sympy = sympy
        self._parse_latex = parse_latex

    @staticmethod
    @lru_cache(maxsize=65536)
    def _normalize(answer: str) -> str:
        return _hendrycks_strip_string(_minerva_normalize_final_answer(answer))

    @lru_cache(maxsize=65536)
    def _parse_answer(self, answer: str) -> object:
        return self._parse_latex(answer)

    def verify(self, record: BenchmarkRecord) -> VerifierVerdict:
        """Return a normalized or symbolic match; reject parse errors/timeouts.

        Symbolic comparison has a five-second alarm on SIGALRM platforms and
        must run in the main thread on those platforms.
        """
        gold = self._normalize(record.gold_answer_canonical)
        candidate = self._normalize(record.candidate_answer)
        equivalent = gold == candidate
        rationale = "minerva_hendrycks_normalized_exact" if equivalent else "minerva_sympy_reject"
        if not equivalent:
            try:
                with _minerva_timeout(5):
                    parsed_gold = self._parse_answer(gold)
                    parsed_candidate = self._parse_answer(candidate)
                    equivalent = bool(self._sympy.simplify(parsed_gold - parsed_candidate) == 0)
                if equivalent:
                    rationale = "minerva_sympy_simplify"
            except _MinervaTimeout:
                rationale = "minerva_sympy_timeout_reject"
            except Exception as exc:
                rationale = f"minerva_parse_or_compare_reject:{type(exc).__name__}"
        return VerifierVerdict(
            verifier_name=self.name,
            verdict="accept" if equivalent else "reject",
            confidence=1.0,
            normalized_gold=gold,
            normalized_candidate=candidate,
            matched_answer=record.candidate_answer if equivalent else None,
            normalization_trace=["minerva_appendix_d", "hendrycks_math_strip", rationale],
            rationale=rationale,
        )


VerifierSpec = Union[str, dict[str, Any]]


def normalize_verifier_spec(spec: VerifierSpec) -> tuple[str, dict[str, Any]]:
    """Return a verifier name and options from a name or a config object."""
    if isinstance(spec, str):
        return spec, {}
    if isinstance(spec, dict):
        name = spec.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"verifier config must contain a non-empty string name: {spec!r}")
        return name, {key: value for key, value in spec.items() if key != "name"}
    raise TypeError(f"verifier spec must be a string or object, got {type(spec).__name__}")


def build_verifier(spec: VerifierSpec) -> Verifier:
    """Instantiate a named verifier; only the API judge accepts extra options."""
    name, params = normalize_verifier_spec(spec)
    if name == ExactMatchVerifier.name:
        _reject_rule_params(name, params)
        return ExactMatchVerifier()
    if name == NormalizedRuleVerifier.name:
        _reject_rule_params(name, params)
        return NormalizedRuleVerifier()
    if name == HybridRuleVerifier.name:
        _reject_rule_params(name, params)
        return HybridRuleVerifier()
    if name == MathVerifyVerifier.name:
        _reject_rule_params(name, params)
        return MathVerifyVerifier()
    if name == MinervaIsEquivVerifier.name:
        _reject_rule_params(name, params)
        return MinervaIsEquivVerifier()
    if name == "llm_judge_openai_compatible":
        from mvb_rlvr.verifiers.llm_judge import OpenAICompatibleJudgeVerifier

        return OpenAICompatibleJudgeVerifier(**params)
    raise ValueError(f"unknown verifier: {name}")


def verifier_names_from_specs(specs: list[VerifierSpec]) -> list[str]:
    """Instantiate specifications and return their effective output names."""
    return [build_verifier(spec).name for spec in specs]


def _reject_rule_params(name: str, params: dict[str, Any]) -> None:
    if params:
        keys = ", ".join(sorted(params))
        raise ValueError(f"{name} does not accept verifier config keys: {keys}")
