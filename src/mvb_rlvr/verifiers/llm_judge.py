"""Judge answer equivalence through an OpenAI-compatible chat endpoint."""

from __future__ import annotations

import json
import os
import socket
import time
from typing import Any
import urllib.error
import urllib.request

from mvb_rlvr.schema import BenchmarkRecord, VerifierVerdict
from mvb_rlvr.verifiers.base import Verifier


class OpenAICompatibleJudgeVerifier(Verifier):
    """Request JSON verdicts and convert API or parsing failures to errors."""

    name = "llm_judge_openai_compatible"

    def __init__(
        self,
        *,
        output_name: str | None = None,
        verifier_name: str | None = None,
        base_url: str | None = None,
        base_url_env: str = "LLM_JUDGE_BASE_URL",
        api_key: str | None = None,
        api_key_env: str = "LLM_JUDGE_API_KEY",
        model: str | None = None,
        model_env: str = "LLM_JUDGE_MODEL",
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
        temperature: float = 0.0,
        seed: int | None = 42,
        max_tokens: int = 256,
        response_format: bool = True,
    ) -> None:
        """Configure request settings and environment-based credentials.

        Explicit base_url, api_key, and model override their *_env lookups at
        verification time. base_url is an API root or chat/completions URL.
        output_name takes precedence over the verifier_name output alias.
        max_retries is the total attempt count, with at least one attempt;
        timeout_seconds applies to each request. temperature, seed, max_tokens,
        and response_format control the chat payload. None omits the seed;
        response_format=False omits the JSON-object response request.
        """
        self.name = output_name or verifier_name or self.name
        self.base_url = base_url
        self.base_url_env = base_url_env
        self.api_key = api_key
        self.api_key_env = api_key_env
        self.model = model
        self.model_env = model_env
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = int(max_retries)
        self.temperature = float(temperature)
        self.seed = seed
        self.max_tokens = int(max_tokens)
        self.response_format = bool(response_format)

    def verify(self, record: BenchmarkRecord) -> VerifierVerdict:
        """Return the judge's verdict and normalized answers, or an error."""
        base_url = self.base_url or os.environ.get(self.base_url_env, "")
        api_key = self.api_key or os.environ.get(self.api_key_env, "")
        model = self.model or os.environ.get(self.model_env, "")
        if not base_url:
            return self._error("missing_base_url", record)
        if not api_key:
            return self._error("missing_api_key", record)
        if not model:
            return self._error("missing_model", record)

        payload = self._build_payload(model, record)
        try:
            response = self._post_json(base_url, api_key, payload)
            content = self._extract_content(response)
            parsed = self._parse_judge_json(content)
        except Exception as exc:
            return self._error(f"{type(exc).__name__}: {exc}", record)

        verdict = str(parsed.get("verdict", "")).strip().lower()
        if verdict not in {"accept", "reject", "ambiguous"}:
            return self._error(f"invalid_verdict: {verdict!r}", record, raw_output=json.dumps(parsed))

        confidence = _clamp_float(parsed.get("confidence", 0.0), 0.0, 1.0)
        rationale = _truncate(str(parsed.get("rationale", "")).strip(), 1000) or None
        matched = record.candidate_answer if verdict == "accept" else None
        return VerifierVerdict(
            verifier_name=self.name,
            verdict=verdict,
            confidence=confidence,
            normalized_gold=str(parsed.get("normalized_gold") or record.gold_answer_canonical),
            normalized_candidate=str(parsed.get("normalized_candidate") or record.candidate_answer),
            matched_answer=matched,
            normalization_trace=["llm_judge_openai_compatible"],
            rationale=rationale,
            raw_output=json.dumps(parsed, ensure_ascii=False, sort_keys=True),
        )

    def _build_payload(self, model: str, record: BenchmarkRecord) -> dict[str, Any]:
        user = {
            "problem_language": record.language,
            "source_dataset": record.source_dataset,
            "problem": record.problem_text,
            "gold_answer": record.gold_answer_canonical,
            "candidate_answer": record.candidate_answer,
        }
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a multilingual math-answer verifier. Decide whether the "
                        "candidate answer is mathematically equivalent to the gold answer. "
                        "Return only a JSON object with keys verdict, confidence, rationale, "
                        "normalized_gold, normalized_candidate. verdict must be one of "
                        "accept, reject, ambiguous. Do not include markdown or chain-of-thought."
                    ),
                },
                {"role": "user", "content": json.dumps(user, ensure_ascii=False, sort_keys=True)},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.seed is not None:
            payload["seed"] = self.seed
        if self.response_format:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def _post_json(self, base_url: str, api_key: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Post a chat payload with bounded retries for transient failures."""
        url = _chat_completions_url(base_url)
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        # The configured count includes the initial request.
        attempts = max(1, self.max_retries)
        last_error: Exception | None = None
        for attempt in range(attempts):
            request = urllib.request.Request(url, data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code not in {408, 409, 425, 429, 500, 502, 503, 504}:
                    detail = _truncate(exc.read().decode("utf-8", errors="replace"), 500)
                    raise RuntimeError(f"http_{exc.code}: {detail}") from exc
            except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
                last_error = exc
            if attempt + 1 < attempts:
                time.sleep(min(8.0, 2.0**attempt))
        raise RuntimeError(f"retry_exhausted: {last_error!r}")

    def _extract_content(self, response: dict[str, Any]) -> str:
        """Read nonempty text from the first choice or raise with usage details."""
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("missing choices")
        message = choices[0].get("message", {})
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content
        usage = response.get("usage", {})
        raise ValueError(
            f"empty message.content; finish_reason={choices[0].get('finish_reason')!r}; "
            f"completion_tokens={usage.get('completion_tokens')}; "
            f"response_model={response.get('model')!r}"
        )

    def _parse_judge_json(self, content: str) -> dict[str, Any]:
        """Decode an object, tolerating code fences and surrounding prose."""
        text = _strip_code_fence(content.strip())
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start < 0 or end <= start:
                raise
            parsed = json.loads(text[start : end + 1])
        if not isinstance(parsed, dict):
            raise ValueError("judge output is not a JSON object")
        return parsed

    def _error(
        self,
        message: str,
        record: BenchmarkRecord,
        raw_output: str | None = None,
    ) -> VerifierVerdict:
        return VerifierVerdict(
            verifier_name=self.name,
            verdict="error",
            confidence=0.0,
            normalized_gold=record.gold_answer_canonical,
            normalized_candidate=record.candidate_answer,
            error=message,
            raw_output=raw_output,
        )


def _chat_completions_url(base_url: str) -> str:
    cleaned = base_url.rstrip("/")
    if cleaned.endswith("/chat/completions"):
        return cleaned
    return f"{cleaned}/chat/completions"


def _strip_code_fence(text: str) -> str:
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _clamp_float(value: Any, low: float, high: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = low
    return min(high, max(low, parsed))


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."
