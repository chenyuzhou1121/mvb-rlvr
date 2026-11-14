"""Build math prompts in English, Japanese, and Chinese."""

from __future__ import annotations

from typing import Literal

Language = Literal["en", "jp", "cn"]
PromptFormat = Literal["text", "messages"]
ReasoningMode = Literal["direct", "reasoning"]

SUPPORTED_LANGUAGES = {"en", "jp", "cn"}

DIRECT_TEMPLATES = {
    "en": (
        "/no_think\nSolve the following math problem. On the final line, "
        "write only 'Final answer: <answer>'.\n\n{problem_text}"
    ),
    "jp": "/no_think\n次の数学問題を解いてください。最後の行には「答えは <答え>」だけを書いてください。\n\n{problem_text}",
    "cn": "/no_think\n请解答下面的数学题。最后一行只写“答案是：<答案>”。\n\n{problem_text}",
}

REASONING_TEMPLATES = {
    "en": (
        "Solve the following math problem. Show your reasoning, then put the "
        "final answer on the last line as 'Final answer: "
        "\\boxed{{<answer>}}'.\n\n{problem_text}"
    ),
    "jp": "次の数学問題を解いてください。考え方を示し、最後の行には「答えは \\boxed{{<答え>}}」だけを書いてください。\n\n{problem_text}",
    "cn": "请解答下面的数学题。展示必要推理，并在最后一行只写“答案是：\\boxed{{<答案>}}”。\n\n{problem_text}",
}


def _validate_language(language: str) -> str:
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"unsupported language={language!r}; expected one of {sorted(SUPPORTED_LANGUAGES)}")
    return language


def build_math_prompt(
    problem_text: str,
    language: str,
    reasoning_mode: ReasoningMode = "direct",
) -> str:
    """Return a prompt for en, jp, or cn with direct or reasoning instructions.

    Direct prompts begin with the literal /no_think directive. Reasoning
    prompts request a boxed answer. Unsupported languages or modes raise.
    """
    language = _validate_language(language)
    templates = DIRECT_TEMPLATES if reasoning_mode == "direct" else REASONING_TEMPLATES
    if reasoning_mode not in ("direct", "reasoning"):
        raise ValueError("reasoning_mode must be 'direct' or 'reasoning'")
    return templates[language].format(problem_text=problem_text)


def build_math_messages(
    problem_text: str,
    language: str,
    reasoning_mode: ReasoningMode = "direct",
) -> list[dict[str, str]]:
    """Wrap the math prompt in a single user message for a chat API."""
    return [{"role": "user", "content": build_math_prompt(problem_text, language, reasoning_mode)}]


def build_prompt_field(
    problem_text: str,
    language: str,
    prompt_format: PromptFormat = "messages",
    reasoning_mode: ReasoningMode = "direct",
) -> str | list[dict[str, str]]:
    """Return prompt text or user messages according to ``prompt_format``."""
    if prompt_format == "text":
        return build_math_prompt(problem_text, language, reasoning_mode)
    if prompt_format == "messages":
        return build_math_messages(problem_text, language, reasoning_mode)
    raise ValueError("prompt_format must be 'text' or 'messages'")
