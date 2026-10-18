from dataclasses import dataclass
from typing import Protocol


class LLMError(Exception):
    """The LLM call failed. `retryable` is True for rate limits and outages."""

    def __init__(self, message: str, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


@dataclass(frozen=True)
class LLMResult:
    text: str
    prompt_tokens: int | None
    completion_tokens: int | None


class LLMProvider(Protocol):
    """Anything that can turn (system instructions, prompt) into text."""

    name: str
    model_name: str

    def generate(self, system: str, prompt: str) -> LLMResult: ...