from app.llm.base import LLMError, LLMResult

_RETRYABLE_MARKERS = ("429", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE")


class GeminiProvider:
    name = "gemini"

    def __init__(self, api_key: str, model_name: str, max_output_tokens: int) -> None:
        # Imported here so tests run without constructing a real client.
        from google import genai
        from google.genai import types

        self._types = types
        self._client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self._max_output_tokens = max_output_tokens

    def generate(self, system: str, prompt: str) -> LLMResult:
        try:
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=self._types.GenerateContentConfig(
                    system_instruction=system,
                    temperature=0.0,
                    max_output_tokens=self._max_output_tokens,
                ),
            )
        except Exception as exc:
            message = str(exc)
            retryable = any(marker in message for marker in _RETRYABLE_MARKERS)
            raise LLMError(f"Gemini request failed: {message[:300]}", retryable) from exc

        text = (response.text or "").strip()
        if not text:
            # Safety blocks, or "thinking" tokens using up the output budget.
            raise LLMError("Gemini returned no text (blocked or empty response)")

        usage = getattr(response, "usage_metadata", None)
        return LLMResult(
            text=text,
            prompt_tokens=getattr(usage, "prompt_token_count", None),
            completion_tokens=getattr(usage, "candidates_token_count", None),
        )