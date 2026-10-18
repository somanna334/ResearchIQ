from functools import lru_cache

from fastapi import HTTPException

from app.config import Settings, get_settings
from app.llm.base import LLMError, LLMProvider


def build_provider(settings: Settings) -> LLMProvider:
    """The one place that knows which provider classes exist."""
    if settings.llm_provider == "gemini":
        if settings.gemini_api_key is None:
            raise LLMError("RESEARCHIQ_GEMINI_API_KEY is not set in backend/.env")
        from app.llm.gemini import GeminiProvider

        return GeminiProvider(
            api_key=settings.gemini_api_key.get_secret_value(),
            model_name=settings.llm_model,
            max_output_tokens=settings.llm_max_output_tokens,
        )
    raise LLMError(f"Unknown LLM provider: {settings.llm_provider}")


@lru_cache
def get_llm_provider() -> LLMProvider:
    """FastAPI dependency. Tests replace it with a fake."""
    try:
        return build_provider(get_settings())
    except LLMError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc