"""Run with: python -m app.llm.check"""
import sys

from app.config import get_settings
from app.llm.base import LLMError
from app.llm.factory import build_provider


def main() -> int:
    settings = get_settings()
    if settings.gemini_api_key is None:
        print("RESEARCHIQ_GEMINI_API_KEY is not set in backend/.env")
        return 1

    from google import genai

    client = genai.Client(api_key=settings.gemini_api_key.get_secret_value())
    names = sorted(m.name for m in client.models.list() if "gemini" in m.name)
    print(f"{len(names)} Gemini models in the catalog (listed does NOT mean usable):")
    for name in names:
        print("  ", name)
    print(f"Configured model: {settings.llm_model!r}")

    try:
        result = build_provider(settings).generate(
            "Reply with a single word.", "Say: ready"
        )
    except LLMError as exc:
        print("Test call FAILED:", exc)
        return 1
    print("Test call OK:", repr(result.text), result.prompt_tokens, result.completion_tokens)
    return 0


if __name__ == "__main__":
    sys.exit(main())