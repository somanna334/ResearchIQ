import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Pricing:
    currency: str
    input_per_million: float
    output_per_million: float


def load_pricing(path: Path, model: str) -> Pricing | None:
    """Prices for one model, or None when the file or the numbers are missing."""
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None
    entry = data.get("per_million_tokens", {}).get(model)
    if not isinstance(entry, dict):
        return None
    price_in, price_out = entry.get("input"), entry.get("output")
    if not isinstance(price_in, (int, float)) or not isinstance(price_out, (int, float)):
        return None
    return Pricing(str(data.get("currency", "USD")), float(price_in), float(price_out))