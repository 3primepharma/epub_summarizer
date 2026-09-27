"""OpenRouter model catalog: fetch, filter, and price models for chapter summarization."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

import httpx

MODELS_URL = "https://openrouter.ai/api/v1/models"

# Rough heuristics used for context checks and cost estimates
CHARS_PER_TOKEN = 4
PROMPT_OVERHEAD_TOKENS = 450  # instructions wrapped around the chapter text
OUTPUT_TOKENS_PER_CHAPTER = 700  # typical structured digest size
OUTPUT_HEADROOM_TOKENS = 2000  # context reserved for the response

STRUCTURED_PARAMS = {"structured_outputs", "tools"}


@dataclass(frozen=True)
class ModelInfo:
    id: str
    name: str
    context_length: int
    prompt_price: float  # USD per input token
    completion_price: float  # USD per output token

    @property
    def is_free(self) -> bool:
        return self.id.endswith(":free") or (self.prompt_price == 0 and self.completion_price == 0)

    @property
    def input_per_million(self) -> float:
        return self.prompt_price * 1_000_000

    @property
    def output_per_million(self) -> float:
        return self.completion_price * 1_000_000


def fetch_models(timeout: float = 20.0) -> list[dict]:
    """Fetch the raw OpenRouter model catalog (no API key required)."""
    response = httpx.get(MODELS_URL, timeout=timeout)
    response.raise_for_status()
    return response.json()["data"]


def required_context(chars_per_chapter: int) -> int:
    """Minimum context window needed to summarize one chapter of the given size."""
    return chars_per_chapter // CHARS_PER_TOKEN + PROMPT_OVERHEAD_TOKENS + OUTPUT_HEADROOM_TOKENS


def _parse_model(raw: dict, today: date) -> ModelInfo | None:
    """Return a ModelInfo if the model is suitable for text summarization, else None."""
    model_id = raw.get("id", "")
    if model_id.startswith("~") or model_id.endswith(":batch"):
        return None

    arch = raw.get("architecture") or {}
    if arch.get("tokenizer") == "Router":  # meta-models that pick a model per request; unpredictable cost
        return None
    if "text" not in (arch.get("input_modalities") or []):
        return None
    if (arch.get("output_modalities") or []) != ["text"]:
        return None

    if not STRUCTURED_PARAMS & set(raw.get("supported_parameters") or []):
        return None

    expiration = raw.get("expiration_date")
    if expiration and date.fromisoformat(expiration) <= today:
        return None

    return _priced_model(raw)


def _priced_model(raw: dict) -> ModelInfo | None:
    """Build a ModelInfo if the model has fixed, non-negative per-token prices."""
    pricing = raw.get("pricing") or {}
    try:
        prompt_price = float(pricing["prompt"])
        completion_price = float(pricing["completion"])
    except (KeyError, TypeError, ValueError):
        return None
    if prompt_price < 0 or completion_price < 0:
        return None

    model_id = raw.get("id", "")
    return ModelInfo(
        id=model_id,
        name=raw.get("name") or model_id,
        context_length=int(raw.get("context_length") or 0),
        prompt_price=prompt_price,
        completion_price=completion_price,
    )


def filter_models(
    raw_models: list[dict],
    *,
    min_context: int,
    max_input_price: float,
    include_free: bool,
    today: date | None = None,
) -> list[ModelInfo]:
    """Filter the raw catalog to suitable models, cheapest first.

    max_input_price is in USD per million input tokens.
    """
    today = today or date.today()
    models = []
    for raw in raw_models:
        model = _parse_model(raw, today)
        if model is None or model.context_length < min_context:
            continue
        if model.is_free and not include_free:
            continue
        if model.input_per_million > max_input_price + 1e-9:
            continue
        models.append(model)
    return sorted(models, key=lambda m: (m.prompt_price, m.completion_price, m.id))


def find_model(raw_models: list[dict], model_id: str) -> ModelInfo | None:
    """Look up any catalog model by id (no suitability filtering), for user-entered model IDs."""
    for raw in raw_models:
        if raw.get("id") == model_id:
            return _priced_model(raw)
    return None


def tokens_cost(model: ModelInfo, input_tokens: int, output_tokens: int) -> float:
    return input_tokens * model.prompt_price + output_tokens * model.completion_price


def estimate_cost(model: ModelInfo, char_counts: list[int], chars_per_chapter: int) -> float:
    """Estimated USD cost to summarize chapters with the given character counts."""
    input_tokens = sum(
        min(chars, chars_per_chapter) // CHARS_PER_TOKEN + PROMPT_OVERHEAD_TOKENS for chars in char_counts
    )
    output_tokens = OUTPUT_TOKENS_PER_CHAPTER * len(char_counts)
    return tokens_cost(model, input_tokens, output_tokens)


def format_usd(amount: float) -> str:
    """Format dollars, keeping two significant digits for sub-cent amounts."""
    if 0 < amount < 0.01:
        decimals = -math.floor(math.log10(amount)) + 1
        return f"${amount:.{decimals}f}"
    return f"${amount:.2f}"


def format_label(model: ModelInfo, est_cost: float) -> str:
    if model.is_free:
        return f"{model.id} — free (rate-limited)"
    return (
        f"{model.id} — {format_usd(model.input_per_million)} in / "
        f"{format_usd(model.output_per_million)} out per M — est. {format_usd(est_cost)}"
    )
