# OpenRouter Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refresh dependencies to current versions under uv, and replace the three hard-coded LLM providers with an OpenRouter-only flow. The model list is live, filtered, and price-labelled.

**Architecture:** A new pure-logic module, `openrouter_models.py`, fetches, filters and prices the OpenRouter catalog. The Streamlit app uses it to build the model dropdown, then calls the chosen model through pydantic-ai's `OpenRouterModel`, with semaphore-bounded concurrency and exponential-backoff retries.

**Tech Stack:** Python 3.13 (uv-managed), Streamlit 1.64, pydantic-ai-slim[openrouter] 2.51, httpx, ebooklib 0.20, beautifulsoup4 4.15, lxml 6.1, pytest.

**Spec:** `docs/superpowers/specs/2026-09-27-openrouter-migration-design.md`

## Global Constraints

- `requires-python = ">=3.12"`, and the local `.venv` uses Python 3.13 created by `uv sync`.
- Direct deps only, with floors: `streamlit>=1.64`, `pydantic-ai-slim[openrouter]>=2.51`, `ebooklib>=0.20`, `beautifulsoup4>=4.15`, `lxml>=6.1`, `httpx>=0.28`. Dev group: `pytest`.
- `requirements.txt` is generated with `uv export --no-dev --no-hashes --no-emit-project -o requirements.txt`.
- Cost constants: `CHARS_PER_TOKEN = 4`, `PROMPT_OVERHEAD_TOKENS = 450`, `OUTPUT_TOKENS_PER_CHAPTER = 700`, `OUTPUT_HEADROOM_TOKENS = 2000`.
- Price slider: $0.10–$20.00 per M input tokens, default $1.00. Concurrency slider: 1–16, default 4. Max attempts: 4. Backoff is capped at 30 s.
- The API key goes directly to `OpenRouterProvider(api_key=...)` and is never written to `os.environ`.

---

### Task 1: uv project, current dependencies, venv

**Files:** modify `pyproject.toml`, `.gitignore`, `.devcontainer/devcontainer.json`; regenerate `requirements.txt`; create `uv.lock`.

- [ ] Replace `pyproject.toml` with:

```toml
[project]
name = "epub-summarizer"
version = "0.2.0"
description = "Streamlit app that inserts LLM-generated pre-reading primers into EPUB chapters via OpenRouter"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "streamlit>=1.64",
    "pydantic-ai-slim[openrouter]>=2.51",
    "ebooklib>=0.20",
    "beautifulsoup4>=4.15",
    "lxml>=6.1",
    "httpx>=0.28",
]

[dependency-groups]
dev = ["pytest>=8"]

[tool.uv]
package = false

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] Write `.python-version` containing `3.13`. Make sure `.venv/` is listed in `.gitignore`.
- [ ] Run `uv sync` and confirm that `.venv/Scripts/python --version` reports 3.13.x.
- [ ] Run `uv export --no-dev --no-hashes --no-emit-project -o requirements.txt`.
- [ ] Devcontainer: set the image to `mcr.microsoft.com/devcontainers/python:1-3.13-bookworm` and simplify `updateContentCommand` to `pip3 install --user -r requirements.txt`.
- [ ] Commit: `build: migrate to uv with current dependency versions`.

### Task 2: `openrouter_models.py` (TDD)

**Files:** create `openrouter_models.py`, `tests/test_openrouter_models.py`, `tests/__init__.py` (empty).

**Produces:**
- `ModelInfo(id, name, context_length, prompt_price, completion_price)`, a frozen dataclass. Prices are USD per token. It has the properties `is_free`, `input_per_million`, `output_per_million`.
- `fetch_models(timeout: float = 20.0) -> list[dict]`
- `required_context(chars_per_chapter: int) -> int`
- `filter_models(raw_models: list[dict], *, min_context: int, max_input_price: float, include_free: bool, today: date | None = None) -> list[ModelInfo]`
- `estimate_cost(model: ModelInfo, char_counts: list[int], chars_per_chapter: int) -> float`
- `tokens_cost(model: ModelInfo, input_tokens: int, output_tokens: int) -> float`
- `format_usd(amount: float) -> str`
- `format_label(model: ModelInfo, est_cost: float) -> str`

- [ ] Write the tests (see the test file in the repo) covering:
  - each exclusion rule: router `-1`, `:batch`, `~` alias, non-text output, image output, no structured/tools, expired, and the context floor;
  - the free toggle, price cap and sort order;
  - cost math;
  - USD formatting.
- [ ] Run `uv run pytest` and confirm the tests FAIL with an import error.
- [ ] Implement `openrouter_models.py`.
- [ ] Run `uv run pytest` and confirm the tests PASS.
- [ ] Commit: `feat: add OpenRouter model catalog filtering and cost estimation`.

### Task 3: Rewrite app for OpenRouter

**Files:** modify `summarize_epub_streamlit.py`.

**Consumes:** everything from Task 2.

- [ ] `EPUBSummaryInserter.__init__(self, model_id: str, api_key: str, chars_per_chapter: int)` builds `Agent(OpenRouterModel(model_id, provider=OpenRouterProvider(api_key=api_key, app_title="EPUB Summary Generator")), output_type=ChapterDigest)`.
- [ ] `get_chapter_summary(content, title, on_status)` returns `(formatted_digest, RunUsage)`. It makes `MAX_ATTEMPTS = 4` attempts with backoff `min(30, 2**attempt + random.uniform(0, 1))`, or `Retry-After` when it's present on a 429. On `ModelHTTPError` with status 401 or 402 it raises `FatalAPIError` immediately.
- [ ] `process_selected_chapters(selected, chapters, book, concurrency)` uses an `asyncio.Semaphore(concurrency)` and `asyncio.as_completed` to report progress per chapter. A `FatalAPIError` cancels the remaining tasks and shows a message. It returns `(bytes, summaries, ok, failed, input_tokens, output_tokens, reported_cost)`.
- [ ] UI: the provider/tier/buffer/batch code is replaced by:
  - the key input;
  - the length selector;
  - the max-price slider;
  - the free-models checkbox;
  - a "Refresh model list" button;
  - the model selectbox, whose labels come from `format_label` and are estimated for the selected chapters (or all chapters, or one chapter before upload);
  - a custom model-ID text input that overrides the dropdown when filled in;
  - the parallel-requests slider, forced to 1 for free models.
- [ ] After a run, show the actual cost: `usage.cost` if OpenRouter reported it, otherwise `tokens_cost`.
- [ ] Smoke-check by importing the module with `uv run python -c "import summarize_epub_streamlit"`.
- [ ] Commit: `feat: switch LLM calls to OpenRouter with live model catalog`.

### Task 4: README and end-to-end verification

- [ ] Rewrite the README's provider, tier, prerequisites, installation, usage and troubleshooting sections for OpenRouter and uv.
- [ ] Run `uv run streamlit run summarize_epub_streamlit.py --server.headless true`. In the browser pane, confirm that the model list loads with prices, and that the price slider and free toggle change the list.
- [ ] Commit: `docs: update README for OpenRouter and uv`.
