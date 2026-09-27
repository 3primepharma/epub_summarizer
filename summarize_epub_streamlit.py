import streamlit as st
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from typing import Callable, List, NamedTuple, Optional
from dataclasses import dataclass, field
import os
import io
import random
import tempfile
import asyncio
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.openrouter import OpenRouterProvider
from pydantic_ai.usage import RunUsage

from openrouter_models import (
    ModelInfo,
    estimate_cost,
    fetch_models,
    filter_models,
    find_model,
    format_label,
    format_usd,
    required_context,
    tokens_cost,
)

APP_TITLE = "EPUB Summary Generator"
MIN_CHAPTER_CHARS = 800
MAX_ATTEMPTS = 4
MAX_BACKOFF_SECONDS = 30
# Status codes that will fail identically for every chapter, so retrying is pointless
FATAL_STATUS_CODES = {401: "Invalid OpenRouter API key.",
                      402: "Insufficient OpenRouter credits for this model.",
                      404: "Model not found on OpenRouter."}
# Preferred defaults, in order; the first one that survives the filters is pre-selected
DEFAULT_MODEL_PREFERENCES = (
    "openai/gpt-6-luna",
    "google/gemini-3.1-flash-lite",
    "deepseek/deepseek-v4-flash",
)


class ChapterDigest(BaseModel):
    """Structured output model for chapter pre-reads"""
    content_type: str = Field(description="Detected type: fiction, non-fiction, news, memoir, or technical")
    at_a_glance: str = Field(description="2-3 sentence orientation to the chapter")
    key_concepts: List[str] = Field(description="3-4 key terms, ideas, or themes to prime the reader")
    questions_to_hold: List[str] = Field(description="2-3 questions to consider while reading")
    points_of_tension: List[str] = Field(description="2-3 areas of complexity, debate, or narrative tension")


class Chapter(NamedTuple):
    id: str
    title: str
    content: str
    char_count: int


class FatalAPIError(Exception):
    """An API error that makes further requests pointless (bad key, no credits, unknown model)."""


@dataclass
class RunResult:
    output_bytes: bytes
    summaries: List[tuple] = field(default_factory=list)  # (chapter index, title, summary)
    succeeded: int = 0
    failed: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    reported_cost: Optional[float] = 0.0  # None once any request lacks an OpenRouter-reported cost


def extract_chapters(epub_bytes: bytes) -> List[Chapter]:
    """Extract document items from an EPUB file."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.epub') as temp_input:
        temp_input.write(epub_bytes)
        temp_input_path = temp_input.name
    try:
        book = epub.read_epub(temp_input_path)
    finally:
        os.remove(temp_input_path)

    chapters = []
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_content(), 'html.parser')
            title = soup.find(['h1', 'h2'])
            title = title.get_text().strip() if title else "Untitled Chapter"
            chapters.append(Chapter(item.id, title, str(soup), len(soup.get_text())))
    return chapters


def _ensure_toc_uids(toc, counter=None) -> None:
    """Give every TOC entry a uid. ebooklib builds nav-derived TOC links without one,
    which makes write_epub fail when generating the NCX."""
    counter = counter if counter is not None else iter(range(1, 1_000_000))
    for entry in toc:
        if isinstance(entry, tuple):  # (Section, [children])
            section, children = entry
            _ensure_toc_uids([section], counter)
            _ensure_toc_uids(children, counter)
        elif getattr(entry, "uid", "") is None:
            entry.uid = f"toc-{next(counter)}"


def read_book(epub_bytes: bytes) -> epub.EpubBook:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.epub') as temp_input:
        temp_input.write(epub_bytes)
        temp_input_path = temp_input.name
    try:
        book = epub.read_epub(temp_input_path)
    finally:
        os.remove(temp_input_path)
    _ensure_toc_uids(book.toc)
    return book


class EPUBSummaryInserter:
    def __init__(self, model_id: str, api_key: str, chars_per_chapter: int):
        if not api_key:
            raise ValueError("API key is required")

        self.model_id = model_id
        self.chars_per_chapter = chars_per_chapter
        # Key is passed to the provider directly rather than via os.environ,
        # so concurrent Streamlit sessions can't see each other's keys.
        provider = OpenRouterProvider(api_key=api_key, app_title=APP_TITLE)
        self.agent = Agent(OpenRouterModel(model_id, provider=provider), output_type=ChapterDigest)

    def build_prompt(self, content: str) -> str:
        # Strip HTML tags for cleaner text
        text = BeautifulSoup(content, 'html.parser').get_text()

        return f"""Analyze this chapter and create a pre-reading primer to help the reader engage more deeply with the material.

<chapter>
{text[:self.chars_per_chapter]}
</chapter>

First, identify the content type (fiction/novel, non-fiction/educational, news/current events, personal essay/memoir, technical/documentation).

Then create a pre-read with these sections:

**At a Glance** (2-3 sentences)
A brief orientation: what is this chapter about and what kind of reading experience to expect.

**Key Concepts** (3-4 items)
Important terms, ideas, or themes the reader should be aware of. For fiction, this might be character dynamics or symbolic elements. For non-fiction, key vocabulary or frameworks.

**Questions to Hold** (2-3 questions)
Thought-provoking questions for the reader to keep in mind while reading. These should enhance engagement, not spoil content.

**Points of Tension** (2-3 items)
For non-fiction: competing perspectives, nuances, or areas of debate within the topic.
For fiction: conflicts, thematic tensions, or narrative questions being developed.
For news: context that adds depth or complexity to the reporting.
"""

    @staticmethod
    def format_digest(digest: ChapterDigest) -> str:
        key_concepts = "\n".join(f"• {c}" for c in digest.key_concepts)
        questions = "\n".join(f"• {q}" for q in digest.questions_to_hold)
        tensions = "\n".join(f"• {t}" for t in digest.points_of_tension)

        return f"""{digest.at_a_glance}

**Key Concepts**
{key_concepts}

**Questions to Hold**
{questions}

**Points of Tension**
{tensions}""".strip()

    async def get_chapter_summary(self, content: str, chapter_title: str,
                                  on_status: Callable[[str], None]) -> tuple[str, RunUsage]:
        """Generate a digest for one chapter, retrying transient failures with exponential backoff."""
        prompt = self.build_prompt(content)

        for attempt in range(1, MAX_ATTEMPTS + 1):
            retry_after = None
            try:
                result = await self.agent.run(prompt)
                return self.format_digest(result.output), result.usage
            except ModelHTTPError as e:
                if e.status_code in FATAL_STATUS_CODES:
                    raise FatalAPIError(f"{FATAL_STATUS_CODES[e.status_code]} ({e.body})") from e
                if attempt == MAX_ATTEMPTS:
                    raise
                if e.status_code == 429 and e.headers:
                    retry_after = e.headers.get('retry-after')
                error = e
            except Exception as e:
                if attempt == MAX_ATTEMPTS:
                    raise
                error = e

            try:
                delay = min(MAX_BACKOFF_SECONDS, float(retry_after))
            except (TypeError, ValueError):
                delay = min(MAX_BACKOFF_SECONDS, 2 ** attempt + random.uniform(0, 1))
            on_status(f"'{chapter_title}': attempt {attempt}/{MAX_ATTEMPTS} failed "
                      f"({type(error).__name__}); retrying in {delay:.0f}s...")
            await asyncio.sleep(delay)

    def insert_summary(self, html_content: str, summary: str) -> str:
        """Insert summary at the start of chapter content in an EPUB-friendly way"""
        soup = BeautifulSoup(html_content, 'html.parser')

        # Create summary div with semantic class names instead of inline styles
        summary_div = soup.new_tag('div')
        summary_div['class'] = 'chapter-digest'

        # Create a style tag for the head if it doesn't exist
        if not soup.find('style'):
            style_tag = soup.new_tag('style')
            style_tag.string = """
                .chapter-digest {
                    margin: 1em 0;
                    padding: 1em;
                    border: 1px solid currentColor;
                }
                .chapter-digest .section {
                    margin-bottom: 1em;
                }
                .chapter-digest .heading {
                    font-weight: bold;
                    margin-bottom: 0.5em;
                }
                .chapter-digest ul {
                    margin: 0;
                    padding-left: 1.5em;
                }
                .chapter-digest li {
                    margin-bottom: 0.3em;
                }
            """
            # Insert style in head, or create head if needed
            head = soup.find('head')
            if not head:
                head = soup.new_tag('head')
                if soup.html:
                    soup.html.insert(0, head)
                else:
                    html = soup.new_tag('html')
                    html.append(head)
                    soup.append(html)
            head.append(style_tag)

        # Split the summary into sections and format them
        sections = summary.split('\n\n')
        for section in sections:
            section_div = soup.new_tag('div')
            section_div['class'] = 'section'

            # Convert the text to semantic HTML
            lines = section.strip().split('\n')
            current_list = None

            for line in lines:
                if line.strip():
                    if line.startswith('•'):
                        # Create list if doesn't exist
                        if not current_list:
                            current_list = soup.new_tag('ul')
                            section_div.append(current_list)
                        li = soup.new_tag('li')
                        li.string = line[1:].strip()
                        current_list.append(li)
                    else:
                        current_list = None  # Reset list
                        p = soup.new_tag('p')
                        if any(heading in line for heading in ['Key Concepts', 'Questions to Hold', 'Points of Tension']):
                            p['class'] = 'heading'
                        p.string = line.replace('**', '')
                        section_div.append(p)

            summary_div.append(section_div)

        # Insert at start of body or main content
        body = soup.find('body') or soup
        body.insert(0, summary_div)

        return str(soup)

    async def process_selected_chapters(self, selected_indices: List[int], chapters: List[Chapter],
                                        book: epub.EpubBook, concurrency: int) -> Optional[RunResult]:
        """Summarize selected chapters concurrently and return the updated EPUB."""
        progress_bar = st.progress(0.0, text=f"0 / {len(selected_indices)} chapters")
        status_line = st.empty()
        semaphore = asyncio.Semaphore(concurrency)
        result = RunResult(output_bytes=b"")

        async def process_chapter(i: int):
            chapter = chapters[i]
            async with semaphore:
                try:
                    summary, usage = await self.get_chapter_summary(
                        chapter.content, chapter.title, status_line.info)
                    return i, summary, usage, None
                except FatalAPIError:
                    raise
                except Exception as e:
                    return i, None, None, e

        tasks = [asyncio.create_task(process_chapter(i)) for i in selected_indices]
        try:
            for done, next_task in enumerate(asyncio.as_completed(tasks), start=1):
                i, summary, usage, error = await next_task
                chapter = chapters[i]
                if error is not None:
                    result.failed += 1
                    st.warning(f"Failed to summarize '{chapter.title}' after {MAX_ATTEMPTS} attempts: {error}")
                else:
                    book.get_item_with_id(chapter.id).set_content(
                        self.insert_summary(chapter.content, summary).encode())
                    result.succeeded += 1
                    result.summaries.append((i, chapter.title, summary))
                    result.input_tokens += usage.input_tokens
                    result.output_tokens += usage.output_tokens
                    if usage.cost is None or result.reported_cost is None:
                        result.reported_cost = None
                    else:
                        result.reported_cost += float(usage.cost)
                progress_bar.progress(done / len(tasks), text=f"{done} / {len(tasks)} chapters")
        except FatalAPIError as e:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            status_line.empty()
            st.error(f"Stopped: {e}")
            return None

        status_line.empty()
        result.summaries.sort()  # reading order for the review section

        output = io.BytesIO()
        epub.write_epub(output, book)
        result.output_bytes = output.getvalue()
        return result


@st.cache_data(ttl=3600, show_spinner="Loading OpenRouter model list...")
def load_model_catalog() -> list[dict]:
    return fetch_models()


@st.cache_data(show_spinner="Reading EPUB...")
def load_chapters(epub_bytes: bytes) -> List[Chapter]:
    return extract_chapters(epub_bytes)


def default_model_index(models: List[ModelInfo]) -> int:
    ids = [m.id for m in models]
    for preferred in DEFAULT_MODEL_PREFERENCES:
        if preferred in ids:
            return ids.index(preferred)
    return 0


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")

    st.markdown("""
        <style>
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("📚 EPUB Summary Generator")

    st.markdown("""
        Enhance your EPUB files with AI-powered pre-reading primers, using any model available on
        [OpenRouter](https://openrouter.ai). Each chapter pre-read includes:
        - At a Glance: Brief orientation to the chapter
        - Key Concepts: Terms, ideas, and themes to prime your reading
        - Questions to Hold: Thought-provoking questions to consider while reading
        - Points of Tension: Areas of complexity, debate, or narrative tension

        **Generated pre-reads are integrated seamlessly to the start of each chapter to activate your thinking before diving into the material**

        ⚠️ **Important Usage Notes:**
        - Requires an [OpenRouter API key](https://openrouter.ai/keys); you are responsible for all API costs
        - Model prices are fetched live from OpenRouter; cost estimates are approximate
        - Maximum file size: 200MB
    """)

    # --- 1. Upload -----------------------------------------------------------
    st.markdown("### 1. Upload EPUB")
    uploaded_file = st.file_uploader(
        "Upload your EPUB file (max 200MB)",
        type=['epub'],
        help="Supported sources: Project Gutenberg, Instapaper, Calibre conversions, and more"
    )

    length_options = {
        "Short (1-2 pages, 4k chars)": 4000,
        "Medium (<15 pages, 20k chars)": 20000,
        "Long (15-30 pages, 40k chars)": 40000,
        "Long (30-50 pages, 100k chars)": 100000
    }
    selected_length = st.selectbox(
        "Text length per chapter",
        options=list(length_options.keys()),
        index=0,
        help="Only this many characters of each chapter are sent to the model"
    )
    chars_per_chapter = length_options[selected_length]

    # --- 2. Chapters -----------------------------------------------------------
    chapters: List[Chapter] = []
    selected_chapters: List[int] = []
    if uploaded_file:
        try:
            chapters = load_chapters(uploaded_file.getvalue())
        except Exception as e:
            st.error(f"Could not read EPUB: {e}")

        if uploaded_file and not chapters:
            st.warning("No chapters found in the uploaded EPUB file.")

    if chapters:
        st.markdown("### 2. Select chapters")
        file_key = uploaded_file.file_id
        eligible = [i for i, c in enumerate(chapters) if c.char_count >= MIN_CHAPTER_CHARS]

        def set_all(value: bool):
            for i in eligible:
                st.session_state[f"chapter_{file_key}_{i}"] = value

        col1, col2, _ = st.columns([1, 1, 4])
        col1.button("Select all", on_click=set_all, args=(True,))
        col2.button("Clear", on_click=set_all, args=(False,))

        with st.expander(f"Chapters ({len(eligible)} of {len(chapters)} long enough to summarize)", expanded=True):
            for i, chapter in enumerate(chapters):
                too_short = i not in eligible
                label = f"{chapter.title} — {chapter.char_count:,} chars"
                if too_short:
                    label += f" (under {MIN_CHAPTER_CHARS} chars, skipped)"
                st.checkbox(label, key=f"chapter_{file_key}_{i}", disabled=too_short)

        selected_chapters = [i for i in eligible if st.session_state.get(f"chapter_{file_key}_{i}")]

    # --- 3. Model ---------------------------------------------------------------
    st.markdown("### 3. Choose a model")

    col1, col2 = st.columns([3, 1])
    with col1:
        max_price = st.slider(
            "Max input price ($ per million tokens)",
            min_value=0.10, max_value=20.0, value=1.0, step=0.05, format="$%.2f",
            help="Hides models priced above this. Large books add up fast with premium models."
        )
    with col2:
        include_free = st.checkbox(
            "Include free models",
            help="Free models are heavily rate-limited (about 20 requests/min and a small daily cap)."
        )
        if st.button("🔄 Refresh model list"):
            load_model_catalog.clear()

    catalog: list[dict] = []
    try:
        catalog = load_model_catalog()
    except Exception as e:
        st.error(f"Could not load the OpenRouter model list ({e}). Enter a model ID below instead.")

    models = filter_models(
        catalog,
        min_context=required_context(chars_per_chapter),
        max_input_price=max_price,
        include_free=include_free,
    )

    # Estimate cost over the chapters that would actually be sent
    if selected_chapters:
        estimate_counts = [chapters[i].char_count for i in selected_chapters]
        estimate_basis = f"{len(estimate_counts)} selected chapter(s)"
    elif chapters:
        estimate_counts = [c.char_count for c in chapters if c.char_count >= MIN_CHAPTER_CHARS]
        estimate_basis = f"all {len(estimate_counts)} eligible chapter(s)"
    else:
        estimate_counts = [chars_per_chapter]
        estimate_basis = "one full-length chapter (upload a book for a real estimate)"

    labels = {m.id: format_label(m, estimate_cost(m, estimate_counts, chars_per_chapter)) for m in models}
    models_by_id = {m.id: m for m in models}

    selected_model: Optional[ModelInfo] = None
    if models:
        if st.session_state.get("model_id") not in models_by_id:
            st.session_state.pop("model_id", None)
        selected_id = st.selectbox(
            f"Model ({len(models)} available, cheapest first — type to search)",
            options=list(models_by_id),
            index=default_model_index(models),
            format_func=labels.get,
            key="model_id",
        )
        selected_model = models_by_id[selected_id]
        st.caption(f"Estimates cover {estimate_basis}. Reasoning models may use more output tokens than estimated.")
    elif catalog:
        st.warning("No models match these filters. Raise the max price or pick a shorter text length.")

    custom_id = st.text_input(
        "Custom model ID (optional, overrides the list above)",
        placeholder="e.g. anthropic/claude-sonnet-5",
        help="Any OpenRouter model ID. It must support structured outputs or tool calling."
    ).strip()
    model_id = custom_id or (selected_model.id if selected_model else "")
    if custom_id:
        selected_model = find_model(catalog, custom_id)
        if selected_model is None:
            st.warning(f"'{custom_id}' is not in the OpenRouter catalog, so its price is unknown.")
        else:
            st.info(format_label(selected_model, estimate_cost(selected_model, estimate_counts, chars_per_chapter)))

    is_free_model = selected_model is not None and selected_model.is_free
    if is_free_model:
        st.warning("Free models allow about 20 requests/minute and a limited number per day; "
                   "requests run one at a time. Best for a handful of chapters.")
    concurrency = st.slider(
        "Parallel requests",
        min_value=1, max_value=16, value=4,
        disabled=is_free_model,
        help="How many chapters are summarized at once. Lower this if you see rate-limit errors."
    )
    if is_free_model:
        concurrency = 1

    # --- 4. Generate -------------------------------------------------------------
    st.markdown("### 4. Generate")
    api_key = st.text_input(
        "OpenRouter API Key",
        type="password",
        help="Get one at https://openrouter.ai/keys. Your key is not stored."
    ).strip()

    if st.button("Generate Summaries", type="primary", disabled=not chapters):
        if not api_key:
            st.warning("Please enter your OpenRouter API key")
        elif not model_id:
            st.warning("Please choose a model")
        elif not selected_chapters:
            st.warning("Please select at least one chapter")
        else:
            processor = EPUBSummaryInserter(model_id=model_id, api_key=api_key,
                                            chars_per_chapter=chars_per_chapter)
            book = read_book(uploaded_file.getvalue())
            with st.spinner(f"Summarizing {len(selected_chapters)} chapter(s) with {model_id}..."):
                result = asyncio.run(processor.process_selected_chapters(
                    selected_chapters, chapters, book, concurrency))

            if result:
                if result.reported_cost is not None:
                    cost_text = format_usd(result.reported_cost)
                elif selected_model is not None:
                    cost_text = "~" + format_usd(tokens_cost(selected_model, result.input_tokens, result.output_tokens))
                else:
                    cost_text = "unknown"
                usage_text = (f"{result.input_tokens:,} input / {result.output_tokens:,} output tokens, "
                              f"cost {cost_text}")

                total_attempted = len(selected_chapters)
                if result.failed > 0:
                    st.warning(f"✅ Processing complete! {result.succeeded} of {total_attempted} chapters "
                               f"processed successfully ({result.failed} failed). {usage_text}")
                else:
                    st.success(f"✅ Processing complete! All {result.succeeded} chapters processed "
                               f"successfully. {usage_text}")

                output_filename = f"{os.path.splitext(uploaded_file.name)[0]}_with_summaries.epub"
                st.download_button(
                    label="📥 Download Processed EPUB",
                    data=result.output_bytes,
                    file_name=output_filename,
                    mime="application/epub+zip",
                    type="primary"
                )

                st.markdown("---")

                if result.summaries:
                    with st.expander("📋 Review Generated Summaries (Optional)", expanded=False):
                        for _, chapter_title, summary in result.summaries:
                            st.markdown(f"**✅ {chapter_title}**")
                            st.markdown(summary)
                            st.markdown("---")

    with st.expander("ℹ️ How to use"):
        st.markdown("""
            1. Upload your EPUB file and pick how much of each chapter to send.
            2. Select the chapters you want to summarize.
            3. Choose a model. Prices are per million tokens, and the estimate covers your selected chapters.
            4. Enter your OpenRouter API key and click 'Generate Summaries'.
            5. Download the processed file when complete.

            **Troubleshooting:**
            - 401 / 402 errors: check your API key and your credit balance at openrouter.ai
            - Rate-limit errors: lower 'Parallel requests', or avoid free models for large books
            - Verify your EPUB file is under 200MB and not DRM protected
            - Each chapter is retried up to 4 times with increasing waits

            **Security Note:** Your API key is never stored and is only used during the active session.
        """)


if __name__ == "__main__":
    main()
