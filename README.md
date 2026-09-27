# EPUB Summary Generator

A Streamlit application that enhances EPUB files with AI-generated pre-reading primers for each chapter, using any model available on [OpenRouter](https://openrouter.ai). Each primer includes an orientation, key concepts, questions to hold while reading, and points of tension.

⚠️ **Important API Usage Notice**
- All requests go through OpenRouter using your own API key ([get one here](https://openrouter.ai/keys))
- The model list and prices are fetched live from OpenRouter; the app shows an estimated cost for your book before you run it
- You are responsible for all API costs incurred

## Features

- Upload and process EPUB files (up to 200MB)
- **Any OpenRouter model**: the list is filtered to models that support structured output and fit your chapter size, sorted cheapest first
- **Cost-aware model picker**: each model shows its input/output price per million tokens and an estimated total for the chapters you've selected
- **Max-price filter**: hides models above your chosen input price ($0.10–$5.00 per million tokens, default $1.00). Pricier models can still be used via the custom model ID field
- **Optional free models**: opt in to OpenRouter's rate-limited free models for small jobs
- **Actual cost report** after each run
- Select specific chapters to summarize
- AI-generated chapter pre-reads including:
  - At a Glance: brief orientation to the chapter
  - Key Concepts: terms, ideas, and themes to prime your reading
  - Questions to Hold: questions to consider while reading
  - Points of Tension: areas of complexity, debate, or narrative tension
- Configurable text length per chapter (4k to 100k characters)
- Parallel processing with automatic retries and backoff
- Download enhanced EPUB with embedded pre-reads

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (installs Python 3.13 for the project automatically)
- An OpenRouter API key with credits ([openrouter.ai/keys](https://openrouter.ai/keys))

## Installation

```bash
uv sync
```

This creates a project-local `.venv` with the locked dependency versions from `uv.lock`.

## Usage

1. Run the Streamlit app:
   ```bash
   uv run streamlit run summarize_epub_streamlit.py
   ```
2. Open the provided URL in your browser
3. Upload your EPUB file and choose the text length per chapter
4. Select the chapters to summarize
5. Choose a model. Adjust the max-price slider or include free models if you like. The estimate updates as you select chapters
6. Set the number of parallel requests (default 4)
7. Enter your OpenRouter API key and click "Generate Summaries"
8. Download the enhanced EPUB file

## Development

```bash
uv run pytest
```

After changing dependencies in `pyproject.toml`, refresh the lock file and the exported `requirements.txt` (used by Streamlit Community Cloud and the devcontainer):

```bash
uv lock --upgrade
uv export --no-dev --no-hashes --no-emit-project -o requirements.txt
```

## Where to Get EPUB Files

There are several excellent sources for obtaining EPUB files:

1. **Project Gutenberg** ([www.gutenberg.org](https://www.gutenberg.org))
   - Vast collection of free, public domain books
   - Classic literature and historical texts
   - No registration required

2. **Instapaper** ([www.instapaper.com](https://www.instapaper.com))
   - Save web articles for later reading
   - Convert saved articles to EPUB format
   - Great for creating collections of articles

3. **Calibre** ([calibre-ebook.com](https://calibre-ebook.com))
   - Convert various document formats to EPUB
   - Manage your ebook library
   - Convert newsletters and documents

4. **EPUBlifier** ([github.com/maoserr/epublifier](https://github.com/maoserr/epublifier))
   - Convert blog content to EPUB format
   - Crawl websites for content
   - Create EPUBs from multiple sources

## Technical Details

- Uses BeautifulSoup for HTML parsing and ebooklib for reading and writing EPUBs
- Uses PydanticAI's OpenRouter model for structured (`ChapterDigest`) output
- The model catalog comes from `https://openrouter.ai/api/v1/models` (no key needed) and is cached for an hour, with a manual refresh button
- Model filtering excludes routers, batch-only and alias variants, image/audio generation models, models without structured-output or tool support, expired models, and models whose context window is too small for the selected chapter length
- Cost estimates assume ~4 characters per token, ~450 prompt tokens, and ~700 output tokens per chapter. Reasoning models may use more output tokens
- Chapters are processed concurrently (bounded by "Parallel requests"). Each chapter is retried up to 4 times with exponential backoff, honouring `Retry-After`
- An invalid key (401), insufficient credits (402), or unknown model (404) stops the run immediately

## Limitations

- Maximum EPUB file size: 200MB
- Chapters shorter than 800 characters are skipped
- Free models allow roughly 20 requests per minute and a limited number per day, so they only suit small jobs
- Some EPUB files with complex formatting may not process correctly

## Troubleshooting

1. **API key or credit errors (401 / 402)**
   - Check the key at [openrouter.ai/keys](https://openrouter.ai/keys) and make sure there are no leading or trailing spaces
   - Top up your credit balance, or pick a cheaper model

2. **Rate limit errors (429)**
   - Lower "Parallel requests"
   - Avoid free models for large books

3. **A model is missing from the list**
   - Raise the max-price slider, or choose a shorter text length (some models have small context windows)
   - Click "Refresh model list", or enter the model ID in the custom field

4. **File processing errors**
   - Verify your EPUB file is under 200MB and not DRM protected
   - Try converting the EPUB to a newer format using Calibre

## Security Considerations

- Your API key is never stored and is only used during the active session
- Uploaded files are not stored permanently; temporary copies are deleted right after reading
- Chapter text is sent only to OpenRouter (and the model provider it routes to)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
