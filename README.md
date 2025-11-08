# EPUB Summary Generator

A Streamlit application that enhances EPUB files by automatically generating AI-powered chapter summaries using Anthropic's Claude API. Each chapter summary includes a concise overview, key perspectives, implications, and thought-provoking questions.

⚠️ **Important API Usage Notice**
- Supports multiple LLM providers: OpenAI, Anthropic, and Google Gemini
- **New**: Select your API tier to optimize batch processing based on your account's rate limits
- Processes chapters in batches with intelligent rate limiting
- Each chapter summary request processes a configurable amount of text (4k-100k characters)
- **Costs**: Please check your provider's pricing for current rates
- You are responsible for all API costs incurred

## Features

- Upload and process EPUB files (up to 200MB)
- **Multi-provider support**: Choose between OpenAI, Anthropic, and Google Gemini
- **Intelligent rate limiting**: Select your API tier for optimized batch processing
- **Customizable buffer settings**: Choose between conservative, balanced, or aggressive rate limit usage
- Select specific chapters for summarization
- AI-generated chapter digests including:
  - Concise chapter summary
  - Key perspectives
  - Important implications
  - Dissenting opinions (opposing viewpoints)
  - Thought-provoking questions
- Configurable text length per chapter (4k to 100k characters)
- Automatic batch processing with intelligent wait times
- Download enhanced EPUB with embedded summaries

## Prerequisites

- Python 3.x
- Anthropic API key ([Get one here](https://www.anthropic.com/))

## Installation
``` bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:
   ``` bash
   streamlit run summarize_epub_streamlit.py
   ```

2. Open the provided URL in your browser
3. Upload your EPUB file
4. Select your preferred LLM provider (OpenAI, Anthropic, or Gemini)
5. Choose the model you want to use
6. **Select your API tier** based on your account (or use "Custom" for exact limits)
7. Choose your rate limit buffer (Conservative, Balanced, or Aggressive)
8. Enter your API key
9. Select the text length per chapter
10. Select chapters to summarize
11. Click "Generate Summaries"
12. Download the enhanced EPUB file

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

- Uses BeautifulSoup for HTML parsing
- Implements parallel processing for chapter summarization using asyncio
- Includes retry logic for API calls (3 attempts per chapter)
- Preserves original EPUB structure and formatting
- **Tier-based rate limiting**: Automatically calculates optimal batch sizes and wait times based on your API tier
- **Dynamic throughput optimization**: Adjusts processing speed to maximize your account's capabilities
- Supports PydanticAI for structured output across multiple LLM providers

## Dependencies

- streamlit
- ebooklib
- beautifulsoup4
- pydantic-ai
- asyncio
- anthropic / openai / google-generativeai (depending on provider)

## Rate Limit Tiers

The application supports different API tiers for each provider:

### OpenAI
- Tier 1 (Free): 500 RPM, 200K TPM (gpt-4o-mini)
- Tier 2-5: Progressively higher limits based on account spend

### Anthropic
- Tier 1: 50 RPM, 50K TPM
- Tier 2-4: Progressively higher limits based on account spend

### Google Gemini
- Free Tier: 5-10 RPM, 32K-250K TPM
- Tier 1 (Paid): 300 RPM, 1M TPM
- Tier 2-3: Higher limits for enterprise customers

**Custom Tier**: Enter your exact RPM and TPM limits if you know them

### How to Check Your API Tier

- **OpenAI**: Visit your [account limits page](https://platform.openai.com/account/limits)
- **Anthropic**: Check your [account settings](https://console.anthropic.com/settings/limits)
- **Google Gemini**: View your [quota page](https://console.cloud.google.com/apis/api/generativelanguage.googleapis.com/quotas) in Google Cloud Console

### Buffer Settings

Choose how aggressively to use your rate limits:
- **Conservative (50%)**: Safest option, reduces risk of hitting rate limits
- **Balanced (70%)**: Good balance between speed and safety (recommended)
- **Aggressive (90%)**: Maximum speed, uses most of your available capacity

## Limitations

- Maximum EPUB file size: 200MB
- Maximum text processed per chapter: Configurable (4k to 100k characters)
- Batch size and wait times calculated dynamically based on selected tier
- Some EPUB files with complex formatting may not process correctly
- Rate limits are per-account and may be shared across multiple applications

## Troubleshooting

Common issues and solutions:

1. **API Key Issues**
   - Ensure your API key is valid and has sufficient credits
   - Check that there are no leading/trailing spaces in the API key
   - Verify you've selected the correct provider for your API key

2. **Rate Limit Errors**
   - Try selecting a lower tier if you're hitting rate limits
   - Switch to "Conservative" buffer setting
   - Check your actual tier in your provider's dashboard
   - Consider using "Custom" tier with exact limits if issues persist

3. **File Processing Errors**
   - Verify your EPUB file is under 200MB
   - Ensure the EPUB is not DRM protected
   - Try converting the EPUB to a newer format using Calibre

4. **Summary Generation Failures**
   - The app includes automatic retry logic (3 attempts)
   - If persistent failures occur, try processing fewer chapters at once
   - Reduce the text length per chapter if chapters are very long

5. **Slow Processing**
   - Select a higher tier if your account supports it
   - Use "Aggressive" buffer setting for faster processing
   - Consider using a faster model (e.g., Haiku for Anthropic, gpt-4o-mini for OpenAI)

## Security Considerations

- Your API key is never stored and is only used during the active session
- Files are processed locally and are not stored permanently
- Temporary files are automatically cleaned up after processing
- No data is sent to external services except the text for summarization to Anthropic

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
