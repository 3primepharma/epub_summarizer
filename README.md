# EPUB Summary Generator

A Streamlit application that enhances EPUB files by automatically generating AI-powered chapter summaries using Anthropic's Claude API. Each chapter summary includes a concise overview, key perspectives, implications, and thought-provoking questions.

⚠️ **Important API Usage Notice**
- This application uses Claude 3 Haiku, Anthropic's most cost-effective model
- Processes chapters in batches of 15 with a 20-second pause between batches
- Each chapter summary request processes up to 4,000 characters
- Approximate usage: 45 requests per minute maximum
- **Costs**: Please check [Anthropic's pricing](https://www.anthropic.com/pricing) for current rates
- You are responsible for all API costs incurred

## Features

- Upload and process EPUB files (up to 200MB)
- Select specific chapters for summarization
- AI-generated chapter digests including:
  - Concise chapter summary
  - Key perspectives
  - Important implications
  - Thought-provoking questions
- Batch processing with rate limiting
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
   ``` base
   streamlit run summarize_epub_streamlit.py
   ```

2. Open the provided URL in your browser
3. Upload your EPUB file
4. Enter your Anthropic API key
5. Select chapters to summarize
6. Click "Generate Summaries"
7. Download the enhanced EPUB file

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
- Implements parallel processing for chapter summarization
- Includes retry logic for API calls
- Preserves original EPUB structure and formatting

## Dependencies

- streamlit
- ebooklib
- beautifulsoup4
- anthropic
- asyncio

## Limitations

- Maximum EPUB file size: 200MB
- Maximum text processed per chapter: 4,000 characters
- Rate limited to 15 chapters per batch
- 20-second cooling period between batches
- Some EPUB files with complex formatting may not process correctly

## Troubleshooting

Common issues and solutions:

1. **API Key Issues**
   - Ensure your API key is valid and has sufficient credits
   - Check that there are no leading/trailing spaces in the API key

2. **File Processing Errors**
   - Verify your EPUB file is under 200MB
   - Ensure the EPUB is not DRM protected
   - Try converting the EPUB to a newer format using Calibre

3. **Summary Generation Failures**
   - The app includes automatic retry logic (3 attempts)
   - If persistent failures occur, try processing fewer chapters at once

## Security Considerations

- Your API key is never stored and is only used during the active session
- Files are processed locally and are not stored permanently
- Temporary files are automatically cleaned up after processing
- No data is sent to external services except the text for summarization to Anthropic

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License

Copyright (c) 2024

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
