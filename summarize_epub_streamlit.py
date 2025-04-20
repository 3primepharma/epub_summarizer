import streamlit as st
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from typing import List, Tuple
import os
import time
import io
import tempfile
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pydantic import BaseModel, Field
from pydantic_ai import Agent

class ChapterDigest(BaseModel):
    """Structured output model for chapter digests"""
    summary: str = Field(description="A concise summary of the chapter (3-5 sentences)")
    perspectives: List[str] = Field(description="3-5 bullet points highlighting main perspectives")
    implications: List[str] = Field(description="3-5 bullet points outlining implications")
    dissenting_opinions: List[str] = Field(description="3-5 bullet points offering opposing viewpoints to the main perspective")
    food_for_thought: List[str] = Field(description="3-5 thought-provoking questions or points")

class EPUBSummaryInserter:
    def __init__(self, provider: str, model: str, api_key: str, chars_per_chapter: int):
        if not api_key:
            raise ValueError("API key is required")
        
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.chars_per_chapter = chars_per_chapter
        self.agent = self._initialize_agent()
    
    def _initialize_agent(self):
        """Initialize PydanticAI agent with the selected model"""
        # Configure environment variables for the selected provider
        if self.provider == "openai":
            os.environ["OPENAI_API_KEY"] = self.api_key
            model_name = f"openai:{self.model}"
        elif self.provider == "anthropic":
            os.environ["ANTHROPIC_API_KEY"] = self.api_key
            # Add "-latest" suffix for Anthropic models
            model_name = f"anthropic:{self.model}-latest"
        elif self.provider == "gemini":
            os.environ["GEMINI_API_KEY"] = self.api_key
            model_name = f"google-gla:{self.model}"
        
        # Initialize PydanticAI agent with the selected model
        return Agent(model=model_name, output_type=ChapterDigest)

    def extract_chapters(self, epub_file_path: str) -> List[Tuple[str, str, str, int]]:
        """
        Extract chapters from EPUB file.
        Returns list of tuples: (chapter_id, title, content, char_count)
        """
        book = epub.read_epub(epub_file_path)
        chapters = []
        
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                soup = BeautifulSoup(item.get_content(), 'html.parser')
                title = soup.find(['h1', 'h2'])
                title = title.get_text().strip() if title else "Untitled Chapter"
                content = str(soup)
                # Get actual character count of cleaned text
                char_count = len(BeautifulSoup(content, 'html.parser').get_text())
                chapters.append((item.id, title, content, char_count))
        
        return chapters

    async def get_chapter_summary(self, content: str) -> str:
        """Get AI-generated summary for chapter content using PydanticAI"""
        # Create a container for status messages that will be overwritten
        status_container = st.empty()
        
        # Strip HTML tags for cleaner text
        text = BeautifulSoup(content, 'html.parser').get_text()
        
        prompt = f"""You are tasked with creating a comprehensive digest of a chapter from an epub, which could be from a book, newspaper, saved articles, or documentation. Your goal is to provide a quick overview of what to expect before reading, along with additional insights to stimulate thinking on the chapter's topic.

Here is the chapter text:
<chapter_text>
{text[:self.chars_per_chapter]}
</chapter_text>

Please analyze this text and create a chapter digest with:
1. A concise summary of the chapter (3-5 sentences)
2. 3-5 bullet points highlighting the main perspectives presented
3. 3-5 bullet points outlining the implications of the content
4. 3-5 bullet points offering dissenting opinions or opposing viewpoints to the main perspective
5. 3-5 thought-provoking questions or points to ponder related to the topic
"""

        max_retries = 3
        for attempt in range(max_retries):
            try:
                status_container.info(f'Attempt {attempt + 1}/{max_retries}: Generating summary...')
                
                # Use PydanticAI to generate the summary
                message_placeholder = st.empty()
                
                # Instead of streaming chunks, get the complete structured output
                async with self.agent.run_stream(prompt) as result:
                    # Get the final structured output
                    digest = await result.get_output()
                    
                    # Format the digest into the expected format
                    formatted_digest = f"""{digest.summary}

Perspectives
• {("• ").join([p + "\n" for p in digest.perspectives])}

Implications
• {("• ").join([i + "\n" for i in digest.implications])}

Dissenting Opinions
• {("• ").join([d + "\n" for d in digest.dissenting_opinions])}

Food For Thought
• {("• ").join([f + "\n" for f in digest.food_for_thought])}"""
                    
                    # Display the formatted digest
                    message_placeholder.markdown(formatted_digest)
                
                # Clear status message on success
                status_container.empty()
                
                return formatted_digest.strip()
                
            except Exception as e:
                if attempt < max_retries - 1:
                    status_container.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying...")
                    await asyncio.sleep(5)
                else:
                    status_container.error(f"All {max_retries} attempts failed for chapter summary. Skipping...")
                    return ""

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
                        if any(heading in line for heading in ['Perspectives', 'Implications', 'Dissenting Opinions', 'Food For Thought']):
                            p['class'] = 'heading'
                        p.string = line
                        section_div.append(p)
            
            summary_div.append(section_div)
        
        # Insert at start of body or main content
        body = soup.find('body') or soup
        body.insert(0, summary_div)
        
        return str(soup)
    
    async def process_selected_chapters(self, selected_indices: List[int], chapters: List[Tuple[str, str, str, int]], book: epub.EpubBook, batch_size: int, batch_wait: int) -> bytes:
        """Process and insert summaries for selected chapters in parallel batches"""
        # Filter out chapters that are too short (less than 400 characters)
        valid_indices = [i for i in selected_indices if 0 <= i < len(chapters) and chapters[i][3] >= 800]
        
        if len(valid_indices) < len(selected_indices):
            skipped_count = len(selected_indices) - len(valid_indices)
            st.warning(f"Skipping {skipped_count} chapter(s) that are too short (less than 800 characters)")
        
        if not valid_indices:
            st.error("No valid chapters to process after filtering out short chapters")
            return None

        # Calculate total characters for selected chapters
        total_chars = sum(min(chapters[i][3], self.chars_per_chapter) for i in valid_indices)
        avg_chars_per_chapter = total_chars / len(valid_indices) if valid_indices else 0
        
        # Adjust batch size based on actual content size
        adjusted_batch_size = min(
            batch_size,
            int(self.chars_per_chapter * batch_size / max(avg_chars_per_chapter, 1))
        )
        
        # Adjust wait time proportionally
        adjusted_wait = max(5, int(batch_wait * avg_chars_per_chapter / self.chars_per_chapter))
        
        st.info(f"""
            Optimized Processing Parameters:
            - Average characters per chapter: {int(avg_chars_per_chapter):,}
            - Adjusted batch size: {adjusted_batch_size} chapters
            - Adjusted wait time: {adjusted_wait} seconds
        """)
        
        progress_bar = st.progress(0)
        current_batch = st.empty()
        
        with st.container():
            st.markdown("### Processing Status")
            status_container = st.empty()
        
        BATCH_SIZE = adjusted_batch_size
        BATCH_WAIT = adjusted_wait  # seconds
        processed_chapters = 0
        
        async def process_chapter(i: int) -> Tuple[str, str, bool]:
            chapter_id, title, content, _ = chapters[i]
            try:
                summary = await self.get_chapter_summary(content)
                if summary:
                    modified_content = self.insert_summary(content, summary)
                    return (chapter_id, modified_content, True)
                return (chapter_id, "", False)
            except Exception as e:
                st.error(f"Error processing {title}: {str(e)}")
                return (chapter_id, "", False)

        # Process chapters in batches
        for batch_start in range(0, len(valid_indices), BATCH_SIZE):
            batch_indices = valid_indices[batch_start:batch_start + BATCH_SIZE]
            current_batch.write(f"📖 Processing batch {batch_start//BATCH_SIZE + 1}")
            
            # Process batch concurrently
            tasks = [process_chapter(i) for i in batch_indices]
            batch_results = await asyncio.gather(*tasks)
            
            # Update book with results
            for chapter_id, content, success in batch_results:
                if success and content:
                    for item in book.get_items():
                        if item.id == chapter_id:
                            item.set_content(content.encode())
                            processed_chapters += 1
                            progress_bar.progress(processed_chapters / len(valid_indices))
            
            # Wait between batches if there are more chapters to process
            if batch_start + BATCH_SIZE < len(valid_indices):
                status_container.info(f"Waiting {BATCH_WAIT} seconds before processing next batch...")
                await asyncio.sleep(BATCH_WAIT)

        current_batch.empty()
        
        # Save modified book
        with tempfile.NamedTemporaryFile(delete=False, suffix='.epub') as temp_output:
            epub.write_epub(temp_output.name, book)
            temp_output_path = temp_output.name
        
        with open(temp_output_path, 'rb') as f:
            output_bytes = f.read()
        
        os.remove(temp_output_path)
        return output_bytes

def main():
    st.set_page_config(page_title="EPUB Summary Generator", layout="wide")
    
    # Add some custom CSS
    st.markdown("""
        <style>
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }
        .upload-section {
            padding: 2rem;
            border-radius: 10px;
            border: 2px dashed #cccccc;
            margin-bottom: 2rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("📚 EPUB Summary Generator")
    
    # Add key information from README
    st.markdown("""
        Enhance your EPUB files with AI-powered chapter summaries using multiple LLM providers. 
        Each chapter summary includes:
        - Concise chapter overview
        - Key perspectives
        - Important implications
        - Dissenting opinions (opposing viewpoints)
        - Thought-provoking questions

        **Generated summaries are integrated seamlessly to the start of each chapter to get you primed and your juices flowing before diving into the material**
        
        ⚠️ **Important Usage Notes:**
        - Supports OpenAI, Anthropic, and Google Gemini models
        - By default, processes chapters in batches with cooling periods based on provider rate limits
        - Set the maximum characters processed per chapter
        - Maximum file size: 200MB
        - You are responsible for all API costs - check your provider's pricing
    """)
    
    # File upload section
    st.markdown("### Upload EPUB")
    uploaded_file = st.file_uploader(
        "Upload your EPUB file (max 200MB)",
        type=['epub'],
        help="Supported sources: Project Gutenberg, Instapaper, Calibre conversions, and more"
    )
    
    # Provider and model selection
    provider_options = {
        "OpenAI": {
            "models": ["gpt-4o-mini", "gpt-4o"],
            "env_var": "OPENAI_API_KEY",
            "label": "OpenAI API Key",
            "rate_limits": {
                "gpt-4o-mini": {"rpm": 500, "tpm": 300000},
                "gpt-4o": {"rpm": 500, "tpm": 300000}
            }
        },
        "Anthropic": {
            "models": ["claude-3-5-haiku", "claude-3-5-sonnet"],
            "env_var": "ANTHROPIC_API_KEY",
            "label": "Anthropic API Key",
            "rate_limits": {
                "claude-3-5-haiku": {"rpm": 45, "tpm": 100000},
                "claude-3-5-sonnet": {"rpm": 5, "tpm": 15000}
            }
        },
        "Gemini": {
            "models": ["gemini-2.0-flash", "gemini-2.0-pro"],
            "env_var": "GEMINI_API_KEY",
            "label": "Gemini API Key",
            "rate_limits": {
                "gemini-2.0-flash": {"rpm": 60, "tpm": 120000},
                "gemini-2.0-pro": {"rpm": 60, "tpm": 120000}
            }
        }
    }

    selected_provider = st.selectbox(
        "Select LLM Provider",
        options=list(provider_options.keys()),
        index=0,
        help="Choose your preferred LLM provider"
    )

    # Get models for the selected provider
    available_models = provider_options[selected_provider]["models"]
    selected_model = st.selectbox(
        f"Select {selected_provider} Model",
        options=available_models,
        index=0,
        help=f"Choose which {selected_provider} model to use"
    )

    # Update API key input label based on selected provider
    api_key = st.text_input(
        provider_options[selected_provider]["label"],
        type="password",
        help="Your API key will not be stored"
    )
    
    # Get rate limits for selected provider and model
    provider_key = selected_provider.lower()
    rate_limits = provider_options[selected_provider]["rate_limits"][selected_model]
    requests_per_minute = rate_limits["rpm"]
    tokens_per_minute = rate_limits["tpm"]

    # Add text length selection
    length_options = {
        "Short (1-2 pages, 4k chars)": 4000,
        "Medium (<15 pages, 20k chars)": 20000,
        "Long (15-30 pages, 40k chars)": 40000,
        "Long (30-50 pages, 100k chars)": 100000
    }
    selected_length = st.selectbox(
        "Select text length per chapter",
        options=list(length_options.keys()),
        index=0,
        help="Choose based on your typical chapter length"
    )

    if uploaded_file and api_key:
        try:
            # Get character limit for chapters
            chars_per_chapter = length_options[selected_length]
            
            # Initialize processor with selected provider, model, and character limit
            processor = EPUBSummaryInserter(
                provider=provider_key,
                model=selected_model,
                api_key=api_key,
                chars_per_chapter=chars_per_chapter
            )
            
            # Extract chapters first to get actual sizes
            with tempfile.NamedTemporaryFile(delete=False, suffix='.epub') as temp_input:
                temp_input.write(uploaded_file.getvalue())
                temp_input_path = temp_input.name
            
            book = epub.read_epub(temp_input_path)
            chapters = processor.extract_chapters(temp_input_path)
            os.remove(temp_input_path)

            if chapters:
                # Calculate average actual chapter size
                avg_chapter_size = sum(min(chapter[3], chars_per_chapter) for chapter in chapters) / len(chapters)
                
                # Calculate batch parameters considering token limits and request limits
                # Estimate tokens per chapter (roughly 4 chars per token)
                tokens_per_chapter = avg_chapter_size / 4
                tokens_batch_size = tokens_per_minute // max(int(tokens_per_chapter), 1)
                
                # Minimum wait time of 20 seconds
                min_wait = 20
                rpm_batch_size = (requests_per_minute * min_wait) // 60
                
                # Take the more conservative of the two limits
                batch_size = min(tokens_batch_size, rpm_batch_size)
                
                # Recalculate wait time based on final batch size and actual chapter sizes
                token_wait = (batch_size * tokens_per_chapter * 60) // tokens_per_minute
                rpm_wait = (batch_size * 60) // requests_per_minute
                batch_wait = max(min_wait, token_wait, rpm_wait)

                # Display calculated processing parameters
                st.info(f"""
                    Processing Parameters:
                    - Average chapter size: {int(avg_chapter_size):,} characters (~{int(tokens_per_chapter):,} tokens)
                    - Batch Size: {batch_size} chapters
                    - Wait Time: {batch_wait} seconds between batches
                    - Characters per chapter limit: {chars_per_chapter:,}
                    - Token rate limit: {tokens_per_minute:,} tokens/minute
                    - Request rate limit: {requests_per_minute} requests/minute
                    - Estimated throughput: {int(batch_size * (60/batch_wait))} chapters/minute
                """)

                # Initialize selected_chapters in session state if not present
                if 'selected_chapters' not in st.session_state:
                    st.session_state.selected_chapters = set()

                # Add select_all to session state if not present
                if 'select_all' not in st.session_state:
                    st.session_state.select_all = False

                # Handle select all checkbox
                select_all = st.checkbox(
                    "Select All Chapters",
                    key='select_all',
                    value=st.session_state.select_all
                )

                # Update selected chapters when select all changes
                if select_all:
                    st.session_state.selected_chapters = set(range(len(chapters)))
                
                # Display individual chapter checkboxes
                for i, (_, title, _, _) in enumerate(chapters):
                    # Use the value from session_state.selected_chapters to set initial state
                    is_checked = st.checkbox(
                        title,
                        key=f"chapter_{i}",
                        value=(i in st.session_state.selected_chapters)
                    )
                    if is_checked:
                        st.session_state.selected_chapters.add(i)
                    else:
                        st.session_state.selected_chapters.discard(i)
                
                # Generate Summaries Button
                if st.button("Generate Summaries", type="primary"):
                    selected_chapters = sorted(list(st.session_state.selected_chapters))
                    
                    if not selected_chapters:
                        st.warning("Please select at least one chapter")
                    else:
                        with st.spinner("Processing EPUB file..."):
                            output_bytes = asyncio.run(processor.process_selected_chapters(
                                selected_chapters, 
                                chapters, 
                                book,
                                batch_size,
                                batch_wait
                            ))
                            
                            if output_bytes:
                                # Auto-download trigger
                                output_filename = f"{os.path.splitext(uploaded_file.name)[0]}_with_summaries.epub"
                                st.download_button(
                                    label="📥 Download Processed EPUB",
                                    data=output_bytes,
                                    file_name=output_filename,
                                    mime="application/epub+zip"
                                )
                                st.success("✅ Processing complete! Your file has been prepared for download.")
            else:
                st.warning("No chapters found in the uploaded EPUB file.")
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    
    # Expand the "How to use" section with troubleshooting
    with st.expander("ℹ️ How to use"):
        st.markdown("""
            1. Upload your EPUB file using the file uploader above.
            2. Select your preferred LLM provider and model.
            3. Enter your API key for the selected provider.
            4. Select the text length per chapter.
            5. Select the chapters you want to summarize.
            6. Click 'Generate Summaries' to process your file.
            7. Download the processed file when complete.
            
            **Troubleshooting:**
            - Ensure your API key is valid and has sufficient credits
            - Verify your EPUB file is under 200MB and not DRM protected
            - If summaries fail, try processing fewer chapters at once
            - The app includes automatic retry logic (3 attempts)
            - Different providers have different rate limits, which may affect processing speed
            
            **Security Note:** Your API key is never stored and is only used during the active session.
        """)

if __name__ == "__main__":
    main()
