import streamlit as st
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import anthropic
from typing import List, Tuple
import os
import time
import io
import tempfile
import asyncio
from concurrent.futures import ThreadPoolExecutor

class EPUBSummaryInserter:
    def __init__(self, api_key: str, chars_per_chapter: int):
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is required")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.chars_per_chapter = chars_per_chapter

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
        """Get AI-generated summary for chapter content"""
        # Create a container for status messages that will be overwritten
        status_container = st.empty()
        
        # Strip HTML tags for cleaner text
        text = BeautifulSoup(content, 'html.parser').get_text()
        
        prompt = f"""You are tasked with creating a comprehensive digest of a chapter from an epub, which could be from a book, newspaper, saved articles, or documentation. Your goal is to provide a quick overview of what to expect before reading, along with additional insights to stimulate thinking on the chapter's topic.

Here is the chapter text:
<chapter_text>
{text[:self.chars_per_chapter]}  # Using dynamic character limit
</chapter_text>

Please follow these steps to create the digest:

1. Carefully read and analyze the provided chapter text.
2. Write a concise summary of the chapter, capturing its main ideas and key points. This summary should be approximately 3-5 sentences long.
3. Identify the main perspectives presented in the chapter. Create 3-5 bullet points highlighting these perspectives.
4. Consider the implications of the chapter's content. Create 3-5 bullet points outlining these implications.
5. Develop 3-5 thought-provoking questions or points to ponder related to the chapter's topic. These should stimulate further thinking and discussion.

Present your digest in the following format, using the specified XML tags:

<chapter_digest>
[Insert your concise summary here]

Perspectives
• [Perspective 1]
• [Perspective 2]
• [Perspective 3]
[Add more if necessary]

Implications
• [Implication 1]
• [Implication 2]
• [Implication 3]
[Add more if necessary]

Food For Thought
• [Thought-provoking question or point 1]
• [Thought-provoking question or point 2]
• [Thought-provoking question or point 3]
[Add more if necessary]
</chapter_digest>"""

        max_retries = 3
        for attempt in range(max_retries):
            try:
                status_container.info(f'Attempt {attempt + 1}/{max_retries}: Generating summary...')
                await asyncio.sleep(5)
                
                response = self.client.messages.create(
                    model="claude-3-5-haiku-20241022",
                    max_tokens=2000,
                    temperature=0,
                    messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}]
                )
                
                # Clear status message on success
                status_container.empty()
                
                # Extract content between chapter_digest tags
                response_text = response.content[0].text
                start_tag = "<chapter_digest>"
                end_tag = "</chapter_digest>"
                start_idx = response_text.find(start_tag) + len(start_tag)
                end_idx = response_text.find(end_tag)
                
                if start_idx == -1 or end_idx == -1:
                    raise ValueError("Could not extract chapter digest from AI response")
                
                return response_text[start_idx:end_idx].strip()
                
            except Exception as e:
                if attempt < max_retries - 1:
                    status_container.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying...")
                    await asyncio.sleep(5)
                else:
                    status_container.error(f"All {max_retries} attempts failed for chapter summary. Skipping...")
                    return ""

    def insert_summary(self, html_content: str, summary: str) -> str:
        """Insert summary at the start of chapter content"""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Create summary div with minimal styling
        summary_div = soup.new_tag('div')
        summary_div['class'] = 'chapter-digest'
        summary_div['style'] = 'margin: 1em 0; padding: 1em; border: 1px solid #ccc;'
        
        # Split the summary into sections and format them
        sections = summary.split('\n\n')
        for section in sections:
            section_div = soup.new_tag('div')
            section_div['style'] = 'margin-bottom: 1em;'
            
            # Convert the text to paragraphs and lists
            lines = section.strip().split('\n')
            for line in lines:
                if line.strip():
                    if line.startswith('•'):
                        p = soup.new_tag('li')
                        p.string = line[1:].strip()
                    else:
                        p = soup.new_tag('p')
                        if any(heading in line for heading in ['Perspectives', 'Implications', 'Food For Thought']):
                            p['style'] = 'font-weight: bold; margin-bottom: 0.5em;'
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
        Enhance your EPUB files with AI-powered chapter summaries using Anthropic's Claude API. 
        Each chapter summary includes:
        - Concise chapter overview
        - Key perspectives
        - Important implications
        - Thought-provoking questions
        
        ⚠️ **Important Usage Notes:**
        - Uses Claude 3.5 Haiku
        - Processes chapters in batches of 15 with 20-second cooling periods
        - Maximum 4,000 characters processed per chapter
        - Maximum file size: 200MB
        - You are responsible for all API costs - check [Anthropic's pricing](https://www.anthropic.com/pricing)
    """)
    
    # File upload section
    st.markdown("### Upload EPUB")
    uploaded_file = st.file_uploader(
        "Upload your EPUB file (max 200MB)",
        type=['epub'],
        help="Supported sources: Project Gutenberg, Instapaper, Calibre conversions, and more"
    )
    
    # API Key input
    api_key = st.text_input(
        "Enter your Anthropic API Key",
        type="password",
        help="Your API key will not be stored"
    )
    
    # Add API tier selection
    tier_options = {
        "Low (100k chars/min)": {"chars_per_min": 100000, "rpm": 45},
        "Medium (200k chars/min)": {"chars_per_min": 200000, "rpm": 500},
        "High (400k chars/min)": {"chars_per_min": 400000, "rpm": 1500}
    }
    selected_tier = st.selectbox(
        "Select your Anthropic API tier",
        options=list(tier_options.keys()),
        index=0,
        help="Choose based on your Anthropic API tier limits"
    )

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
            # Get tier limits
            tier_limits = tier_options[selected_tier]
            chars_per_minute = tier_limits["chars_per_min"]
            requests_per_minute = tier_limits["rpm"]
            chars_per_chapter = length_options[selected_length]
            
            # Initialize processor with selected character limit
            processor = EPUBSummaryInserter(api_key, chars_per_chapter)
            
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
                
                # Calculate batch parameters considering both character and request limits
                chars_batch_size = chars_per_minute // max(int(avg_chapter_size), 1)
                
                # Minimum wait time of 20 seconds
                min_wait = 20
                rpm_batch_size = (requests_per_minute * min_wait) // 60
                
                # Take the more conservative of the two limits
                batch_size = min(chars_batch_size, rpm_batch_size)
                
                # Recalculate wait time based on final batch size and actual chapter sizes
                char_wait = (batch_size * avg_chapter_size * 60) // chars_per_minute
                rpm_wait = (batch_size * 60) // requests_per_minute
                batch_wait = max(min_wait, char_wait, rpm_wait)

                # Display calculated processing parameters
                st.info(f"""
                    Processing Parameters:
                    - Average chapter size: {int(avg_chapter_size):,} characters
                    - Batch Size: {batch_size} chapters
                    - Wait Time: {batch_wait} seconds between batches
                    - Characters per chapter limit: {chars_per_chapter:,}
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
            2. Enter your Anthropic API key.
            3. Select the chapters you want to summarize.
            4. Click 'Generate Summaries' to process your file.
            5. Download the processed file when complete.
            
            **Troubleshooting:**
            - Ensure your API key is valid and has sufficient credits
            - Verify your EPUB file is under 200MB and not DRM protected
            - If summaries fail, try processing fewer chapters at once
            - The app includes automatic retry logic (3 attempts)
            
            **Security Note:** Your API key is never stored and is only used during the active session.
        """)

if __name__ == "__main__":
    main()