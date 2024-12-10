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
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is required")
        self.client = anthropic.Anthropic(api_key=api_key)

    def extract_chapters(self, epub_file_path: str) -> List[Tuple[str, str, str]]:
        """
        Extract chapters from EPUB file.
        Returns list of tuples: (chapter_id, title, content)
        """
        book = epub.read_epub(epub_file_path)
        chapters = []
        
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                soup = BeautifulSoup(item.get_content(), 'html.parser')
                title = soup.find(['h1', 'h2'])
                title = title.get_text().strip() if title else "Untitled Chapter"
                chapters.append((item.id, title, str(soup)))
        
        return chapters

    def get_chapter_summary(self, content: str) -> str:
        """Get AI-generated summary for chapter content"""
        # Create a container for status messages that will be overwritten
        status_container = st.empty()
        
        # Strip HTML tags for cleaner text
        text = BeautifulSoup(content, 'html.parser').get_text()
        
        prompt = f"""You are tasked with creating a comprehensive digest of a chapter from an epub, which could be from a book, newspaper, saved articles, or documentation. Your goal is to provide a quick overview of what to expect before reading, along with additional insights to stimulate thinking on the chapter's topic.

Here is the chapter text:
<chapter_text>
{text[:4000]}  # Limiting text length for API
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
                # Update logging to use status container
                status_container.info(f'Attempt {attempt + 1}/{max_retries}: Generating summary...')
                time.sleep(5)
                
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
                    time.sleep(5)
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

    async def process_selected_chapters(self, selected_indices: List[int], chapters: List[Tuple[str, str, str]], book: epub.EpubBook) -> bytes:
        """Process and insert summaries for selected chapters in parallel batches"""
        valid_indices = [i for i in selected_indices if 0 <= i < len(chapters)]
        if len(valid_indices) != len(selected_indices):
            st.warning(f"Some selected chapter indices were invalid and will be skipped. Valid: {len(valid_indices)}/{len(selected_indices)}")
        
        if not valid_indices:
            raise ValueError("No valid chapters selected for processing")

        progress_bar = st.progress(0)
        current_batch = st.empty()
        
        with st.container():
            st.markdown("### Processing Status")
            status_container = st.empty()
        
        BATCH_SIZE = 15
        BATCH_WAIT = 20  # seconds
        processed_chapters = 0
        
        async def process_chapter(i: int) -> Tuple[str, str, bool]:
            chapter_id, title, content = chapters[i]
            try:
                summary = self.get_chapter_summary(content)
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

        **Generated summaries are integrated seamlessly to the start of each chapter to get you primed and your juices flowing before diving into the material**
        
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
    
    if uploaded_file and api_key:
        try:
            # Reset the pointer of the uploaded_file to the beginning
            uploaded_file.seek(0)
            processor = EPUBSummaryInserter(api_key)
            
            # Read the EPUB content into a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.epub') as temp_input:
                temp_input.write(uploaded_file.read())
                temp_input_path = temp_input.name
            
            # Extract chapters
            book = epub.read_epub(temp_input_path)
            chapters = processor.extract_chapters(temp_input_path)
            
            # Cleanup the temporary input file
            os.remove(temp_input_path)
            
            if chapters:
                st.subheader("Select Chapters to Summarize")
                
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
                for i, (_, title, _) in enumerate(chapters):
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
                            output_bytes = asyncio.run(processor.process_selected_chapters(selected_chapters, chapters, book))
                            
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
