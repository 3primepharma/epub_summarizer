import asyncio
import io

import pytest
from ebooklib import epub
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

import summarize_epub_streamlit as app

DIGEST = {
    "content_type": "fiction",
    "at_a_glance": "A test chapter.",
    "key_concepts": ["one", "two"],
    "questions_to_hold": ["why?"],
    "points_of_tension": ["tension"],
}


def make_epub(chapter_texts):
    book = epub.EpubBook()
    book.set_identifier("test-book")
    book.set_title("Test")
    book.set_language("en")
    items = []
    for n, text in enumerate(chapter_texts, start=1):
        item = epub.EpubHtml(title=f"Chapter {n}", file_name=f"ch{n}.xhtml", lang="en")
        item.content = f"<html><body><h1>Chapter {n}</h1><p>{text}</p></body></html>"
        book.add_item(item)
        items.append(item)
    book.toc = items
    book.spine = ["nav", *items]
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    out = io.BytesIO()
    epub.write_epub(out, book)
    return out.getvalue()


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    async def instant(_):
        return None
    monkeypatch.setattr(app.asyncio, "sleep", instant)


def run(processor, model, epub_bytes, concurrency=2):
    chapters = app.extract_chapters(epub_bytes)
    selected = [i for i, c in enumerate(chapters) if c.char_count >= app.MIN_CHAPTER_CHARS]
    book = app.read_book(epub_bytes)
    with processor.agent.override(model=model):
        return chapters, selected, asyncio.run(
            processor.process_selected_chapters(selected, chapters, book, concurrency))


def test_summaries_are_inserted_into_output_epub():
    epub_bytes = make_epub(["word " * 300, "short", "more words " * 200])
    processor = app.EPUBSummaryInserter("test/model", "sk-test", 4000)

    chapters, selected, result = run(processor, TestModel(custom_output_args=DIGEST), epub_bytes)

    assert len(selected) == 2
    assert (result.succeeded, result.failed) == (2, 0)
    assert [i for i, _, _ in result.summaries] == selected
    assert result.input_tokens > 0

    out_chapters = app.extract_chapters(result.output_bytes)
    by_id = {c.id: c.content for c in out_chapters}
    for i in selected:
        assert "chapter-digest" in by_id[chapters[i].id]
        assert "A test chapter." in by_id[chapters[i].id]


def test_transient_errors_are_retried():
    calls = {"n": 0}

    def flaky(messages, info: AgentInfo):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ModelHTTPError(503, "test/model", headers={"Retry-After": "1"})
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, DIGEST)])

    processor = app.EPUBSummaryInserter("test/model", "sk-test", 4000)
    _, _, result = run(processor, FunctionModel(flaky), make_epub(["word " * 300]), concurrency=1)

    assert (result.succeeded, result.failed) == (1, 0)
    assert calls["n"] == 2


def test_fatal_errors_stop_the_run():
    calls = {"n": 0}

    def no_credits(messages, info: AgentInfo):
        calls["n"] += 1
        raise ModelHTTPError(402, "test/model", body="insufficient credits")

    processor = app.EPUBSummaryInserter("test/model", "sk-test", 4000)
    _, _, result = run(processor, FunctionModel(no_credits), make_epub(["word " * 300] * 3), concurrency=1)

    assert result is None
    assert calls["n"] == 1
