import pytest

from app.services.chunker import chunk_text

PROSE = " ".join(f"Sentence number {i} talks about attention mechanisms in detail." for i in range(60))


def test_chunks_respect_size_and_index():
    chunks = chunk_text(PROSE, page_number=3, chunk_size=300, overlap=60)
    assert len(chunks) > 1
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert all(c.page_number == 3 for c in chunks)
    # A chunk may exceed the budget only if a single sentence is longer than it
    assert all(len(c.content) <= 300 + 80 for c in chunks)


def test_adjacent_chunks_overlap():
    chunks = chunk_text(PROSE, chunk_size=300, overlap=60)
    first_tail = chunks[0].content.split(". ")[-1].rstrip(".")
    assert first_tail in chunks[1].content


def test_sentences_are_not_split_mid_way():
    chunks = chunk_text(PROSE, chunk_size=300, overlap=60)
    for c in chunks:
        assert c.content.endswith(".")


def test_start_index_continues_numbering():
    chunks = chunk_text(PROSE, chunk_size=300, overlap=60, start_index=10)
    assert chunks[0].index == 10


def test_short_text_is_one_chunk():
    chunks = chunk_text("Just one short sentence.", chunk_size=300, overlap=60)
    assert len(chunks) == 1
    assert chunks[0].content == "Just one short sentence."


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        chunk_text(PROSE, chunk_size=100, overlap=100)
