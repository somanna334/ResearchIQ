import pytest
from pydantic import ValidationError

from app.chunking.strategies import ChunkConfig, chunk_pages


def words(n: int) -> str:
    return " ".join(f"word{i}" for i in range(n))


def test_config_key_identifies_the_chunk_set():
    assert ChunkConfig().key == "fixed-800-150"


def test_overlap_larger_than_half_the_size_is_rejected():
    with pytest.raises(ValidationError):
        ChunkConfig(chunk_size=300, overlap=200)


def test_fixed_chunks_respect_size_and_overlap():
    config = ChunkConfig(strategy="fixed", chunk_size=200, overlap=50)
    chunks = chunk_pages([(1, words(200))], config)

    assert len(chunks) > 1
    assert all(len(c.text) <= 200 for c in chunks)
    for first, second in zip(chunks, chunks[1:]):
        # The last word of a chunk is repeated at the start region of the next.
        assert first.text.split()[-1] in second.text.split()


def test_chunks_keep_page_numbers_and_never_span_pages():
    config = ChunkConfig(chunk_size=200, overlap=0)
    chunks = chunk_pages([(3, "alpha " * 100), (7, "beta " * 100)], config)

    assert {c.page_number for c in chunks} == {3, 7}
    for chunk in chunks:
        assert ("alpha" in chunk.text) != ("beta" in chunk.text)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_tiny_fragments_are_dropped():
    assert chunk_pages([(1, "Page 1")], ChunkConfig()) == []


def test_sentence_chunks_end_on_sentence_boundaries():
    text = " ".join(f"This is sentence number {i} of the test." for i in range(40))
    config = ChunkConfig(strategy="sentence", chunk_size=200, overlap=60)
    chunks = chunk_pages([(1, text)], config)

    assert len(chunks) > 1
    assert all(len(c.text) <= 200 for c in chunks)
    assert all(c.text.endswith(".") for c in chunks)