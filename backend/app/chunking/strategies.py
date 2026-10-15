import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Chunks shorter than this are almost always page numbers, running headers
# or stray fragments, so they are dropped.
MIN_CHUNK_CHARS = 20

_WHITESPACE = re.compile(r"\s+")
# Split after . ! ? followed by whitespace. Naive on purpose: "Dr. Smith" also splits.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


class ChunkConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    strategy: Literal["fixed", "sentence"] = "fixed"
    chunk_size: int = Field(800, ge=200, le=4000, description="Maximum characters per chunk")
    overlap: int = Field(150, ge=0, description="Characters repeated between neighbours")

    @model_validator(mode="after")
    def _check_overlap(self) -> "ChunkConfig":
        if self.overlap > self.chunk_size // 2:
            raise ValueError("overlap must be at most half of chunk_size")
        return self

    @property
    def key(self) -> str:
        """Identifies a chunk set, e.g. 'fixed-800-150'."""
        return f"{self.strategy}-{self.chunk_size}-{self.overlap}"


@dataclass(frozen=True)
class TextChunk:
    page_number: int
    chunk_index: int  # position within the whole document, starting at 0
    text: str


def _fixed_split(text: str, size: int, overlap: int) -> list[str]:
    """Sliding character window that prefers to cut at a space."""
    text = _WHITESPACE.sub(" ", text).strip()
    pieces: list[str] = []
    n = len(text)
    start = 0
    while start < n:
        end = min(start + size, n)
        if end < n:
            # Look for a space in the last 20% of the window to avoid cutting a word.
            cut = text.rfind(" ", start + int(size * 0.8), end)
            if cut != -1:
                end = cut
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= n:
            break
        next_start = end - overlap
        if next_start <= start:  # safety: always move forward
            next_start = end
        else:
            # Begin the next chunk at a word boundary.
            space = text.find(" ", next_start, end)
            if space != -1:
                next_start = space + 1
        start = next_start
    return pieces


def _overlap_tail(sentences: list[str], overlap: int) -> list[str]:
    """The last few whole sentences that fit within `overlap` characters."""
    tail: list[str] = []
    for sentence in reversed(sentences):
        if len(" ".join([sentence, *tail])) > overlap:
            break
        tail.insert(0, sentence)
    return tail


def _sentence_split(text: str, size: int, overlap: int) -> list[str]:
    """Pack whole sentences into chunks of at most `size` characters."""
    text = _WHITESPACE.sub(" ", text).strip()
    if not text:
        return []

    parts: list[str] = []
    for sentence in _SENTENCE_END.split(text):
        if len(sentence) <= size:
            parts.append(sentence)
        else:
            # A "sentence" longer than a chunk (tables, lists): hard-split it.
            parts.extend(_fixed_split(sentence, size, 0))

    chunks: list[str] = []
    current: list[str] = []
    for part in parts:
        if current and len(" ".join([*current, part])) > size:
            chunks.append(" ".join(current))
            current = _overlap_tail(current, overlap)
            if current and len(" ".join([*current, part])) > size:
                current = []  # carried sentences would make the chunk too big
        current.append(part)
    if current:
        chunks.append(" ".join(current))
    return chunks


_SPLITTERS: dict[str, Callable[[str, int, int], list[str]]] = {
    "fixed": _fixed_split,
    "sentence": _sentence_split,
}


def chunk_pages(
    pages: Iterable[tuple[int, str]], config: ChunkConfig
) -> list[TextChunk]:
    """Chunk each page separately so every chunk has exactly one page number."""
    splitter = _SPLITTERS[config.strategy]
    chunks: list[TextChunk] = []
    for page_number, text in pages:
        for piece in splitter(text, config.chunk_size, config.overlap):
            if len(piece) < MIN_CHUNK_CHARS:
                continue
            chunks.append(
                TextChunk(
                    page_number=page_number,
                    chunk_index=len(chunks),
                    text=piece,
                )
            )
    return chunks