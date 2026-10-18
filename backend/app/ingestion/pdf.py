import logging
from dataclasses import dataclass
from pathlib import Path
from pathlib import Path
import logging
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from pypdf import PdfReader

logger = logging.getLogger(__name__)

# A page with fewer extracted characters than this is treated as "no text"
# (scanned pages often yield a few stray characters).
MIN_TEXT_CHARS = 10


class PdfExtractionError(Exception):
    """Raised when a PDF cannot be turned into page text."""


@dataclass(frozen=True)
class ExtractedPage:
    page_number: int  # 1-based, same as a PDF viewer's page counter
    text: str

    @property
    def has_text(self) -> bool:
        return len(self.text) >= MIN_TEXT_CHARS


@dataclass(frozen=True)
class ExtractionResult:
    pages: list[ExtractedPage]

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def pages_with_text(self) -> int:
        return sum(page.has_text for page in self.pages)


def _normalize(text: str) -> str:
       # NFKC turns ligatures such as "ﬁ" into "fi" so keyword search can match them.
       text = unicodedata.normalize("NFKC", text)
       # NUL characters are legal in some PDFs but cannot be stored in PostgreSQL text.
       return text.replace("\x00", "").strip()


def extract_pdf(path: Path, max_pages: int) -> ExtractionResult:
    """Extract text from every page of a PDF, keeping page numbers."""
    try:
        reader = PdfReader(str(path))
        encrypted = reader.is_encrypted
    except Exception as exc:
        raise PdfExtractionError(
            "File could not be read as a PDF (it may be corrupted)"
        ) from exc

    if encrypted:
        raise PdfExtractionError("Encrypted or password-protected PDFs are not supported")

    try:
        total_pages = len(reader.pages)
    except Exception as exc:
        raise PdfExtractionError(
            "File could not be read as a PDF (it may be corrupted)"
        ) from exc

    if total_pages == 0:
        raise PdfExtractionError("PDF contains no pages")
    if total_pages > max_pages:
        raise PdfExtractionError(
            f"PDF has {total_pages} pages; the limit is {max_pages}"
        )

    pages: list[ExtractedPage] = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            # One bad page should not lose the whole document.
            logger.warning("Could not extract text from page %d", page_number)
            text = ""
        pages.append(ExtractedPage(page_number=page_number, text=_normalize(text)))

    return ExtractionResult(pages=pages)