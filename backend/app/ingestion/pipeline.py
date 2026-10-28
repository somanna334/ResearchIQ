"""Background document indexing.

`process_document` does the slow work (extract, chunk, embed) outside the upload
request. Rules it follows:

* It opens its OWN database session. The request's session is closed by the time a
  background task runs, so it must never be reused here.
* Only one document (``index_concurrency``) is indexed at a time, because
  extraction and embedding are memory-hungry.
* "ready" is set last, so "ready" always means "searchable".
* It never raises. A failure marks the document ``failed`` with a fixed, harmless
  message; the traceback goes to the server log only.
"""

import logging
import threading
from collections.abc import Callable

from sqlalchemy import Engine, update
from sqlalchemy.orm import Session

from app.chunking.service import build_chunks, default_chunk_config
from app.config import Settings
from app.embeddings.embedder import Embedder
from app.embeddings.service import embed_chunk_set
from app.ingestion.pdf import PdfExtractionError, extract_pdf
from app.models import Document, Page

logger = logging.getLogger(__name__)

EmbedderFactory = Callable[[], Embedder]

FAILED_MESSAGE = "Indexing failed. Use Retry to try again."
MISSING_FILE_MESSAGE = "The uploaded file is no longer on the server. Upload it again."
INTERRUPTED_MESSAGE = "Indexing was interrupted by a server restart. Use Retry to try again."
NO_TEXT_MESSAGE = (
    "No extractable text found. This looks like a scanned PDF; OCR is not supported yet."
)

_semaphores: dict[int, threading.BoundedSemaphore] = {}
_semaphores_lock = threading.Lock()


def _semaphore_for(limit: int) -> threading.BoundedSemaphore:
    limit = max(1, limit)
    with _semaphores_lock:
        semaphore = _semaphores.get(limit)
        if semaphore is None:
            semaphore = _semaphores[limit] = threading.BoundedSemaphore(limit)
        return semaphore


def process_document(
    engine: Engine,
    settings: Settings,
    embedder_factory: EmbedderFactory,
    document_id: str,
    embed: bool = True,
) -> None:
    """Index one document. Never raises."""
    try:
        with _semaphore_for(settings.index_concurrency):
            _index(engine, settings, embedder_factory, document_id, embed)
    except Exception:  # noqa: BLE001 - a background task must not crash the worker
        logger.exception("document_indexing_failed id=%s", document_id)
        _mark_failed(engine, document_id, FAILED_MESSAGE)


def _index(
    engine: Engine,
    settings: Settings,
    embedder_factory: EmbedderFactory,
    document_id: str,
    embed: bool,
) -> None:
    with Session(engine) as db:
        document = db.get(Document, document_id)
        if document is None:  # deleted while it was waiting in the queue
            return

        path = settings.upload_dir / f"{document_id}.pdf"
        if not path.exists():
            document.status = "failed"
            document.error_message = MISSING_FILE_MESSAGE
            db.commit()
            return

        try:
            result = extract_pdf(path, settings.max_pages)
        except PdfExtractionError as exc:
            # These messages are fixed strings written by us ("encrypted", "N pages...").
            document.status = "failed"
            document.error_message = str(exc)
            db.commit()
            path.unlink(missing_ok=True)  # retrying the same bad file cannot help
            logger.info("document_failed id=%s reason=%s", document_id, exc)
            return

        # Clear pages from an earlier attempt and flush BEFORE adding the new ones:
        # inserts run before deletes inside one flush, which would otherwise hit the
        # unique (document_id, page_number) constraint on a retry.
        document.pages = []
        db.flush()
        document.pages = [
            Page(page_number=p.page_number, text=p.text, char_count=len(p.text))
            for p in result.pages
        ]
        document.page_count = result.page_count
        document.pages_with_text = result.pages_with_text

        if result.pages_with_text == 0:
            document.status = "no_text"
            document.error_message = NO_TEXT_MESSAGE
            db.commit()
            logger.info("document_ingested id=%s status=no_text", document_id)
            return

        db.commit()  # pages are saved; status is still "processing"
        config = default_chunk_config(settings)
        build_chunks(db, document, config)
        if embed:
            # The (slow) model load happens here, in the worker thread.
            embedder = embedder_factory()
            embed_chunk_set(db, embedder, document_id, config.key)

        document.status = "ready"
        document.error_message = None
        db.commit()
        logger.info(
            "document_ingested id=%s status=ready pages=%d pages_with_text=%d embedded=%s",
            document_id,
            document.page_count,
            document.pages_with_text,
            embed,
        )


def _mark_failed(engine: Engine, document_id: str, message: str) -> None:
    try:
        with Session(engine) as db:
            db.execute(
                update(Document)
                .where(Document.id == document_id)
                .values(status="failed", error_message=message)
            )
            db.commit()
    except Exception:  # noqa: BLE001
        logger.exception("could_not_mark_failed id=%s", document_id)


def recover_interrupted_documents(engine: Engine) -> int:
    """Mark documents left in 'processing' by a restart as failed. Returns how many.

    Correct only for a single server process: with several workers, one worker's
    restart would wrongly fail documents another worker is still indexing.
    """
    with Session(engine) as db:
        result = db.execute(
            update(Document)
            .where(Document.status == "processing")
            .values(status="failed", error_message=INTERRUPTED_MESSAGE)
        )
        db.commit()
        count = int(result.rowcount or 0)  # type: ignore[attr-defined]
    if count:
        logger.warning("recovered_interrupted_documents count=%d", count)
    return count
