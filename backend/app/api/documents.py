import hashlib
import logging
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, cast

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from sqlalchemy import Engine, func, select
from sqlalchemy.orm import Session

from app.auth.deps import CurrentUser, get_owned_document
from app.config import Settings, get_settings
from app.database import get_db
from app.embeddings.embedder import Embedder, get_embedder_factory
from app.ingestion.pipeline import process_document
from app.models import Document, Page
from app.ratelimit import upload_rate_limit
from app.schemas import DocumentOut, PageOut

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])

CHUNK_SIZE = 1024 * 1024  # read uploads 1 MB at a time

EmbedderFactory = Callable[[], Embedder]


def _clean_filename(raw: str | None) -> str:
    # Some clients send full Windows paths; keep only the last segment.
    return (raw or "").replace("\\", "/").split("/")[-1].strip()[:255]


def _save_upload(upload: UploadFile, dest: Path, max_bytes: int) -> tuple[int, str]:
    """Stream the upload to disk, enforcing the size limit. Returns (size, sha256)."""
    digest = hashlib.sha256()
    size = 0
    with dest.open("wb") as out:
        while chunk := upload.file.read(CHUNK_SIZE):
            size += len(chunk)
            if size > max_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=f"File exceeds the {max_bytes // (1024 * 1024)} MB limit",
                )
            digest.update(chunk)
            out.write(chunk)
    return size, digest.hexdigest()


def _check_pdf_header(path: Path) -> None:
    with path.open("rb") as f:
        if f.read(5) != b"%PDF-":
            raise HTTPException(status_code=400, detail="File is not a valid PDF")


def _bytes_in_use(db: Session, user_id: str, settings: Settings) -> int:
    """Checks the document-count quota and returns the bytes this user already stores.

    Both quota checks are read-then-write, so two simultaneous uploads can overshoot
    a limit slightly. That is acceptable for a quota (it is not a security boundary).
    """
    count, used = db.execute(
        select(func.count(Document.id), func.coalesce(func.sum(Document.size_bytes), 0))
        .where(Document.owner_id == user_id)
    ).one()
    if count >= settings.max_documents_per_user:
        raise HTTPException(
            status_code=403,
            detail=(
                f"Document limit reached ({settings.max_documents_per_user}). "
                "Delete a document to upload another."
            ),
        )
    return int(used)


def _start_indexing(
    db: Session,
    background: BackgroundTasks,
    settings: Settings,
    embedder_factory: EmbedderFactory,
    document_id: str,
) -> None:
    """Hand a 'processing' document to the indexing pipeline.

    Background mode: schedule it and return (the task runs after the response).
    Inline mode (tests, background_indexing=False): run it now, without embedding,
    which is the behaviour from before Milestone 9C.
    """
    engine = cast(Engine, db.get_bind())
    if settings.background_indexing:
        background.add_task(
            process_document, engine, settings, embedder_factory, document_id, True
        )
    else:
        process_document(engine, settings, embedder_factory, document_id, False)


@router.post(
    "",
    response_model=DocumentOut,
    status_code=201,
    dependencies=[Depends(upload_rate_limit)],
)
def upload_document(
    file: Annotated[UploadFile, File()],
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
    background: BackgroundTasks,
    embedder_factory: Annotated[EmbedderFactory, Depends(get_embedder_factory)],
) -> Document:
    filename = _clean_filename(file.filename)
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are accepted")

    used_bytes = _bytes_in_use(db, user.id, settings)

    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    document_id = str(uuid.uuid4())
    # The stored name comes from us, never from the client.
    path = settings.upload_dir / f"{document_id}.pdf"

    try:
        size, sha256 = _save_upload(file, path, settings.max_upload_bytes)
        _check_pdf_header(path)
        if used_bytes + size > settings.max_storage_bytes_per_user:
            raise HTTPException(
                status_code=403,
                detail=(
                    f"Storage limit reached ({settings.max_storage_mb_per_user:g} MB). "
                    "Delete a document to free space."
                ),
            )
    except Exception:
        path.unlink(missing_ok=True)
        raise

    document = Document(
        id=document_id,
        owner_id=user.id,
        filename=filename,
        size_bytes=size,
        sha256=sha256,
        status="processing",
    )
    db.add(document)
    db.commit()

    _start_indexing(db, background, settings, embedder_factory, document_id)
    db.refresh(document)  # background mode: still "processing"; inline: the final state
    return document


@router.post(
    "/{document_id}/retry",
    response_model=DocumentOut,
    status_code=202,
    dependencies=[Depends(upload_rate_limit)],
)
def retry_document(
    document_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
    background: BackgroundTasks,
    embedder_factory: Annotated[EmbedderFactory, Depends(get_embedder_factory)],
) -> Document:
    """Index a failed document again, from the file that is still on disk."""
    document = get_owned_document(db, user, document_id)  # foreign ids: 404
    path = settings.upload_dir / f"{document_id}.pdf"
    if document.status != "failed":
        raise HTTPException(
            status_code=409, detail="Only a failed document can be retried"
        )
    if not path.exists():
        raise HTTPException(
            status_code=409,
            detail="The original file is no longer on the server. Upload it again.",
        )

    document.status = "processing"
    document.error_message = None
    db.commit()

    _start_indexing(db, background, settings, embedder_factory, document_id)
    db.refresh(document)
    return document


@router.get("", response_model=list[DocumentOut])
def list_documents(
    user: CurrentUser, db: Annotated[Session, Depends(get_db)]
) -> list[Document]:
    return list(
        db.scalars(
            select(Document)
            .where(Document.owner_id == user.id)
            .order_by(Document.created_at.desc())
        )
    )


@router.get("/{document_id}", response_model=DocumentOut)
def get_document(
    document_id: str, user: CurrentUser, db: Annotated[Session, Depends(get_db)]
) -> Document:
    return get_owned_document(db, user, document_id)


@router.get("/{document_id}/pages/{page_number}", response_model=PageOut)
def get_page(
    document_id: str,
    page_number: int,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> Page:
    get_owned_document(db, user, document_id)
    page = db.scalar(
        select(Page).where(
            Page.document_id == document_id, Page.page_number == page_number
        )
    )
    if page is None:
        raise HTTPException(status_code=404, detail="Page not found")
    return page


@router.delete("/{document_id}", status_code=204)
def delete_document(
    document_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> None:
    document = get_owned_document(db, user, document_id)
    db.delete(document)  # pages and chunks (and their vectors) go too
    db.commit()
    (settings.upload_dir / f"{document_id}.pdf").unlink(missing_ok=True)
