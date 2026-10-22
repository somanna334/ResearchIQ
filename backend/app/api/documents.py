import hashlib
import logging
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.auth.deps import CurrentUser, get_owned_document
from app.chunking.service import build_chunks, default_chunk_config
from app.config import Settings, get_settings
from app.database import get_db
from app.ingestion.pdf import PdfExtractionError, extract_pdf
from app.models import Document, Page
from app.schemas import DocumentOut, PageOut

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])

CHUNK_SIZE = 1024 * 1024  # read uploads 1 MB at a time


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


@router.post("", response_model=DocumentOut, status_code=201)
def upload_document(
    file: Annotated[UploadFile, File()],
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> Document:
    filename = _clean_filename(file.filename)
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are accepted")

    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    document_id = str(uuid.uuid4())
    # The stored name comes from us, never from the client.
    path = settings.upload_dir / f"{document_id}.pdf"

    try:
        size, sha256 = _save_upload(file, path, settings.max_upload_bytes)
        _check_pdf_header(path)
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

    try:
        result = extract_pdf(path, settings.max_pages)
    except PdfExtractionError as exc:
        document.status = "failed"
        document.error_message = str(exc)
        path.unlink(missing_ok=True)
        db.commit()
        db.refresh(document)
        logger.info("document_failed id=%s reason=%s", document_id, exc)
        return document

    document.pages = [
        Page(page_number=p.page_number, text=p.text, char_count=len(p.text))
        for p in result.pages
    ]
    document.page_count = result.page_count
    document.pages_with_text = result.pages_with_text
    if result.pages_with_text == 0:
        document.status = "no_text"
        document.error_message = (
            "No extractable text found. This looks like a scanned PDF; "
            "OCR is not supported yet."
        )
    else:
        document.status = "ready"
    db.commit()
    if document.status == "ready":
        build_chunks(db, document, default_chunk_config(settings))
    db.refresh(document)
    logger.info(
        "document_ingested id=%s status=%s pages=%d pages_with_text=%d",
        document_id,
        document.status,
        document.page_count,
        document.pages_with_text,
    )
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