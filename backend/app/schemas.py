from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

DocumentStatus = Literal["processing", "ready", "no_text", "failed"]


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    size_bytes: int
    sha256: str
    status: DocumentStatus
    error_message: str | None
    page_count: int
    pages_with_text: int
    created_at: datetime


class PageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    page_number: int
    text: str
    char_count: int