from datetime import datetime, timezone
from typing import Annotated

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.auth.security import hash_token
from app.config import Settings, get_settings
from app.database import get_db
from app.models import AuthSession, Document, User

bearer = HTTPBearer(auto_error=False)


def _unauthorized() -> HTTPException:
    return HTTPException(
        status_code=401,
        detail="Not authenticated",
        headers={"WWW-Authenticate": "Bearer"},
    )


def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
    db: Annotated[Session, Depends(get_db)],
) -> User:
    if credentials is None or not credentials.credentials:
        raise _unauthorized()
    user = db.scalar(
        select(User)
        .join(AuthSession, AuthSession.user_id == User.id)
        .where(
            AuthSession.token_hash == hash_token(credentials.credentials),
            AuthSession.expires_at > datetime.now(timezone.utc),
        )
    )
    if user is None:
        raise _unauthorized()
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]


def is_admin(user: User, settings: Settings) -> bool:
    return user.email in settings.admin_email_list


def require_admin(
    user: CurrentUser, settings: Annotated[Settings, Depends(get_settings)]
) -> User:
    if not is_admin(user, settings):
        raise HTTPException(status_code=403, detail="Administrator access required")
    return user


def get_owned_document(db: Session, user: User, document_id: str) -> Document:
    """A document the user owns. Unknown and foreign IDs look identical (404)."""
    document = db.scalar(
        select(Document).where(Document.id == document_id, Document.owner_id == user.id)
    )
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return document


def resolve_document_ids(
    db: Session, user: User, requested: list[str] | None
) -> list[str]:
    """The document IDs a search may touch: always an explicit list of the user's own.

    None or an empty list means 'all of my documents'. Naming a document the user
    doesn't own is a 404. The result can be empty, and retrieval treats an empty
    list as 'match nothing', never as 'no filter'.
    """
    owned = set(db.scalars(select(Document.id).where(Document.owner_id == user.id)))
    if not requested:
        return sorted(owned)
    if any(document_id not in owned for document_id in requested):
        raise HTTPException(status_code=404, detail="Document not found")
    return list(dict.fromkeys(requested))