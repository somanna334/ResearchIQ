import hmac
from datetime import datetime, timedelta, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.auth.deps import CurrentUser, bearer, is_admin
from app.auth.security import (
    dummy_hash,
    hash_password,
    hash_token,
    is_valid_email,
    needs_rehash,
    new_session_token,
    normalize_email,
    verify_password,
)
from app.config import Settings, get_settings
from app.database import get_db
from app.models import AuthSession, User
from app.ratelimit import LoginThrottle, auth_rate_limit, client_ip, too_many

router = APIRouter(prefix="/auth", tags=["auth"])


class RegisterRequest(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=10, max_length=128)
    invite_code: str | None = Field(None, max_length=128)


class LoginRequest(BaseModel):
    email: str = Field(min_length=1, max_length=254)
    password: str = Field(min_length=1, max_length=128)


class UserOut(BaseModel):
    id: str
    email: str
    is_admin: bool


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_at: datetime
    user: UserOut


def _issue_token(db: Session, settings: Settings, user: User) -> TokenOut:
    now = datetime.now(timezone.utc)
    db.execute(delete(AuthSession).where(AuthSession.expires_at <= now))  # tidy up
    token, token_hash = new_session_token()
    expires_at = now + timedelta(hours=settings.session_ttl_hours)
    db.add(AuthSession(user_id=user.id, token_hash=token_hash, expires_at=expires_at))
    db.commit()
    return TokenOut(
        access_token=token,
        expires_at=expires_at,
        user=UserOut(id=user.id, email=user.email, is_admin=is_admin(user, settings)),
    )


def _check_invite(settings: Settings, supplied: str | None) -> None:
    """Constant-time comparison; the same answer for 'missing' and 'wrong'."""
    if settings.invite_code is None:
        return
    expected = settings.invite_code.get_secret_value().encode()
    if not hmac.compare_digest((supplied or "").encode(), expected):
        raise HTTPException(status_code=403, detail="A valid invite code is required")


@router.post(
    "/register",
    response_model=TokenOut,
    status_code=201,
    dependencies=[Depends(auth_rate_limit)],
)
def register(
    body: RegisterRequest,
    db: Annotated[Session, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> TokenOut:
    if not settings.allow_registration:
        raise HTTPException(status_code=403, detail="Registration is disabled")
    # Before anything that reveals whether an email is taken.
    _check_invite(settings, body.invite_code)
    email = normalize_email(body.email)
    if not is_valid_email(email):
        raise HTTPException(status_code=422, detail="Enter a valid email address")
    if db.scalar(select(User.id).where(User.email == email)) is not None:
        raise HTTPException(status_code=409, detail="An account with this email already exists")

    user = User(email=email, password_hash=hash_password(settings, body.password))
    db.add(user)
    try:
        db.commit()
    except IntegrityError:  # two registrations raced past the check above
        db.rollback()
        raise HTTPException(
            status_code=409, detail="An account with this email already exists"
        ) from None
    return _issue_token(db, settings, user)


@router.post(
    "/login", response_model=TokenOut, dependencies=[Depends(auth_rate_limit)]
)
def login(
    body: LoginRequest,
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> TokenOut:
    email = normalize_email(body.email)
    throttle = LoginThrottle(settings, client_ip(request, settings), email)
    wait = throttle.wait_seconds()
    if wait:
        # Checked before any password work, and the same for real and unknown
        # emails, so it reveals nothing about which accounts exist.
        raise too_many(wait, "failed sign-in attempts")

    failure = HTTPException(status_code=401, detail="Incorrect email or password")
    user = db.scalar(select(User).where(User.email == email))
    if user is None:
        verify_password(settings, dummy_hash(settings), body.password)  # equalize timing
        throttle.record_failure()
        raise failure
    if not verify_password(settings, user.password_hash, body.password):
        throttle.record_failure()
        raise failure
    throttle.record_success()
    if needs_rehash(settings, user.password_hash):
        user.password_hash = hash_password(settings, body.password)
        db.commit()
    return _issue_token(db, settings, user)


@router.post("/logout", status_code=204)
def logout(
    user: CurrentUser,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
    db: Annotated[Session, Depends(get_db)],
) -> None:
    """Ends this session only; the user's other sessions stay valid."""
    if credentials is not None:
        db.execute(
            delete(AuthSession).where(
                AuthSession.token_hash == hash_token(credentials.credentials)
            )
        )
        db.commit()


@router.get("/me", response_model=UserOut)
def me(user: CurrentUser, settings: Annotated[Settings, Depends(get_settings)]) -> UserOut:
    return UserOut(id=user.id, email=user.email, is_admin=is_admin(user, settings))
