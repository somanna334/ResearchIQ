import hashlib
import re
import secrets
from functools import lru_cache

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError

from app.config import Settings

EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


@lru_cache
def _hasher(time_cost: int, memory_kib: int, parallelism: int) -> PasswordHasher:
    return PasswordHasher(
        time_cost=time_cost, memory_cost=memory_kib, parallelism=parallelism
    )


def _for(settings: Settings) -> PasswordHasher:
    return _hasher(
        settings.argon2_time_cost, settings.argon2_memory_kib, settings.argon2_parallelism
    )


def hash_password(settings: Settings, password: str) -> str:
    return _for(settings).hash(password)


def verify_password(settings: Settings, stored_hash: str, password: str) -> bool:
    """True only for a correct password. A malformed stored hash is just 'false'."""
    try:
        return _for(settings).verify(stored_hash, password)
    except (VerificationError, InvalidHashError):
        return False


def needs_rehash(settings: Settings, stored_hash: str) -> bool:
    """True when the hash was made with weaker parameters than the current settings."""
    try:
        return _for(settings).check_needs_rehash(stored_hash)
    except InvalidHashError:
        return False


@lru_cache
def _dummy_hash(time_cost: int, memory_kib: int, parallelism: int) -> str:
    return _hasher(time_cost, memory_kib, parallelism).hash("dummy-password-for-timing")


def dummy_hash(settings: Settings) -> str:
    """Verified against when the email is unknown, so timing doesn't reveal it."""
    return _dummy_hash(
        settings.argon2_time_cost, settings.argon2_memory_kib, settings.argon2_parallelism
    )


def normalize_email(email: str) -> str:
    return email.strip().lower()


def is_valid_email(email: str) -> bool:
    return len(email) <= 254 and EMAIL_PATTERN.fullmatch(email) is not None


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def new_session_token() -> tuple[str, str]:
    """Returns (token for the client, hash to store)."""
    token = secrets.token_urlsafe(32)
    return token, hash_token(token)