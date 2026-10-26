"""Rate limiting and login throttling.

State lives in this process's memory. That is fine for one uvicorn worker (this
project's setup) but NOT for several workers or machines: each would count
separately. For that, keep the same interface and back it with Redis.
"""
import math
import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Annotated

from fastapi import Depends, HTTPException, Request

from app.auth.deps import CurrentUser
from app.config import Settings, get_settings

SettingsDep = Annotated[Settings, Depends(get_settings)]


class SlidingWindow:
    """Counts events per key over the last `window` seconds."""

    def __init__(
        self,
        limit: int,
        window: float,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.limit = limit
        self.window = window
        # Looked up at call time, so tests can move time forward.
        self._clock = clock or (lambda: time.monotonic())
        self._events: dict[str, deque[float]] = {}
        self._lock = threading.Lock()
        self._last_sweep = self._clock()

    def _live(self, key: str, now: float) -> deque[float] | None:
        events = self._events.get(key)
        if events is None:
            return None
        cutoff = now - self.window
        while events and events[0] <= cutoff:
            events.popleft()
        if not events:
            del self._events[key]
            return None
        return events

    def _sweep(self, now: float) -> None:
        """Forget keys that went quiet, so memory can't grow without bound."""
        if now - self._last_sweep < self.window:
            return
        self._last_sweep = now
        for key in list(self._events):
            self._live(key, now)

    def _wait(self, events: deque[float] | None, now: float) -> int:
        if not events or len(events) < self.limit:
            return 0
        frees_at = events[len(events) - self.limit] + self.window
        return max(1, math.ceil(frees_at - now))

    def count(self, key: str) -> int:
        with self._lock:
            events = self._live(key, self._clock())
            return len(events) if events else 0

    def retry_after(self, key: str) -> int:
        """Seconds until `key` may act again; 0 when it is not blocked."""
        with self._lock:
            now = self._clock()
            return self._wait(self._live(key, now), now)

    def hit(self, key: str) -> tuple[bool, int]:
        """Record an event if allowed. Returns (allowed, retry_after_seconds)."""
        with self._lock:
            now = self._clock()
            self._sweep(now)
            events = self._live(key, now)
            wait = self._wait(events, now)
            if wait:
                return False, wait
            self._events.setdefault(key, deque()).append(now)
            return True, 0

    def add(self, key: str) -> None:
        """Record an event unconditionally (used for failed logins)."""
        with self._lock:
            now = self._clock()
            self._sweep(now)
            self._events.setdefault(key, deque()).append(now)

    def clear(self, key: str) -> None:
        with self._lock:
            self._events.pop(key, None)


_registry: dict[str, SlidingWindow] = {}
_registry_lock = threading.Lock()


def get_window(name: str, limit: int, window: float) -> SlidingWindow:
    """One shared window per name. Changing the settings builds a fresh one."""
    with _registry_lock:
        existing = _registry.get(name)
        if existing is None or (existing.limit, existing.window) != (limit, window):
            existing = SlidingWindow(limit, window)
            _registry[name] = existing
        return existing


def reset_limiters() -> None:
    with _registry_lock:
        _registry.clear()


def client_ip(request: Request, settings: Settings) -> str:
    """The caller's address.

    Behind N trusted reverse proxies, each appends the address it saw to
    X-Forwarded-For, so the real client is the Nth entry from the RIGHT. Anything
    further left was supplied by the client and can be forged. With
    trusted_proxy_count=0 (the default) the header is ignored entirely.
    """
    if settings.trusted_proxy_count > 0:
        parts = [
            p.strip()
            for p in request.headers.get("x-forwarded-for", "").split(",")
            if p.strip()
        ]
        if len(parts) >= settings.trusted_proxy_count:
            return parts[-settings.trusted_proxy_count]
    return request.client.host if request.client else "unknown"


def describe_wait(seconds: int) -> str:
    if seconds < 90:
        return f"{seconds} second{'s' if seconds != 1 else ''}"
    minutes = math.ceil(seconds / 60)
    return f"{minutes} minutes"


def too_many(retry_after: int, what: str = "requests") -> HTTPException:
    return HTTPException(
        status_code=429,
        detail=f"Too many {what}. Try again in {describe_wait(retry_after)}.",
        headers={"Retry-After": str(retry_after)},
    )


def _enforce(name: str, limit: int, key: str, window: float = 60) -> None:
    if limit <= 0:  # 0 turns a limit off (handy in local development)
        return
    allowed, wait = get_window(name, limit, window).hit(key)
    if not allowed:
        raise too_many(wait)


def auth_rate_limit(request: Request, settings: SettingsDep) -> None:
    """Register and login: per client address."""
    _enforce("auth", settings.auth_requests_per_minute, f"ip:{client_ip(request, settings)}")


def llm_rate_limit(user: CurrentUser, settings: SettingsDep) -> None:
    """Routes that call the LLM: per user. They cost money and quota."""
    _enforce("llm", settings.llm_requests_per_minute, f"user:{user.id}")


def upload_rate_limit(user: CurrentUser, settings: SettingsDep) -> None:
    _enforce("upload", settings.upload_requests_per_minute, f"user:{user.id}")


class LoginThrottle:
    """Counts FAILED sign-ins, per (client, email) and per client.

    Locking an email globally would let anyone lock a victim out by guessing
    wrong passwords for them. Pairing the email with the client address stops
    that. The per-client counter (4x the allowance) covers one machine trying
    many different emails.
    """

    def __init__(self, settings: Settings, ip: str, email: str) -> None:
        limit = max(1, settings.login_max_failures)
        window = settings.login_window_seconds
        self._guards = [
            (get_window("login_pair", limit, window), f"{ip}|{email}"),
            (get_window("login_ip", limit * 4, window), f"ip:{ip}"),
        ]
        self.enabled = settings.login_max_failures > 0

    def wait_seconds(self) -> int:
        if not self.enabled:
            return 0
        return max(window.retry_after(key) for window, key in self._guards)

    def record_failure(self) -> None:
        if self.enabled:
            for window, key in self._guards:
                window.add(key)

    def record_success(self) -> None:
        window, key = self._guards[0]
        window.clear(key)
