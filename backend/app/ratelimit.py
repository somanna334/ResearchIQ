"""In-memory sliding-window rate limiting and login throttling.

Limitations (deliberate, documented): state is per-process and lost on
restart. Behind several workers, use a shared store such as Redis.
"""
from __future__ import annotations

import math
import threading
import time
from collections import defaultdict, deque
from typing import Callable

from fastapi import Depends, HTTPException, Request

from app.auth.deps import get_current_user

# Only set True when running behind a proxy you control that overwrites
# X-Forwarded-For. Otherwise clients can spoof their IP.
TRUST_PROXY_HEADERS = False

_PRUNE_THRESHOLD = 10_000


def client_ip(request: Request) -> str:
    if TRUST_PROXY_HEADERS:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


class SlidingWindowLimiter:
    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()
        self._clock = clock

    def hit(self, key: str, limit: int, window: float) -> tuple[bool, int]:
        """Record a hit. Returns (allowed, retry_after_seconds)."""
        now = self._clock()
        with self._lock:
            if len(self._hits) > _PRUNE_THRESHOLD:
                self._prune(now, window)
            q = self._hits[key]
            cutoff = now - window
            while q and q[0] <= cutoff:
                q.popleft()
            if len(q) >= limit:
                return False, max(1, math.ceil(q[0] + window - now))
            q.append(now)
            return True, 0

    def _prune(self, now: float, window: float) -> None:
        for key in [k for k, q in self._hits.items() if not q or q[-1] <= now - window]:
            del self._hits[key]

    def clear(self) -> None:
        with self._lock:
            self._hits.clear()


class LoginThrottle:
    """Failure counters per email and per IP; locks out when either is exceeded."""

    def __init__(
        self,
        max_email_failures: int = 5,
        max_ip_failures: int = 20,
        window: float = 900.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.max_email_failures = max_email_failures
        self.max_ip_failures = max_ip_failures
        self.window = window
        self._clock = clock
        self._failures: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    @staticmethod
    def _email_key(email: str) -> str:
        return f"email:{email.strip().lower()}"

    def _live(self, key: str, now: float) -> deque[float]:
        q = self._failures[key]
        while q and q[0] <= now - self.window:
            q.popleft()
        return q

    def check(self, email: str, ip: str) -> None:
        """Raise 429 if this email or IP is currently locked out."""
        now = self._clock()
        with self._lock:
            for key, limit in (
                (self._email_key(email), self.max_email_failures),
                (f"ip:{ip}", self.max_ip_failures),
            ):
                q = self._live(key, now)
                if len(q) >= limit:
                    retry = max(1, math.ceil(q[0] + self.window - now))
                    raise HTTPException(
                        status_code=429,
                        detail="Too many failed attempts. Try again later.",
                        headers={"Retry-After": str(retry)},
                    )

    def record_failure(self, email: str, ip: str) -> None:
        now = self._clock()
        with self._lock:
            self._live(self._email_key(email), now).append(now)
            self._live(f"ip:{ip}", now).append(now)

    def record_success(self, email: str, ip: str) -> None:
        """A good login clears the email counter (not the IP counter)."""
        with self._lock:
            self._failures.pop(self._email_key(email), None)

    def clear(self) -> None:
        with self._lock:
            self._failures.clear()


# Process-wide instances. Part 2 makes the numbers come from Settings.
limiter = SlidingWindowLimiter()
login_throttle = LoginThrottle()


def _reject(retry_after: int) -> HTTPException:
    return HTTPException(
        status_code=429,
        detail="Too many requests. Slow down.",
        headers={"Retry-After": str(retry_after)},
    )


def rate_limit(name: str, limit: int, window: float):
    """Dependency: limit per client IP. Use on public routes (login, register)."""

    def dependency(request: Request) -> None:
        allowed, retry = limiter.hit(f"{name}:ip:{client_ip(request)}", limit, window)
        if not allowed:
            raise _reject(retry)

    return dependency


def user_rate_limit(name: str, limit: int, window: float):
    """Dependency: limit per signed-in user. Use on LLM-calling routes."""

    def dependency(user=Depends(get_current_user)) -> None:
        allowed, retry = limiter.hit(f"{name}:user:{user.id}", limit, window)
        if not allowed:
            raise _reject(retry)

    return dependency


def reset_all() -> None:
    """For tests."""
    limiter.clear()
    login_throttle.clear()
