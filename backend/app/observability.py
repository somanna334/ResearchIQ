"""Request IDs, structured JSON logging, and security headers.

Pure ASGI middleware (not BaseHTTPMiddleware) so the request-id contextvar
is reliably visible in route handlers and log calls.
"""
from __future__ import annotations

import contextvars
import json
import logging
import re
import time
import uuid
from datetime import datetime, timezone

from starlette.datastructures import MutableHeaders

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default="-"
)

ACCESS_LOGGER = "researchiq.access"
_SAFE_REQUEST_ID = re.compile(r"[A-Za-z0-9._-]{8,64}")
_DOCS_PATHS = ("/docs", "/redoc", "/openapi.json")


class JsonFormatter(logging.Formatter):
    """One JSON object per line. Extra structured data goes in `record.fields`."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": request_id_var.get(),
        }
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            payload.update(fields)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    """Install the JSON formatter on the root logger (idempotent)."""
    root = logging.getLogger()
    root.setLevel(level.upper())
    for handler in root.handlers:
        if getattr(handler, "_researchiq_json", False):
            return
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    handler._researchiq_json = True  # type: ignore[attr-defined]
    root.addHandler(handler)


class RequestContextMiddleware:
    """Assign a request ID, return it as a header, and write one access-log line."""

    def __init__(self, app):
        self.app = app
        self.logger = logging.getLogger(ACCESS_LOGGER)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        incoming = dict(scope["headers"]).get(b"x-request-id", b"").decode("latin-1")
        request_id = incoming if _SAFE_REQUEST_ID.fullmatch(incoming) else uuid.uuid4().hex
        token = request_id_var.set(request_id)
        started = time.perf_counter()
        status = 500

        async def send_with_id(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                MutableHeaders(scope=message)["X-Request-ID"] = request_id
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        finally:
            client = scope.get("client")
            self.logger.info(
                "request",
                extra={
                    "fields": {
                        "method": scope.get("method"),
                        "path": scope.get("path"),  # never the query string
                        "status": status,
                        "duration_ms": round((time.perf_counter() - started) * 1000, 1),
                        "client_ip": client[0] if client else None,
                    }
                },
            )
            request_id_var.reset(token)


class SecurityHeadersMiddleware:
    """Add conservative security headers to every HTTP response."""

    def __init__(self, app, enable_hsts: bool = False):
        self.app = app
        self.enable_hsts = enable_hsts

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        is_docs = path.startswith(_DOCS_PATHS)

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                headers.setdefault("X-Content-Type-Options", "nosniff")
                headers.setdefault("X-Frame-Options", "DENY")
                headers.setdefault("Referrer-Policy", "no-referrer")
                headers.setdefault("Cross-Origin-Resource-Policy", "same-site")
                if not is_docs:
                    # Swagger UI loads scripts from a CDN, so skip CSP on docs pages.
                    headers.setdefault(
                        "Content-Security-Policy",
                        "default-src 'none'; frame-ancestors 'none'",
                    )
                if self.enable_hsts:
                    headers.setdefault(
                        "Strict-Transport-Security",
                        "max-age=31536000; includeSubDomains",
                    )
            await send(message)

        await self.app(scope, receive, send_with_headers)
