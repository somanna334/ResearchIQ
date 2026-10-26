"""Request IDs, structured logging, access log and security headers.

Written as plain ASGI middleware (not BaseHTTPMiddleware) so it adds almost no
overhead and cannot interfere with streaming or background work.
"""
import contextvars
import json
import logging
import re
import time
import uuid
from datetime import datetime, timezone

from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default="-"
)
access_logger = logging.getLogger("researchiq.access")

# A caller-supplied ID is only trusted if it is short and boring. Otherwise a
# client could put newlines or huge strings into our logs ("log injection").
_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]{8,64}$")

# Swagger UI loads scripts from a CDN, so the strict API policy would blank it.
_DOC_PATHS = {"/docs", "/redoc", "/openapi.json", "/docs/oauth2-redirect"}


# ---------------------------------------------------------------- logging


def install_record_factory() -> None:
    """Give EVERY log record a request_id, whichever handler prints it."""
    current = logging.getLogRecordFactory()
    if getattr(current, "_researchiq", False):
        return

    def factory(*args, **kwargs) -> logging.LogRecord:  # type: ignore[no-untyped-def]
        record = current(*args, **kwargs)
        record.request_id = request_id_var.get()
        return record

    factory._researchiq = True  # type: ignore[attr-defined]
    logging.setLogRecordFactory(factory)


class JsonFormatter(logging.Formatter):
    """One JSON object per line: easy for log tools to filter by request_id."""

    EXTRA_FIELDS = ("method", "path", "status", "duration_ms")

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname,
            "logger": record.name,
            "request_id": getattr(record, "request_id", "-"),
            "message": record.getMessage(),
        }
        for field in self.EXTRA_FIELDS:
            if hasattr(record, field):
                payload[field] = getattr(record, field)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str = "INFO", fmt: str = "text") -> None:
    install_record_factory()
    handler = logging.StreamHandler()
    handler._researchiq = True  # type: ignore[attr-defined]
    if fmt == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s [%(request_id)s] %(name)s: %(message)s"
            )
        )
    root = logging.getLogger()
    for existing in list(root.handlers):
        if getattr(existing, "_researchiq", False):
            root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level.upper())


# ------------------------------------------------------------ middleware


class RequestContextMiddleware:
    """Assigns a request ID, returns it as X-Request-ID, logs one line per request.

    The log line has the method, the path WITHOUT its query string, the status and
    the duration. It never includes headers or bodies, so passwords, bearer tokens
    and document text cannot end up in it.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        supplied = dict(scope["headers"]).get(b"x-request-id", b"").decode("latin-1")
        request_id = supplied if _SAFE_ID.match(supplied) else uuid.uuid4().hex
        token = request_id_var.set(request_id)
        started = time.perf_counter()
        status = 500  # what the caller sees if the app crashes before responding

        async def send_with_id(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                message.setdefault("headers", [])
                MutableHeaders(scope=message)["X-Request-ID"] = request_id
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        finally:
            access_logger.info(
                "request",
                extra={
                    "method": scope["method"],
                    "path": scope["path"],
                    "status": status,
                    "duration_ms": round((time.perf_counter() - started) * 1000, 1),
                },
            )
            request_id_var.reset(token)


class SecurityHeadersMiddleware:
    """Defensive headers on every API response.

    The API returns JSON only, so the content policy can forbid everything.
    The frontend's own headers are set by whatever serves its files.
    """

    def __init__(self, app: ASGIApp, hsts: bool = False) -> None:
        self.app = app
        self.hsts = hsts

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        is_docs = scope["path"] in _DOC_PATHS

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                message.setdefault("headers", [])
                headers = MutableHeaders(scope=message)
                headers.setdefault("X-Content-Type-Options", "nosniff")
                headers.setdefault("X-Frame-Options", "DENY")
                headers.setdefault("Referrer-Policy", "no-referrer")
                headers.setdefault(
                    "Permissions-Policy", "camera=(), microphone=(), geolocation=()"
                )
                headers.setdefault("Cache-Control", "no-store")
                if not is_docs:
                    headers.setdefault(
                        "Content-Security-Policy",
                        "default-src 'none'; frame-ancestors 'none'",
                    )
                if self.hsts:
                    headers.setdefault(
                        "Strict-Transport-Security",
                        "max-age=31536000; includeSubDomains",
                    )
            await send(message)

        await self.app(scope, receive, send_with_headers)
