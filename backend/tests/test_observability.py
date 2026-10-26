import json
import logging
import re

from fastapi.testclient import TestClient
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route

from app.middleware import JsonFormatter, SecurityHeadersMiddleware

PASSWORD = "correct-horse-battery"


def test_every_response_carries_a_request_id(client):
    response = client.get("/health")

    assert re.fullmatch(r"[0-9a-f]{32}", response.headers["x-request-id"])


def test_a_clean_incoming_id_is_kept_and_a_dirty_one_is_replaced(client):
    kept = client.get("/health", headers={"X-Request-ID": "trace-abc-12345"})
    replaced = client.get("/health", headers={"X-Request-ID": "x y!"})

    assert kept.headers["x-request-id"] == "trace-abc-12345"
    assert re.fullmatch(r"[0-9a-f]{32}", replaced.headers["x-request-id"])


def test_security_headers_are_present_and_docs_stay_usable(client):
    api = client.get("/health").headers
    docs = client.get("/docs").headers

    assert api["x-content-type-options"] == "nosniff"
    assert api["x-frame-options"] == "DENY"
    assert api["referrer-policy"] == "no-referrer"
    assert api["cache-control"] == "no-store"
    assert "frame-ancestors 'none'" in api["content-security-policy"]
    assert "strict-transport-security" not in api  # development: plain http
    assert docs["x-content-type-options"] == "nosniff"
    assert "content-security-policy" not in docs  # Swagger UI needs its CDN scripts


def test_hsts_is_sent_only_when_enabled():
    async def hello(request):
        return PlainTextResponse("hi")

    plain = Starlette(routes=[Route("/", hello)])

    on = TestClient(SecurityHeadersMiddleware(plain, hsts=True)).get("/")
    off = TestClient(SecurityHeadersMiddleware(plain, hsts=False)).get("/")

    assert "max-age=" in on.headers["strict-transport-security"]
    assert "strict-transport-security" not in off.headers


def test_the_access_log_has_context_and_no_secrets(client, caplog):
    caplog.set_level(logging.INFO, logger="researchiq.access")
    token = client.headers["Authorization"].split()[1]

    anon = TestClient(client.app)
    anon.post("/auth/login", json={"email": "admin@example.com", "password": PASSWORD})
    response = client.get("/documents?note=hunter2")

    records = [r for r in caplog.records if r.name == "researchiq.access"]
    last = records[-1]
    assert (last.method, last.path, last.status) == ("GET", "/documents", 200)
    assert last.request_id == response.headers["x-request-id"]
    ours = [r for r in caplog.records if r.name == "researchiq.access"]
    everything = " ".join(f"{r.getMessage()} {r.__dict__}" for r in ours)
    assert token not in everything
    assert PASSWORD not in everything
    assert "hunter2" not in everything  # query strings are not logged


def test_json_log_lines_are_valid_json_with_the_request_id():
    record = logging.LogRecord("t", logging.INFO, __file__, 1, "hello %s", ("x",), None)
    record.request_id = "abc"
    record.status = 200

    data = json.loads(JsonFormatter().format(record))

    assert data["message"] == "hello x"
    assert data["request_id"] == "abc"
    assert data["status"] == 200
    assert data["level"] == "INFO"
