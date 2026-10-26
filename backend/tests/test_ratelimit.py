import time

from fastapi.testclient import TestClient

from app.llm.base import LLMResult
from app.llm.factory import get_llm_provider
from app.main import app
from app.ratelimit import SlidingWindow

PASSWORD = "correct-horse-battery"
WRONG = "definitely-the-wrong-one"


class FakeLLM:
    name = "fake"
    model_name = "fake-1"

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, system, prompt):
        self.calls += 1
        return LLMResult("Revenue rose 12 percent in 2024 [S1].", 10, 5)


def sign_in(anon: TestClient, email: str, password: str):
    return anon.post("/auth/login", json={"email": email, "password": password})


def upload(who, make_pdf, n: int, pages=None):
    path = make_pdf(pages or [f"Page about topic {n}"], name=f"d{n}.pdf")
    return who.post(
        "/documents", files={"file": (f"d{n}.pdf", path.read_bytes(), "application/pdf")}
    )


# ---- the window itself ------------------------------------------------------


def test_window_blocks_after_the_limit_and_recovers():
    now = [0.0]
    window = SlidingWindow(2, 60, clock=lambda: now[0])

    assert window.hit("k") == (True, 0)
    assert window.hit("k") == (True, 0)
    assert window.hit("k") == (False, 60)
    now[0] = 30
    assert window.hit("k") == (False, 30)
    now[0] = 61
    assert window.hit("k") == (True, 0)


def test_window_keys_are_independent():
    window = SlidingWindow(1, 60, clock=lambda: 0.0)

    assert window.hit("alice")[0] is True
    assert window.hit("alice")[0] is False
    assert window.hit("bob")[0] is True


# ---- login throttling -------------------------------------------------------


def test_wrong_passwords_lock_that_client_and_email(client, tune):
    tune(login_max_failures=3)
    anon = TestClient(app)

    for _ in range(3):
        assert sign_in(anon, "admin@example.com", WRONG).status_code == 401
    blocked = sign_in(anon, "admin@example.com", PASSWORD)  # even the right password

    assert blocked.status_code == 429
    assert int(blocked.headers["Retry-After"]) > 0
    assert "x-request-id" in blocked.headers
    # Someone else's email from the same machine is not locked by this.
    assert sign_in(anon, "other@example.com", WRONG).status_code == 401


def test_a_successful_sign_in_resets_the_failure_count(client, tune):
    tune(login_max_failures=3)
    anon = TestClient(app)

    for _ in range(2):
        assert sign_in(anon, "admin@example.com", WRONG).status_code == 401
    assert sign_in(anon, "admin@example.com", PASSWORD).status_code == 200
    for _ in range(2):  # would be the 3rd and 4th failure without the reset
        assert sign_in(anon, "admin@example.com", WRONG).status_code == 401


def test_unknown_emails_are_throttled_exactly_like_real_ones(client, tune):
    tune(login_max_failures=2)
    anon = TestClient(app)

    for _ in range(2):
        assert sign_in(anon, "ghost@example.com", WRONG).status_code == 401
    assert sign_in(anon, "ghost@example.com", WRONG).status_code == 429


def test_the_lock_lifts_when_the_window_passes(client, tune, monkeypatch):
    tune(login_max_failures=2, login_window_seconds=60)
    real, skew = time.monotonic, [0.0]
    monkeypatch.setattr(time, "monotonic", lambda: real() + skew[0])
    anon = TestClient(app)

    for _ in range(2):
        sign_in(anon, "admin@example.com", WRONG)
    assert sign_in(anon, "admin@example.com", PASSWORD).status_code == 429
    skew[0] += 61

    assert sign_in(anon, "admin@example.com", PASSWORD).status_code == 200


def test_one_client_trying_many_emails_is_blocked(client, tune):
    tune(login_max_failures=1)  # per-client allowance is 4x: four failures
    anon = TestClient(app)

    for n in range(4):
        assert sign_in(anon, f"user{n}@example.com", WRONG).status_code == 401

    assert sign_in(anon, "user99@example.com", WRONG).status_code == 429


# ---- general limits ---------------------------------------------------------


def test_register_and_login_are_limited_per_client(client, tune):
    tune(auth_requests_per_minute=3)
    anon = TestClient(app)

    for _ in range(3):
        assert sign_in(anon, "ghost@example.com", WRONG).status_code == 401

    assert sign_in(anon, "ghost@example.com", WRONG).status_code == 429


def test_forwarded_header_is_ignored_unless_proxies_are_trusted(client, tune):
    tune(auth_requests_per_minute=2)  # trusted_proxy_count stays 0
    anon = TestClient(app)

    for forged in ("1.1.1.1", "2.2.2.2"):
        response = anon.post(
            "/auth/login",
            json={"email": "g@example.com", "password": WRONG},
            headers={"X-Forwarded-For": forged},
        )
        assert response.status_code == 401
    third = anon.post(
        "/auth/login",
        json={"email": "g@example.com", "password": WRONG},
        headers={"X-Forwarded-For": "3.3.3.3"},
    )

    assert third.status_code == 429  # spoofing a new address did not help


def test_trusted_proxy_uses_the_rightmost_forwarded_entry(client, tune):
    tune(auth_requests_per_minute=2, trusted_proxy_count=1)
    anon = TestClient(app)

    def attempt(forwarded: str) -> int:
        return anon.post(
            "/auth/login",
            json={"email": "g@example.com", "password": WRONG},
            headers={"X-Forwarded-For": forwarded},
        ).status_code

    assert attempt("9.9.9.9") == 401
    assert attempt("6.6.6.6, 9.9.9.9") == 401  # forged left entry is ignored
    assert attempt("5.5.5.5, 9.9.9.9") == 429  # still the client at 9.9.9.9
    assert attempt("8.8.8.8") == 401  # a genuinely different client


def test_llm_routes_are_limited_per_user(client, make_client, make_pdf, tune):
    llm = FakeLLM()
    app.dependency_overrides[get_llm_provider] = lambda: llm
    bob = make_client("bob@example.com")
    tune(llm_requests_per_minute=2, min_top_score=0.0)
    document = upload(client, make_pdf, 1, ["Revenue rose 12 percent in 2024."]).json()
    assert client.post(f"/documents/{document['id']}/embeddings").status_code == 200

    def ask(who):
        return who.post("/ask", json={"question": "How much did revenue rise?"})

    assert ask(client).status_code == 200
    assert ask(client).status_code == 200
    third = ask(client)

    assert third.status_code == 429
    assert int(third.headers["Retry-After"]) > 0
    assert llm.calls == 2  # the blocked request never reached the model
    assert ask(bob).status_code == 200  # one user's limit is not another's


def test_uploads_are_limited_per_user(client, make_client, make_pdf, tune):
    bob = make_client("bob@example.com")
    tune(upload_requests_per_minute=2)

    assert upload(client, make_pdf, 1).status_code == 201
    assert upload(client, make_pdf, 2).status_code == 201
    assert upload(client, make_pdf, 3).status_code == 429
    assert upload(bob, make_pdf, 4).status_code == 201
