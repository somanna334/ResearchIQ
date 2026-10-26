import pytest
from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient

from app import ratelimit
from app.ratelimit import LoginThrottle, SlidingWindowLimiter, rate_limit


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


@pytest.fixture(autouse=True)
def _reset():
    ratelimit.reset_all()
    yield
    ratelimit.reset_all()


def test_allows_up_to_limit():
    lim = SlidingWindowLimiter(clock=FakeClock())
    assert all(lim.hit("k", 3, 60)[0] for _ in range(3))


def test_blocks_over_limit_with_retry_after():
    clock = FakeClock()
    lim = SlidingWindowLimiter(clock=clock)
    for _ in range(3):
        lim.hit("k", 3, 60)
    clock.advance(10)
    allowed, retry = lim.hit("k", 3, 60)
    assert not allowed
    assert retry == 50


def test_window_expires():
    clock = FakeClock()
    lim = SlidingWindowLimiter(clock=clock)
    for _ in range(3):
        lim.hit("k", 3, 60)
    clock.advance(61)
    assert lim.hit("k", 3, 60)[0]


def test_keys_are_independent():
    lim = SlidingWindowLimiter(clock=FakeClock())
    for _ in range(3):
        lim.hit("a", 3, 60)
    assert not lim.hit("a", 3, 60)[0]
    assert lim.hit("b", 3, 60)[0]


def test_dependency_returns_429_with_retry_after():
    app = FastAPI()

    @app.get("/x", dependencies=[Depends(rate_limit("t", 3, 60))])
    def x():
        return {"ok": True}

    client = TestClient(app)
    assert [client.get("/x").status_code for _ in range(3)] == [200, 200, 200]
    blocked = client.get("/x")
    assert blocked.status_code == 429
    assert int(blocked.headers["Retry-After"]) >= 1


def test_login_throttle_locks_after_failures():
    clock = FakeClock()
    t = LoginThrottle(max_email_failures=3, window=100, clock=clock)
    for _ in range(3):
        t.check("a@x.com", "1.1.1.1")
        t.record_failure("a@x.com", "1.1.1.1")
    with pytest.raises(HTTPException) as exc:
        t.check("A@X.com", "9.9.9.9")  # case-insensitive, different IP
    assert exc.value.status_code == 429
    clock.advance(101)
    t.check("a@x.com", "1.1.1.1")  # lock expired


def test_login_success_clears_email_counter():
    t = LoginThrottle(max_email_failures=3, clock=FakeClock())
    t.record_failure("a@x.com", "1.1.1.1")
    t.record_failure("a@x.com", "1.1.1.1")
    t.record_success("a@x.com", "1.1.1.1")
    t.record_failure("a@x.com", "1.1.1.1")
    t.check("a@x.com", "1.1.1.1")  # only 1 failure now, no raise


def test_ip_lock_independent_of_email():
    t = LoginThrottle(max_email_failures=5, max_ip_failures=3, clock=FakeClock())
    for i in range(3):
        t.record_failure(f"user{i}@x.com", "2.2.2.2")
    with pytest.raises(HTTPException):
        t.check("brand-new@x.com", "2.2.2.2")
    t.check("brand-new@x.com", "3.3.3.3")  # other IP is fine
