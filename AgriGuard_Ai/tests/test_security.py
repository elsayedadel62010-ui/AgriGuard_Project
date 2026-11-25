"""Tests for the optional API key and rate limiter.

The default configuration is the one that ships - both guards off - so the
first test here is the important one: an unconfigured deployment must not
change the behaviour any other test in this suite depends on. Everything else
builds an isolated app with the guard enabled, because `app` is a module-level
singleton and re-importing it with different environment variables would leak
between tests.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agriguard.security import (
    API_KEY_HEADER,
    RATE_LIMIT_ENV,
    SecurityMiddleware,
    TokenBucketLimiter,
    build_security_options,
    is_valid_key,
    parse_rate_limit,
)


def guarded_app(**options):
    """A minimal app with the security middleware attached and routes to hit."""
    app = FastAPI()

    @app.get("/v1/health")
    def health():
        return {"status": "ok"}

    @app.get("/v1/predict/irrigation")
    def predict():
        return {"ok": True}

    app.add_middleware(SecurityMiddleware, **options)
    return TestClient(app)


# ------------------------------------------------------------------ defaults
def test_no_configuration_means_no_guard(monkeypatch):
    """The shipped default: no env vars, no middleware, every caller served."""
    monkeypatch.delenv("AGRIGUARD_API_KEY", raising=False)
    monkeypatch.delenv(RATE_LIMIT_ENV, raising=False)
    assert build_security_options() is None

    unguarded = guarded_app(api_key=None, rate_limit=None)
    assert unguarded.get("/v1/predict/irrigation").status_code == 200


# --------------------------------------------------------------------- parsing
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("30/minute", (30, 60.0)),
        ("30/min", (30, 60.0)),
        ("60/second", (60, 1.0)),
        ("1000/hour", (1000, 3600.0)),
        ("  30 / Minute  ", (30, 60.0)),
    ],
)
def test_valid_rate_limits_parse(raw, expected):
    assert parse_rate_limit(raw) == expected


@pytest.mark.parametrize("raw", ["", "   ", "0/minute", "-5/minute", "lots/day", "30"])
def test_unparseable_rate_limits_are_rejected_not_guessed(raw):
    """A limiter that fails open is worse than no limiter, because it looks set.

    Returning None here is what stops `AGRIGUARD_RATE_LIMIT=30 per minute`
    from silently meaning "unlimited".
    """
    assert parse_rate_limit(raw) is None


def test_a_mistyped_rate_limit_fails_the_build_instead_of_running_open(monkeypatch):
    monkeypatch.setenv(RATE_LIMIT_ENV, "30 per minute")
    monkeypatch.delenv("AGRIGUARD_API_KEY", raising=False)
    with pytest.raises(ValueError, match="could not be parsed"):
        build_security_options()


# ------------------------------------------------------------------- api key
def test_a_configured_key_rejects_missing_and_wrong_keys_identically():
    client = guarded_app(api_key="s3cret", rate_limit=None)

    missing = client.get("/v1/predict/irrigation")
    wrong = client.get("/v1/predict/irrigation", headers={API_KEY_HEADER: "guess"})
    right = client.get("/v1/predict/irrigation", headers={API_KEY_HEADER: "s3cret"})

    assert missing.status_code == 401
    assert wrong.status_code == 401
    # Identical bodies: a caller must not be able to probe for key existence.
    assert missing.json() == wrong.json()
    assert right.status_code == 200


def test_the_key_applies_to_health_too():
    """Model status is information; an unauthenticated probe should not get it."""
    client = guarded_app(api_key="s3cret", rate_limit=None)
    assert client.get("/v1/health").status_code == 401
    assert client.get("/v1/health", headers={API_KEY_HEADER: "s3cret"}).status_code == 200


def test_a_blank_configured_key_is_treated_as_off(monkeypatch):
    """AGRIGUARD_API_KEY="" must not lock out every client with an empty header."""
    monkeypatch.setenv("AGRIGUARD_API_KEY", "   ")
    monkeypatch.delenv(RATE_LIMIT_ENV, raising=False)
    assert build_security_options() is None


def test_key_comparison_rejects_the_empty_and_missing_cases():
    assert is_valid_key("abc", "abc") is True
    assert is_valid_key("abc", "abd") is False
    assert is_valid_key(None, "abc") is False
    assert is_valid_key("abc", None) is False
    # A blank expected key is "no key configured", which must not authenticate.
    assert is_valid_key("", "") is False


# --------------------------------------------------------------- rate limits
def test_the_budget_runs_out_and_then_recovers():
    client = guarded_app(api_key=None, rate_limit=(3, 60.0))

    assert [client.get("/v1/predict/irrigation").status_code for _ in range(5)] == [
        200, 200, 200, 429, 429,
    ]


def test_a_throttled_response_says_when_to_retry():
    client = guarded_app(api_key=None, rate_limit=(1, 60.0))
    client.get("/v1/predict/irrigation")
    response = client.get("/v1/predict/irrigation")
    assert response.status_code == 429
    # Rounded up and never zero: "Retry-After: 0" tells a client to retry
    # immediately, which is exactly what it should not do.
    assert int(response.headers["Retry-After"]) >= 1


def test_health_is_exempt_from_the_limiter_but_not_from_the_key():
    """Monitoring must not be able to eat the budget prediction requests need."""
    client = guarded_app(api_key="s3cret", rate_limit=(1, 60.0))
    headers = {API_KEY_HEADER: "s3cret"}

    for _ in range(5):
        assert client.get("/v1/health", headers=headers).status_code == 200

    # The first prediction still gets through: the health probes were free.
    assert client.get("/v1/predict/irrigation", headers=headers).status_code == 200
    assert client.get("/v1/predict/irrigation", headers=headers).status_code == 429
    # ...but an unauthenticated health probe is still rejected.
    assert client.get("/v1/health").status_code == 401


def test_authentication_is_checked_before_the_limiter():
    """Bad keys must not spend a legitimate client's budget.

    Otherwise an attacker with the port but not the key could lock the real
    client out by sending junk, which turns the limiter into a denial-of-
    service primitive.
    """
    client = guarded_app(api_key="s3cret", rate_limit=(3, 60.0))
    headers = {API_KEY_HEADER: "s3cret"}

    assert client.get("/v1/predict/irrigation", headers=headers).status_code == 200
    for _ in range(10):
        assert client.get("/v1/predict/irrigation", headers={API_KEY_HEADER: "junk"}).status_code == 401
    # The full budget of three is still there, none of the junk was charged to it.
    assert [client.get("/v1/predict/irrigation", headers=headers).status_code for _ in range(4)] == [
        200, 200, 429, 429,
    ]


def test_clients_are_limited_independently():
    """Two keys are two budgets, not one shared pool."""
    client = guarded_app(api_key=None, rate_limit=(1, 60.0))
    first = {API_KEY_HEADER: "aaa"}
    second = {API_KEY_HEADER: "bbb"}

    assert client.get("/v1/predict/irrigation", headers=first).status_code == 200
    assert client.get("/v1/predict/irrigation", headers=first).status_code == 429
    assert client.get("/v1/predict/irrigation", headers=second).status_code == 200


# ------------------------------------------------------------ bucket internals
def test_the_bucket_refills_over_time():
    """Injecting the clock keeps the refill test instant and deterministic."""
    limiter = TokenBucketLimiter(limit=2, window=60.0)
    assert limiter.check("c", now=0.0)[0] is True
    assert limiter.check("c", now=0.0)[0] is True
    assert limiter.check("c", now=0.0)[0] is False

    # 10 s of a 2-per-60 s bucket earns 0.33 tokens: not enough yet.
    assert limiter.check("c", now=10.0)[0] is False
    # 30 s more earns another token.
    assert limiter.check("c", now=40.0)[0] is True


def test_the_bucket_never_exceeds_its_limit_however_long_it_idles():
    limiter = TokenBucketLimiter(limit=2, window=60.0)
    for _ in range(2):
        limiter.check("c", now=0.0)
    # 10,000 s of idling is far longer than the 60 s window. The bucket refills
    # to its ceiling, so exactly two more requests are served and no more.
    assert limiter.check("c", now=10_000.0)[0] is True
    assert limiter.check("c", now=10_000.0)[0] is True
    assert limiter.check("c", now=10_000.0)[0] is False


def test_the_client_table_is_bounded_so_spoofing_cannot_grow_the_process():
    """Address spoofing must not be a way to allocate memory for free."""
    limiter = TokenBucketLimiter(limit=5, window=60.0, max_clients=8)
    for index in range(500):
        limiter.check(f"client-{index}", now=0.0)
    assert len(limiter._buckets) <= 8
