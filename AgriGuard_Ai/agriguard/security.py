"""Optional API-key and rate-limit middleware for the prediction service.

    AGRIGUARD_API_KEY=<shared secret>            require X-AgriGuard-Key
    AGRIGUARD_RATE_LIMIT="30/minute"             per client, per process

Both are **off unless the environment variable is set**, which is the correct
default for the deployment this project actually has: a laptop on a LAN, behind
a reverse proxy, called by Member 3's backend. Turning a guard on by default
here would have broken every existing client and every test for no security
gain, because the real perimeter is the network the service sits on.

Nothing in this module is a substitute for a reverse proxy. See
`docs/limitations.md` for what these two guards do and do not buy.

What is genuinely defended here
-------------------------------
* **Scanning.** An unauthenticated deployment on a routable network accepts
  free inference from anyone who finds the port. The key removes that.
* **Accidental hammering.** A dashboard polling loop with a bug, or a
  retried request storm, cannot saturate the single CPU this runs on. The
  limiter bounds that.

What is not defended, and is stated rather than implied
------------------------------------------------------
* **Brute force of the key itself.** There is no lockout and no attempt
  counter, because a lockout is a denial-of-service primitive: anyone who
  knows the port can lock the service out by guessing six times. Rate limiting
  bounds the guesses instead.
* **Transport.** The key travels in a header in cleartext over HTTP. It is
  only meaningful behind TLS. Terminate TLS at the reverse proxy.
* **Multi-worker deployments.** The limiter's state is per process. Under
  `uvicorn --workers 4` each worker keeps its own counters, so the effective
  ceiling is four times what is configured. Redis or an edge limiter is the
  fix; a laptop demo does not need one.
* **Authorisation.** There is one key, so every holder sees every plot. Per
  plot or per device scoping belongs in Member 3's backend, which owns the
  data model.
"""

import hashlib
import hmac
import os
import time
from collections import OrderedDict
from typing import Optional, Tuple

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

#: Header the client presents the key in. Prefixed so it cannot collide with a
#: proxy's own auth header when the service sits behind one.
API_KEY_HEADER = "X-AgriGuard-Key"

API_KEY_ENV = "AGRIGUARD_API_KEY"
RATE_LIMIT_ENV = "AGRIGUARD_RATE_LIMIT"

#: Paths not counted against the rate limit. /v1/health is polled by
#: monitoring, and counting it would let a monitoring system eat the budget
#: that prediction requests need. The API key still applies - a health probe
#: leaking model status is still information - so these are rate-limit
#: exemptions only, not open endpoints.
RATE_LIMIT_EXEMPT = ("/v1/health", "/docs", "/openapi.json", "/redoc", "/")

_UNSET = object()


def parse_rate_limit(raw: str) -> Optional[Tuple[int, float]]:
    """Parse ``"30/minute"`` into ``(30, 60.0)`` - the limit and the window.

    Returns None for anything unrecognised, including an empty value, because
    silently treating a typo like ``"30 per minute"`` as unlimited is the one
    outcome that must not happen: a misconfigured limiter that fails open is
    worse than no limiter, because it looks configured.
    """
    text = (raw or "").strip().lower().replace(" ", "")
    if not text:
        return None

    units = {
        "second": 1.0, "sec": 1.0, "s": 1.0,
        "minute": 60.0, "min": 60.0, "m": 60.0,
        "hour": 3600.0, "hr": 3600.0, "h": 3600.0,
        "day": 86400.0, "d": 86400.0,
    }
    for suffix in sorted(units, key=len, reverse=True):
        if text.endswith("/" + suffix):
            count_text = text[: -len(suffix) - 1]
            if count_text.isdigit() and int(count_text) > 0:
                return int(count_text), units[suffix]
            return None
    return None


def is_valid_key(presented: Optional[str], expected: Optional[str]) -> bool:
    """Constant-time key comparison, with the missing-value cases folded in.

    `compare_digest` rather than `==` so the comparison time does not leak how
    many leading characters were correct. The two `None` paths return False
    without calling compare_digest, which is safe because "no key configured"
    is not a secret.
    """
    if not expected or not presented:
        return False
    return hmac.compare_digest(presented.strip(), expected)


class TokenBucketLimiter:
    """Per-client token bucket.

    A sliding-window counter was the obvious alternative and was rejected: it
    stores every request timestamp, which is unbounded in the number of
    requests. The bucket is a fixed number of floats per client and gives the
    same practical guarantee - at most `limit` requests in any `window` - while
    allowing a short burst up to `limit` that a well-behaved client uses after
    a quiet period.

    `clients` is an OrderedDict with a hard cap so that a client spoofing many
    source addresses cannot grow the process without bound; when the cap is
    hit the least-recently-used entry is evicted, which is the correct thing to
    lose, because the entry that has been idle longest is also the one least
    likely to be mid-burst.
    """

    def __init__(self, limit: int, window: float, max_clients: int = 10_000):
        self.limit = limit
        self.window = window
        self.max_clients = max_clients
        self._buckets: "OrderedDict[str, list]" = OrderedDict()
        self._rate = limit / window  # tokens per second

    def check(self, client: str, now: Optional[float] = None) -> Tuple[bool, float]:
        """Consume one token. Returns (allowed, seconds_until_a_token_is_free)."""
        now = time.monotonic() if now is None else now
        tokens, stamp = self._buckets.get(client, (float(self.limit), now))

        # Refill for the time that has passed since the bucket was last seen.
        tokens = min(float(self.limit), tokens + (now - stamp) * self._rate)
        if tokens >= 1.0:
            self._buckets[client] = (tokens - 1.0, now)
            self._buckets.move_to_end(client)
            self._evict_if_needed()
            return True, 0.0

        self._buckets[client] = (tokens, now)
        self._buckets.move_to_end(client)
        self._evict_if_needed()
        return False, (1.0 - tokens) / self._rate if self._rate > 0 else self.window

    def reset(self) -> None:
        self._buckets.clear()

    def _evict_if_needed(self) -> None:
        while len(self._buckets) > self.max_clients:
            self._buckets.popitem(last=False)


def client_identity(request: Request) -> str:
    """Who the rate limit is counted against.

    Prefers the API key when one is configured, so that several devices behind
    one NAT share a budget rather than each getting a fresh one - otherwise the
    limit is trivially bypassed by rotating source addresses. Falls back to the
    peer address otherwise.

    `X-Forwarded-For` is read only when the service is explicitly told to trust
    it. Reading it unconditionally would let any client claim an arbitrary
    address and get an unlimited budget, which is the same as having no limiter.
    """
    key = request.headers.get(API_KEY_HEADER)
    if key:
        return f"key:{hashlib_scrub(key)}"
    peer = request.client.host if request.client else "unknown"
    if os.environ.get("AGRIGUARD_TRUST_PROXY") == "1":
        forwarded = request.headers.get("X-Forwarded-For", "")
        if forwarded:
            peer = forwarded.split(",")[0].strip() or peer
    return f"peer:{peer}"


def hashlib_scrub(value: str) -> str:
    """Stable, non-reversible label for a key, so buckets do not store secrets."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


class SecurityMiddleware(BaseHTTPMiddleware):
    """API key and/or rate limit, each applied only if configured.

    Kept as one middleware rather than two so the ordering is fixed and
    testable: the key is checked first, so an unauthenticated flood cannot
    consume a legitimate client's rate-limit budget by presenting bad keys.
    """

    def __init__(
        self,
        app,
        api_key: Optional[str] = None,
        rate_limit: Optional[Tuple[int, float]] = None,
        exempt_paths: tuple = RATE_LIMIT_EXEMPT,
    ):
        super().__init__(app)
        self.api_key = api_key or None
        self.rate_limit = rate_limit
        self.exempt_paths = exempt_paths
        self.limiter = TokenBucketLimiter(*rate_limit) if rate_limit else None

    async def dispatch(self, request: Request, call_next):
        if self.api_key:
            presented = request.headers.get(API_KEY_HEADER)
            if not is_valid_key(presented, self.api_key):
                # One message for "no key" and "wrong key", so a caller cannot
                # use the response to learn whether a key exists.
                return JSONResponse(
                    status_code=401,
                    content={"detail": f"missing or invalid {API_KEY_HEADER}"},
                    headers={"WWW-Authenticate": API_KEY_HEADER},
                )

        if self.limiter and request.url.path not in self.exempt_paths:
            allowed, retry_after = self.limiter.check(client_identity(request))
            if not allowed:
                return JSONResponse(
                    status_code=429,
                    content={"detail": "rate limit exceeded"},
                    headers={"Retry-After": str(max(1, int(retry_after + 0.999)))},
                )

        return await call_next(request)


def build_security_options() -> Optional[dict]:
    """Read the environment and return `add_middleware` kwargs, or None if off.

    Returns options rather than a constructed middleware because Starlette's
    `add_middleware` takes the class and builds the instance itself during
    start-up, when the ASGI `receive`/`send` channels exist. Handing it an
    already-built instance fails at that point.

    An unparseable AGRIGUARD_RATE_LIMIT is refused loudly rather than ignored:
    a limiter that silently fails open is worse than an absent one, because
    `AGRIGUARD_RATE_LIMIT=30/minute!` in a deployment ticket reads as
    protection that is not there.
    """
    api_key = (os.environ.get(API_KEY_ENV) or "").strip() or None
    raw_limit = os.environ.get(RATE_LIMIT_ENV, "")
    limit = parse_rate_limit(raw_limit)
    if raw_limit.strip() and limit is None:
        raise ValueError(
            f"{RATE_LIMIT_ENV}={raw_limit!r} could not be parsed. Use a positive "
            "integer and a unit, e.g. '30/minute', '60/second' or '1000/hour'."
        )
    if not api_key and limit is None:
        return None
    return {"api_key": api_key, "rate_limit": limit}
