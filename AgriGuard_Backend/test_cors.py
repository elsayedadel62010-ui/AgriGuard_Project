"""CORS must be an explicit allow-list, not a wildcard.

The backend is reachable on the LAN and can write, so `allow_origins=["*"]`
advertised every prediction and plot-registration endpoint to any page the
user happened to have open. It was also paired with `allow_credentials=True`,
a combination browsers reject outright - so it was a broken policy as well as
a permissive one.
"""

import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent))

from main import CORS_ORIGINS_ENV, configure_cors, parse_cors_origins  # noqa: E402


def _cors_kwargs(app):
    for middleware in app.user_middleware:
        if middleware.cls is CORSMiddleware:
            return middleware.kwargs
    return None


def test_a_wildcard_is_never_configured():
    """The old setting must not come back through the environment.

    `parse_cors_origins` only splits and trims, so a "*" survives parsing; it is
    `configure_cors` - the function the app actually calls - that drops it. This
    asserts the composed path, which is the one an operator can reach.
    """
    # A wildcard-only list attaches no middleware at all, which is the
    # same-origin-safe outcome.
    assert configure_cors(FastAPI(), parse_cors_origins("*")).user_middleware == []

    # A wildcard mixed with real origins is dropped, not honoured.
    mixed = parse_cors_origins(" http://a.test , * , http://b.test ")
    kwargs = _cors_kwargs(configure_cors(FastAPI(), mixed))
    assert kwargs["allow_origins"] == ["http://a.test", "http://b.test"]


def test_credentials_are_never_allowed():
    application = configure_cors(FastAPI(), ["http://localhost:5173"])
    kwargs = _cors_kwargs(application)
    assert kwargs is not None
    assert kwargs["allow_credentials"] is False
    assert kwargs["allow_origins"] == ["http://localhost:5173"]


def test_no_middleware_without_an_allow_list():
    """Opt-in: same-origin deployments need no CORS header at all."""
    assert configure_cors(FastAPI(), []) .user_middleware == []
    assert configure_cors(FastAPI(), None).user_middleware == []
    assert configure_cors(FastAPI(), ["", "   "]).user_middleware == []


def test_origins_are_split_and_trimmed():
    raw = "http://localhost:5173, http://127.0.0.1:5173 ,, "
    assert parse_cors_origins(raw) == [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]


def test_only_the_documented_methods_are_allowed():
    kwargs = _cors_kwargs(configure_cors(FastAPI(), ["http://localhost:5173"]))
    assert set(kwargs["allow_methods"]) <= {"GET", "POST", "OPTIONS"}


def test_a_listed_origin_is_allowed_and_an_unlisted_one_is_not():
    application = configure_cors(
        FastAPI(), ["http://localhost:5173", "http://192.168.1.50:5173"]
    )

    @application.get("/api/plots")
    def _plots():
        return []

    client = TestClient(application)
    allowed = client.get("/api/plots", headers={"Origin": "http://192.168.1.50:5173"})
    assert allowed.headers.get("access-control-allow-origin") == "http://192.168.1.50:5173"

    blocked = client.get("/api/plots", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in blocked.headers


def test_the_environment_variable_is_the_documented_one():
    assert CORS_ORIGINS_ENV == "AGRIGUARD_CORS_ORIGINS"
