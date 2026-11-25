"""The AI client must actually put the plot's crop on the wire.

`test_crop_flow.py` stubs `main.predict_irrigation`, so it proves the endpoint
passes the right crop *argument* but not that the HTTP body carries it. A
regression that sent `default` instead of the resolved crop would pass that
suite, so this file stubs `httpx.post` and inspects the real payload.
"""

import json as jsonlib
import sys
from pathlib import Path


import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ai_client  # noqa: E402


READING = {
    "device_id": "ESP32_01",
    "soil_moisture": 22.0,
    "temperature": 25.0,
    "humidity": 60.0,
    "light": 20000.0,
}


def _response(body, status=200):
    request = httpx.Request("POST", "http://test/v1/predict/irrigation")
    return httpx.Response(status, json=body, request=request)


def _raw_response(text, status=200):
    """A 200 whose body is exactly `text`, for JSON that json= cannot express."""
    request = httpx.Request("POST", "http://test/v1/predict/irrigation")
    return httpx.Response(status, content=text, request=request)



BODY = {
    "model": "irrigation-lr",
    "version": "0.2.0",
    "prediction": {
        "irrigation_needed": True,
        "prediction": "Irrigation Recommended",
        "prediction_score": 0.75,
        "decision_source": "model",
        "reason": "because",
        "advisory_only": True,
    },
}


@pytest.fixture()
def sent(monkeypatch):
    """Capture the outgoing request instead of making one."""
    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        return _response(BODY)

    monkeypatch.setattr(ai_client.httpx, "post", fake_post)
    return captured


def test_the_plot_crop_is_sent_to_the_ai(sent):
    ai_client.predict_irrigation(READING, crop_type="tomato")
    assert sent["json"]["crop_type"] == "tomato"
    assert sent["url"].endswith("/v1/predict/irrigation")


@pytest.mark.parametrize("crop", ["tomato", "rice", "pepper", "maize"])
def test_every_crop_reaches_the_ai_unchanged(sent, crop):
    ai_client.predict_irrigation(READING, crop_type=crop)
    assert sent["json"]["crop_type"] == crop


def test_the_crop_is_not_left_to_the_callings_reading(sent):
    """A stale crop_type on the reading must not win over the resolved one."""
    ai_client.predict_irrigation(
        dict(READING, crop_type="default"), crop_type="rice"
    )
    assert sent["json"]["crop_type"] == "rice"


def test_the_response_is_flattened_to_the_fields_the_backend_stores(sent):
    result = ai_client.predict_irrigation(READING, crop_type="tomato")
    assert result["decision_source"] == "model"
    assert result["irrigation_needed"] is True
    assert result["prediction_score"] == 0.75
    assert result["advisory_only"] is True


def test_history_is_only_sent_when_supplied(sent):
    ai_client.predict_irrigation(READING, crop_type="tomato")
    assert "history" not in sent["json"]

    ai_client.predict_irrigation(
        READING, crop_type="tomato", history={"last_irrigation_at": "x"}
    )
    assert sent["json"]["history"] == {"last_irrigation_at": "x"}


def test_an_unreachable_ai_service_raises(monkeypatch):
    def boom(*args, **kwargs):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(ai_client.httpx, "post", boom)

    with pytest.raises(ai_client.AIServiceError) as excinfo:
        ai_client.predict_irrigation(READING, crop_type="tomato")
    assert "unreachable" in str(excinfo.value)


def test_a_response_without_a_prediction_raises(monkeypatch):
    def fake_post(url, json=None, timeout=None):
        return _response({"model": "irrigation-lr", "version": "0.2.0"})

    monkeypatch.setattr(ai_client.httpx, "post", fake_post)

    with pytest.raises(ai_client.AIServiceError):
        ai_client.predict_irrigation(READING, crop_type="tomato")


# ------------------------------------------------ non-object response bodies
# A reverse proxy, a wrong service on AGRIGUARD_AI_URL, or a misrouted request
# can all answer 200 with a JSON array or a bare string. Calling .get() on one
# of those raised AttributeError, which escaped as an unhandled 500 *after* the
# reading had already been committed - so the client was told the whole
# submission failed and a retry inserted a second reading. These pin that such
# a body is reported as the AI contract being broken, and never mistaken for a
# successful prediction.


@pytest.mark.parametrize(
    "body",
    [
        pytest.param([{"irrigation_needed": True}], id="json-array"),
        pytest.param([1, 2, 3], id="json-array-of-numbers"),
        pytest.param("irrigation recommended", id="json-string"),
        pytest.param("42", id="json-numeric-string"),
        pytest.param(42, id="json-number"),
        pytest.param(None, id="json-null"),
    ],
)
def test_a_non_object_response_raises_a_service_error(body, monkeypatch):
    def fake_post(url, json=None, timeout=None):
        # httpx's json=None means "send no body", so a JSON null has to be
        # written out by hand to reach the type check.
        return _raw_response("null" if body is None else jsonlib.dumps(body))

    monkeypatch.setattr(ai_client.httpx, "post", fake_post)

    with pytest.raises(ai_client.AIServiceError) as excinfo:
        ai_client.predict_irrigation(READING, crop_type="tomato")

    message = str(excinfo.value)
    assert "expected a JSON object" in message
    # The internal failure mode must not reach the client.
    assert "AttributeError" not in message
    assert "object has no attribute" not in message



def test_a_non_object_response_never_yields_a_prediction(monkeypatch):
    """The failure must be an exception, not a half-populated result.

    If this ever returned a dict the endpoint would persist it and report
    success, so assert the call raises rather than inspecting a return value.
    """
    for body in ([], "irrigation recommended", 7):
        monkeypatch.setattr(
            ai_client.httpx, "post", lambda url, json=None, timeout=None, b=body: _response(b)
        )
        with pytest.raises(ai_client.AIServiceError):
            ai_client.predict_irrigation(READING, crop_type="tomato")


def test_a_valid_object_response_is_still_accepted(monkeypatch):
    """Guard the fix against over-rejecting: a real response must still work."""
    monkeypatch.setattr(
        ai_client.httpx, "post", lambda url, json=None, timeout=None: _response(BODY)
    )

    result = ai_client.predict_irrigation(READING, crop_type="tomato")

    assert result["irrigation_needed"] is True
    assert result["prediction"] == "Irrigation Recommended"
    assert result["prediction_score"] == pytest.approx(0.75)
    assert result["decision_source"] == "model"
    assert result["advisory_only"] is True

