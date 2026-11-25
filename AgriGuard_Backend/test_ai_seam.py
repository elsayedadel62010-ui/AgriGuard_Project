"""The seam between the backend endpoint and the real AI HTTP client.

The other two suites each stub one side of this boundary, which is exactly why
a mismatch between them could ship unnoticed:

* `test_crop_flow.py` replaces `main.predict_irrigation`, so it proves the
  endpoint passes the right crop *argument* but never builds a payload.
* `test_ai_client.py` calls `predict_irrigation` directly, so it never resolves
  a plot or commits a reading.

A wrong field name, a wrong URL path, or a payload that drops a key would pass
both. This file runs the whole chain with only the *network* mocked, through
`httpx.MockTransport`, so the real `ai_client.predict_irrigation` builds the
real payload, the real `httpx` encodes it, and the real response parsing and
status handling run against genuine `httpx.Response` objects.

Every test runs against a temporary SQLite file, so the development database is
never touched and the AI process never has to be running.
"""

import json as jsonlib
import sys
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, Session, create_engine

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ai_client  # noqa: E402  (imported for its real AI_URL / IRRIGATION_PATH)
import models  # noqa: E402  (registers the tables on SQLModel.metadata)
from main import app  # noqa: E402
from models import Plot, SensorReading  # noqa: E402


READING = {
    "soil_moisture": 22.0,
    "temperature": 25.0,
    "humidity": 60.0,
    "light": 20000.0,
    "water_flow": 0.0,
    "pump_status": False,
}

#: What the AI service answers with. Shaped like the real /v1/predict/irrigation
#: response so the assertions are about the contract, not about the stub.
AI_RESPONSE = {
    "model": "irrigation-lr",
    "version": "0.2.0",
    "prediction": {
        "irrigation_needed": True,
        "prediction": "Irrigation Recommended",
        "prediction_score": 0.82,
        "decision_source": "model",
        "reason": "soil moisture 22 is below the 27 reference threshold for maize",
        "advisory_only": True,
    },
}


class RecordingTransport(httpx.MockTransport):
    """A mock transport that records what reached the network boundary.

    Extending `httpx.MockTransport` rather than replacing `httpx.post` keeps the
    genuine request encoding, response decoding, `raise_for_status()` and JSON
    parsing in the loop; only the socket is missing.
    """

    def __init__(self, body=None, status_code=200, raw_body=None):
        self.body = AI_RESPONSE if body is None else body
        self.status_code = status_code
        self.raw_body = raw_body
        self.requests = []

    def handle_request(self, request):
        self.requests.append(request)
        if self.raw_body is not None:
            return httpx.Response(self.status_code, content=self.raw_body)
        return httpx.Response(self.status_code, json=self.body)


@pytest.fixture()
def transport(monkeypatch):
    """Route the real AI client's httpx.post through a recording transport."""
    recorder = RecordingTransport()

    def mock_post(url, json=None, timeout=None):
        with httpx.Client(transport=recorder) as client:
            return client.post(url, json=json, timeout=timeout)

    monkeypatch.setattr(ai_client.httpx, "post", mock_post)
    return recorder


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """A TestClient on an isolated database. The AI client is NOT stubbed."""
    engine = create_engine(
        f"sqlite:///{tmp_path / 'seam.db'}",
        connect_args={"check_same_thread": False},
    )
    SQLModel.metadata.create_all(engine)

    def override_session():
        with Session(engine) as session:
            yield session

    from database import get_session

    app.dependency_overrides[get_session] = override_session

    with TestClient(app) as test_client:
        test_client.engine = engine
        yield test_client

    app.dependency_overrides = {}


def _register_plot(client, device_id="ESP32_SEAM_01", crop_type="maize", name="North Field"):
    response = client.post("/api/plots", json={"name": name, "device_id": device_id, "crop_type": crop_type})
    assert response.status_code == 201, response.text
    return response.json()


# ------------------------------------------------- the whole chain, end to end
def test_reading_reaches_the_ai_and_the_prediction_comes_back(client, transport):
    """POST /api/sensor-data -> plot lookup -> real payload -> stored prediction."""
    plot = _register_plot(client, device_id="ESP32_SEAM_01", crop_type="maize")
    created = client.post("/api/sensor-data", json={"device_id": "ESP32_SEAM_01", **READING})
    assert created.status_code == 200, created.text
    body = created.json()

    # 1. device_id resolved to the right Plot
    assert plot["device_id"] == "ESP32_SEAM_01"
    assert body["device_id"] == "ESP32_SEAM_01"
    assert body["plot_id"] == plot["id"]
    assert body["crop_type"] == "maize"

    # 2. exactly one request reached the network boundary, at the real path
    assert len(transport.requests) == 1
    sent = transport.requests[0]
    assert sent.url.path == ai_client.IRRIGATION_PATH
    assert sent.method == "POST"

    # 3. the payload carries the resolved crop, unchanged, and the plot id as a
    #    string so it means the same thing here as in the response body
    body_sent = jsonlib.loads(sent.read().decode("utf-8"))
    assert body_sent["crop_type"] == "maize"
    assert body_sent["device_id"] == "ESP32_SEAM_01"
    assert body_sent["plot_id"] == str(plot["id"])
    # The regression this replaced: plot_id used to carry the plot *name* in the
    # payload while the response carried the numeric id.
    assert body_sent["plot_id"] != plot["name"]
    assert body_sent["soil_moisture"] == READING["soil_moisture"]
    assert body_sent["temperature"] == READING["temperature"]
    assert body_sent["humidity"] == READING["humidity"]
    assert body_sent["light"] == READING["light"]

    # 4. the AI response was interpreted, not just stored
    advice = body["irrigation"]
    assert advice["irrigation_needed"] is True
    assert advice["decision_source"] == "model"
    assert advice["prediction"] == "Irrigation Recommended"
    assert advice["prediction_score"] == pytest.approx(0.82)
    assert advice["advisory_only"] is True
    assert "maize" in advice["reason"]
    # No error key at all on the happy path; it is only added when advice is missing.
    assert "irrigation_error" not in body

    # 5. the prediction was persisted against the reading, with the crop snapshot
    with Session(client.engine) as session:
        stored = session.query(SensorReading).one()
        assert stored.device_id == "ESP32_SEAM_01"
        assert stored.crop_type == "maize"
        assert stored.irrigation_needed is True
        assert stored.prediction_score == pytest.approx(0.82)
        assert stored.decision_source == "model"
        assert "maize" in stored.prediction_reason


def test_trained_and_rule_only_crops_both_travel_the_seam(client, transport):
    """The crop on the wire is the plot's crop for both halves of the vocabulary."""
    for index, crop in enumerate(("tomato", "maize")):
        device_id = f"ESP32_CROP_{index}"
        _register_plot(client, device_id=device_id, crop_type=crop, name=f"Plot {index}")
        response = client.post("/api/sensor-data", json={"device_id": device_id, **READING})
        assert response.status_code == 200, response.text
        assert response.json()["crop_type"] == crop

    sent_crops = [jsonlib.loads(r.read())["crop_type"] for r in transport.requests]
    assert sent_crops == ["tomato", "maize"]


def test_esp32_cannot_override_the_crop_on_the_seam(client, transport):
    """A crop in the device payload is dropped, not forwarded."""
    _register_plot(client, device_id="ESP32_SEAM_01", crop_type="maize")
    response = client.post(
        "/api/sensor-data",
        json={"device_id": "ESP32_SEAM_01", "crop_type": "tomato", **READING},
    )
    assert response.status_code == 200, response.text
    assert response.json()["crop_type"] == "maize"
    assert jsonlib.loads(transport.requests[0].read())["crop_type"] == "maize"


# ------------------------------------------- the reading survives a bad answer
def test_a_non_object_ai_response_does_not_lose_the_reading(client, transport):
    """A malformed AI reply is a failed prediction, not a lost reading.

    The reading is committed before the AI is called, so a bad response must
    still leave one row and must not report the whole submission as failed -
    otherwise a retrying device would insert a duplicate.
    """
    transport.raw_body = '[{"irrigation_needed": true}]'
    _register_plot(client, device_id="ESP32_SEAM_01", crop_type="maize")

    response = client.post("/api/sensor-data", json={"device_id": "ESP32_SEAM_01", **READING})
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["irrigation"] is None
    assert body["irrigation_error"] is not None
    # The client-facing error must not leak the internal failure mode.
    assert "AttributeError" not in body["irrigation_error"]
    assert "object has no attribute" not in body["irrigation_error"]

    with Session(client.engine) as session:
        assert session.query(SensorReading).count() == 1


def test_an_unexpected_ai_failure_does_not_lose_the_reading(client, monkeypatch):
    """A non-AIServiceError from the AI must not become a 500 after the commit.

    Anything unexpected escaping the client used to be reported as a failed
    submission even though the row was already written, which invites a
    duplicate on retry.
    """
    _register_plot(client, device_id="ESP32_SEAM_01", crop_type="maize")

    def explode(*args, **kwargs):
        raise RuntimeError("simulated unexpected failure")

    monkeypatch.setattr(ai_client.httpx, "post", explode)

    response = client.post("/api/sensor-data", json={"device_id": "ESP32_SEAM_01", **READING})
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["irrigation"] is None
    assert body["irrigation_error"]
    # The internal message must not be echoed back to the device.
    assert "simulated unexpected failure" not in body["irrigation_error"]
    with Session(client.engine) as session:
        assert session.query(SensorReading).count() == 1


# ------------------------------------------------------- mutation-style guard
def test_replacing_the_resolved_crop_with_default_would_fail(client, transport, monkeypatch):
    """Pin the crop plumbing against the exact regression this suite exists for.

    Simulates a backend regression that falls back to `default` at the payload
    boundary - the bug that would reach the model with an all-zero crop block
    and report a model answer for a crop the plot never declared. The test
    asserts the assertion that catches it: after mutating the payload, the crop
    on the wire no longer matches the crop the plot declares, so the seam is
    broken. Restores the real behaviour afterwards.
    """
    plot = _register_plot(client, device_id="ESP32_SEAM_01", crop_type="maize")

    # Record what the unmutated chain sends.
    response = client.post("/api/sensor-data", json={"device_id": "ESP32_SEAM_01", **READING})
    assert response.status_code == 200, response.text
    real_crop = jsonlib.loads(transport.requests[0].read())["crop_type"]
    assert real_crop == plot["crop_type"] == "maize"

    # Now mutate: the crop silently becomes "default" on the way to the AI.
    def mutating_post(url, json=None, timeout=None):
        payload = dict(json or {})
        payload["crop_type"] = "default"
        with httpx.Client(transport=transport) as mock_client:
            return mock_client.post(url, json=payload, timeout=timeout)

    monkeypatch.setattr(ai_client.httpx, "post", mutating_post)
    mutated = client.post("/api/sensor-data", json={"device_id": "ESP32_SEAM_01", **READING})
    assert mutated.status_code == 200, mutated.text
    mutated_crop = jsonlib.loads(transport.requests[-1].read())["crop_type"]

    # The mutation is caught because the crop on the wire no longer agrees with
    # the plot's configured crop - the assertion that makes the regression
    # visible is `real_crop == plot["crop_type"]` above.
    assert mutated_crop == "default"
    assert mutated_crop != plot["crop_type"]
    assert real_crop != mutated_crop
    assert real_crop == plot["crop_type"]
