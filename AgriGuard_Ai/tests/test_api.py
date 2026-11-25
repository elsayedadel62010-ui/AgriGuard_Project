"""API tests: both prediction endpoints, their contracts and their error codes.

The crop-health endpoint is exercised with real images from the unseen test
split where the model is available, and with synthetic solid-colour images only
for the validation/transport cases.
"""

import io

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from agriguard.api import app, configure_cors, parse_cors_origins
from agriguard.config import IRRIGATION_CSV
from agriguard.cropdata import CLASS_NAMES
from agriguard.preprocess import load_irrigation_csv

client = TestClient(app, raise_server_exceptions=False)

IRRIGATION_PAYLOAD = {
    "timestamp": "2026-09-27T18:30:00Z",
    "device_id": "ESP32_01",
    "plot_id": "PLOT_A",
    "soil_moisture": 35,
    "temperature": 29,
    "humidity": 58,
    "light": 700,
    "npk_n": None,
    "npk_p": None,
    "npk_k": None,
    "water_flow": 0,
    "pump_status": 0,
    "crop_type": "tomato",
}


def solid_png_bytes(colour=(40, 180, 60)):
    buffer = io.BytesIO()
    Image.new("RGB", (80, 80), colour).save(buffer, format="PNG")
    return buffer.getvalue()


# ------------------------------------------------------------------ service
def test_root_lists_the_endpoints():
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["service"] == "AgriGuard AI"
    assert body["advisory_only"] is True
    assert "POST /v1/predict/irrigation" in body["endpoints"]


def test_health_reports_model_status():
    response = client.get("/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in {"ok", "degraded"}
    assert "irrigation" in body["models"]
    assert "crop_health" in body["models"]
    assert "load_error" in body["models"]["irrigation"]


def test_crop_health_classes_endpoint():
    response = client.get("/v1/classes/crop-health")
    assert response.status_code == 200
    labels = [c["label"] for c in response.json()["classes"]]
    assert labels == ["Healthy", "Possible Disease"]


# ------------------------------------------------------------------ irrigation
def test_irrigation_accepts_the_agreed_payload():
    response = client.post("/v1/predict/irrigation", json=IRRIGATION_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "irrigation"
    assert body["version"] == "0.2.0"
    prediction = body["prediction"]
    assert isinstance(prediction["irrigation_needed"], bool)
    assert prediction["prediction"] in {"Irrigation Recommended", "Irrigation Not Needed"}
    assert 0.0 <= prediction["prediction_score"] <= 1.0
    assert prediction["irrigation_duration_min"] is None
    assert prediction["advisory_only"] is True
    assert prediction["reason"]
    # Every response says what produced the advice, so a client never has to
    # infer it from the score.
    assert prediction["decision_source"] in {
        "model",
        "pump_guard",
        "water_guard",
        "rule_fallback",
        "rule_fallback_out_of_range",
    }


def test_irrigation_response_has_exactly_the_contract_keys():
    response = client.post("/v1/predict/irrigation", json=IRRIGATION_PAYLOAD)
    assert set(response.json()) == {"model", "version", "prediction"}
    assert set(response.json()["prediction"]) == {
        "irrigation_needed",
        "prediction",
        "prediction_score",
        "decision_source",
        "irrigation_duration_min",
        "reason",
        "history_used",
        "advisory_only",
    }


def test_irrigation_works_with_only_the_four_required_fields():
    response = client.post(
        "/v1/predict/irrigation",
        json={
            "soil_moisture": 35,
            "temperature": 29,
            "humidity": 58,
            "light": 700,
        },
    )
    assert response.status_code == 200


def test_dry_soil_returns_a_recommendation():
    payload = dict(IRRIGATION_PAYLOAD, soil_moisture=11.0)
    body = client.post("/v1/predict/irrigation", json=payload).json()
    assert body["prediction"]["irrigation_needed"] is True


def test_wet_soil_returns_no_recommendation():
    payload = dict(IRRIGATION_PAYLOAD, soil_moisture=85.0)
    body = client.post("/v1/predict/irrigation", json=payload).json()
    assert body["prediction"]["irrigation_needed"] is False


def test_pump_running_returns_no_recommendation():
    payload = dict(IRRIGATION_PAYLOAD, soil_moisture=11.0, pump_status=1, water_flow=6.0)
    body = client.post("/v1/predict/irrigation", json=payload).json()
    assert body["prediction"]["irrigation_needed"] is False


def test_history_block_is_accepted():
    payload = dict(IRRIGATION_PAYLOAD)
    payload["history"] = {
        "recent_soil_moisture": [
            {"timestamp": "2026-09-27T18:10:00Z", "soil_moisture": 36.4},
            {"timestamp": "2026-09-27T18:20:00Z", "soil_moisture": 35.6},
        ],
        "last_irrigation_at": "2026-09-27T15:00:00Z",
    }
    response = client.post("/v1/predict/irrigation", json=payload)
    assert response.status_code == 200
    assert response.json()["prediction"]["irrigation_needed"] in {True, False}


def test_history_changes_the_score_it_is_not_ignored():
    base = dict(IRRIGATION_PAYLOAD, soil_moisture=24.0)
    without = client.post("/v1/predict/irrigation", json=base).json()["prediction"]
    with_history = dict(
        base,
        history={
            "recent_soil_moisture": [
                {"timestamp": "2026-09-27T18:10:00Z", "soil_moisture": 30.0},
                {"timestamp": "2026-09-27T18:20:00Z", "soil_moisture": 27.0},
            ],
            "last_irrigation_at": "2026-09-27T09:00:00Z",
        },
    )
    got = client.post("/v1/predict/irrigation", json=with_history).json()["prediction"]
    assert without["prediction_score"] != got["prediction_score"]


@pytest.mark.parametrize(
    "bad",
    [
        {"soil_moisture": -5, "temperature": 29, "humidity": 58, "light": 700},
        {"soil_moisture": 140, "temperature": 29, "humidity": 58, "light": 700},
        {"soil_moisture": 35, "temperature": 29, "humidity": 150, "light": 700},
        {"soil_moisture": 35, "temperature": 29, "humidity": 58, "light": -1},
        {"soil_moisture": 35, "temperature": "very hot", "humidity": 58, "light": 700},
        {"temperature": 29, "humidity": 58, "light": 700},
    ],
)
def test_invalid_irrigation_input_is_rejected_with_422(bad):
    response = client.post("/v1/predict/irrigation", json=bad)
    assert response.status_code == 422
    assert "detail" in response.json()


def test_validation_error_lists_the_offending_field():
    response = client.post(
        "/v1/predict/irrigation",
        json={"soil_moisture": 35, "temperature": 29, "humidity": 58, "light": 700,
              "pump_status": 9},
    )
    assert response.status_code == 422
    fields = [error["field"] for error in response.json()["errors"]]
    assert "pump_status" in fields


def test_missing_body_is_rejected():
    response = client.post("/v1/predict/irrigation", content=b"", headers={"Content-Type": "application/json"})
    assert response.status_code == 422


# ------------------------------------------------------------------ crop health
def test_crop_health_accepts_a_real_healthy_test_image(crop_health_trained, real_healthy_images):
    response = client.post(
        "/v1/predict/crop-health",
        files={"image": (real_healthy_images[0].name, real_healthy_images[0].read_bytes(), "image/jpeg")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "crop-health"
    assert body["version"] == "0.2.0"
    prediction = body["prediction"]
    assert prediction["class"] in CLASS_NAMES
    assert prediction["label"] in {"Healthy", "Possible Disease"}
    assert prediction["method"] == "MobileNetV3-Small"
    assert prediction["model_version"] == "0.2.0"
    assert 0.0 <= prediction["confidence"] <= 1.0


def test_crop_health_accepts_a_real_disease_test_image(crop_health_trained, real_disease_images):
    response = client.post(
        "/v1/predict/crop-health",
        files={"image": (real_disease_images[0].name, real_disease_images[0].read_bytes(), "image/jpeg")},
    )
    assert response.status_code == 200
    assert response.json()["prediction"]["label"] in {"Healthy", "Possible Disease"}


def test_crop_health_returns_503_when_the_model_is_missing(monkeypatch):
    from agriguard import api

    monkeypatch.setattr(api.crop_health_model, "_loaded", False)
    monkeypatch.setattr(
        api.crop_health_model, "load_error", "weights not found", raising=False
    )
    response = client.post(
        "/v1/predict/crop-health",
        files={"image": ("x.png", solid_png_bytes(), "image/png")},
    )
    assert response.status_code == 503
    assert "unavailable" in response.json()["detail"]


def test_crop_health_rejects_a_non_image_content_type():
    response = client.post(
        "/v1/predict/crop-health",
        files={"image": ("notes.txt", b"just text", "text/plain")},
    )
    assert response.status_code == 415


def test_crop_health_rejects_an_empty_upload():
    response = client.post(
        "/v1/predict/crop-health", files={"image": ("empty.png", b"", "image/png")}
    )
    assert response.status_code == 400


def test_crop_health_rejects_a_corrupted_image():
    response = client.post(
        "/v1/predict/crop-health",
        files={"image": ("broken.png", b"\x89PNG\r\n\x1a\n" + b"\x00" * 200, "image/png")},
    )
    assert response.status_code == 422
    assert "image" in response.json()["detail"].lower()


def test_crop_health_requires_the_image_field():
    response = client.post("/v1/predict/crop-health", data={"not_image": "x"})
    assert response.status_code == 422


def test_errors_never_leak_a_python_traceback():
    response = client.post(
        "/v1/predict/crop-health",
        files={"image": ("broken.png", b"not an image at all", "image/png")},
    )
    body = response.text
    assert "Traceback" not in body
    assert "File \"" not in body
    assert ".py" not in body


# ------------------------------------------------- docs cannot drift from reality
def test_the_documented_example_payload_matches_the_real_response():
    """The README and docs/api_contract.md show one canonical request.

    Documentation that quotes a score nobody can reproduce is worse than no
    number, so the documented example is asserted against the live model. If a
    retrain or a default change moves the score, this test fails and the docs get
    updated on purpose rather than by accident.
    """
    payload = {
        "timestamp": "2026-09-27T18:30:00Z",
        "device_id": "ESP32_01",
        "plot_id": "PLOT_A",
        "soil_moisture": 20,
        "temperature": 33,
        "humidity": 42,
        "light": 68000,
        "water_flow": 0,
        "pump_status": 0,
        "crop_type": "tomato",
        "history": {
            "recent_soil_moisture": [
                {"timestamp": "2026-09-27T18:10:00Z", "soil_moisture": 23.4},
                {"timestamp": "2026-09-27T18:20:00Z", "soil_moisture": 21.9},
            ],
            "last_irrigation_at": "2026-09-27T15:00:00Z",
        },
    }
    response = client.post("/v1/predict/irrigation", json=payload)
    assert response.status_code == 200
    prediction = response.json()["prediction"]

    # values quoted in README.md and docs/api_contract.md
    assert prediction["irrigation_needed"] is True
    assert prediction["prediction_score"] == 0.624
    assert prediction["history_used"] is True
    assert prediction["decision_source"] == "model"
    assert ">= 0.35" in prediction["reason"]


def test_documented_request_readings_stay_inside_the_trained_range():
    """The documented example must not use an out-of-distribution reading.

    An earlier version of the docs used soil_moisture 14.5, which is below the
    training minimum, so the example demonstrated extrapolation rather than the
    model's actual behaviour.
    """
    frame, _ = load_irrigation_csv(IRRIGATION_CSV)
    low = float(frame["soil_moisture"].min())
    high = float(frame["soil_moisture"].max())

    payload = dict(IRRIGATION_PAYLOAD)
    payload["soil_moisture"] = 20  # the value used throughout the docs
    assert low <= payload["soil_moisture"] <= high

# ------------------------------------------------------------------- CORS
def test_cors_origins_are_opt_in_and_parsed_strictly():
    """An unset allow-list must add no CORS headers at all.

    Same-origin deployment (reverse proxy) is the default, and adding a
    permissive header there would only widen what a browser sends.
    """
    assert parse_cors_origins("") == []
    assert parse_cors_origins("   ") == []

    # Stray whitespace from copy/paste is tolerated; a blank entry is dropped
    # rather than becoming an empty origin, which would allow null-origin
    # requests.
    assert parse_cors_origins(
        " http://localhost:5173 , ,https://dash.agriguard.local "
    ) == ["http://localhost:5173", "https://dash.agriguard.local"]


def test_configured_cors_answers_preflight_from_an_allowed_origin():
    cors_app = configure_cors(FastAPI(), ["http://localhost:5173"])

    @cors_app.get("/v1/health")
    def _health():
        return {"status": "ok"}

    preflight = TestClient(cors_app).options(
        "/v1/health",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert preflight.status_code == 200
    assert preflight.headers["access-control-allow-origin"] == "http://localhost:5173"

    actual = TestClient(cors_app).get("/v1/health", headers={"Origin": "http://localhost:5173"})
    assert actual.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_configured_cors_refuses_an_origin_outside_the_allow_list():
    """The point of the allow-list: another site gets no grant."""
    cors_app = configure_cors(FastAPI(), ["http://localhost:5173"])

    @cors_app.get("/v1/health")
    def _health():
        return {"status": "ok"}

    response = TestClient(cors_app).get(
        "/v1/health", headers={"Origin": "https://evil.example"}
    )
    assert response.status_code == 200  # the server still answers
    assert "access-control-allow-origin" not in response.headers


def test_the_shipped_app_ships_without_cors_by_default():
    """Guards against the middleware being attached unconditionally."""
    response = client.get("/v1/health", headers={"Origin": "http://localhost:5173"})
    assert "access-control-allow-origin" not in response.headers
