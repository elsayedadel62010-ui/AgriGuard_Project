"""Crop-aware plot configuration and prediction flow.

Every test runs against a temporary SQLite file and a stubbed AI service, so
the suite never touches the development database and never needs the AI
process to be running. The stub records the payload it was called with, which
is how "the backend sends tomato to the AI" is asserted rather than assumed.
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, Session, create_engine

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ai_client  # noqa: E402
import crops  # noqa: E402
import models  # noqa: E402  (registers the tables on SQLModel.metadata)
from main import app  # noqa: E402
from models import Plot  # noqa: E402


READING = {
    "soil_moisture": 22.0,
    "temperature": 25.0,
    "humidity": 60.0,
    "light": 20000.0,
    "water_flow": 0.0,
    "pump_status": False,
}


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """A TestClient on an isolated database, with the AI call captured."""
    engine = create_engine(
        f"sqlite:///{tmp_path / 'test.db'}",
        connect_args={"check_same_thread": False},
    )
    SQLModel.metadata.create_all(engine)

    def override_session():
        with Session(engine) as session:
            yield session

    from database import get_session
    app.dependency_overrides[get_session] = override_session

    calls = []

    def fake_predict(reading, crop_type, history=None):
        calls.append({"reading": reading, "crop_type": crop_type})
        source = (
            "model" if crop_type in crops.TRAINED_CROPS else "rule_fallback"
        )
        return {
            "irrigation_needed": True,
            "prediction": "Irrigation Recommended",
            "prediction_score": 0.75,
            "decision_source": source,
            "reason": f"stubbed answer for {crop_type}",
            "advisory_only": True,
            "model": "irrigation-lr",
            "version": "0.2.0",
        }

    monkeypatch.setattr("main.predict_irrigation", fake_predict)

    with TestClient(app) as test_client:
        test_client.calls = calls
        yield test_client

    app.dependency_overrides = {}


def register(client, crop_type="tomato", device_id="ESP32_01", name="Plot A"):
    return client.post(
        "/api/plots",
        json={"name": name, "crop_type": crop_type, "device_id": device_id},
    )


# ---------------------------------------------- 1. create a tomato plot
def test_creating_a_tomato_plot(client):
    response = register(client, crop_type="tomato", device_id="ESP32_01")
    assert response.status_code == 201
    assert response.json()["crop_type"] == "tomato"


def test_plot_creation_returns_the_stored_crop(client):
    response = register(client, crop_type="tomato", device_id="ESP32_01")
    assert response.status_code == 201
    body = response.json()
    assert body["crop_type"] == "tomato"
    assert body["device_id"] == "ESP32_01"
    assert body["name"] == "Plot A"
    assert body["id"] > 0


# ---------------------------------------------- 2. retrieve the plot
def test_retrieving_the_plot(client):
    created = register(client).json()
    listing = client.get("/api/plots")
    assert listing.status_code == 200
    plots = listing.json()
    assert len(plots) == 1
    assert plots[0] == created


# ---------------------------------------------- 3. associate ESP32_01
def test_esp32_01_is_associated_with_the_plot(client):
    plot = register(client, device_id="ESP32_01").json()
    assert plot["device_id"] == "ESP32_01"

    found = client.get("/api/plots").json()
    assert [p["device_id"] for p in found] == ["ESP32_01"]


def test_a_device_cannot_be_registered_to_two_plots(client):
    assert register(client, device_id="ESP32_01").status_code == 201
    second = register(client, device_id="ESP32_01", name="Plot B")
    assert second.status_code == 409
    assert "already registered" in second.json()["detail"]


# ------------------------------------- 4. sensor data resolves to tomato
def test_sensor_data_from_esp32_01_resolves_to_tomato(client):
    register(client, crop_type="tomato", device_id="ESP32_01")
    response = client.post(
        "/api/sensor-data", json=dict(READING, device_id="ESP32_01")
    )
    assert response.status_code == 200
    body = response.json()
    assert body["crop_type"] == "tomato"
    assert body["device_id"] == "ESP32_01"


# ------------------------------------- 5. backend sends tomato to the AI
def test_backend_sends_the_plot_crop_to_the_ai(client):
    register(client, crop_type="tomato", device_id="ESP32_01")
    client.post(
        "/api/sensor-data", json=dict(READING, device_id="ESP32_01")
    )
    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["crop_type"] == "tomato"
    assert call["reading"]["crop_type"] == "tomato"
    # The firmware never names the crop; the backend injects it.
    assert call["reading"]["device_id"] == "ESP32_01"


# ------------------------- 6. tomato/rice/pepper produce crop-aware calls
@pytest.mark.parametrize("crop", ["tomato", "rice", "pepper"])
def test_model_crops_produce_crop_aware_ai_requests(client, crop):
    register(client, crop_type=crop, device_id=f"ESP_{crop}")
    response = client.post(
        "/api/sensor-data", json=dict(READING, device_id=f"ESP_{crop}")
    )
    assert response.status_code == 200
    call = client.calls[-1]
    assert call["crop_type"] == crop
    # Only these three are scored by the model.
    assert response.json()["irrigation"]["decision_source"] == "model"


# ------------------------------- 7. missing crop configuration rejected
def test_missing_crop_configuration_is_rejected_clearly(client):
    response = client.post(
        "/api/sensor-data", json=dict(READING, device_id="ESP32_99")
    )
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "no plot configured" in detail
    assert "ESP32_99" in detail
    assert client.calls == [], "no prediction may be attempted without a crop"


def test_missing_device_id_is_rejected_clearly(client):
    response = client.post("/api/sensor-data", json=dict(READING))
    assert response.status_code == 422
    # device_id is required in the request schema, so an omitted field is now
    # rejected by Pydantic rather than by the endpoint. The rejection must
    # still name the offending field.
    assert "device_id" in str(response.json()["detail"])
    assert client.calls == []


def test_blank_device_id_is_rejected_clearly(client):
    """A whitespace-only device_id reaches the endpoint and is rejected there.

    The schema requires the field to be present but does not constrain its
    contents, so the endpoint's own message is still what a blank value gets.
    """
    response = client.post("/api/sensor-data", json=dict(READING, device_id="   "))
    assert response.status_code == 422
    assert "device_id is required" in response.json()["detail"]
    assert client.calls == []


# ------------------------------------------- 8. invalid crop rejected
@pytest.mark.parametrize("bad", ["cassava", "", "   ", "tomatoe"])
def test_invalid_crop_is_rejected_clearly(client, bad):
    response = register(client, crop_type=bad)
    assert response.status_code in (400, 422)
    # Either the endpoint's own message or schema validation, but the
    # rejection must name the offending field.
    assert "crop_type" in str(response.json()["detail"])


def test_unknown_crop_is_never_silently_defaulted(client):
    response = register(client, crop_type="cassava")
    assert response.status_code == 400
    assert "unknown crop_type" in response.json()["detail"]


# ------------------------- 9. known rule-only crop uses rule_fallback
@pytest.mark.parametrize("crop", sorted(crops.RULE_ONLY_CROPS))
def test_a_known_rule_only_crop_uses_rule_fallback(client, crop):
    registered = register(client, crop_type=crop, device_id=f"ESP_{crop}")
    assert registered.status_code == 201

    response = client.post(
        "/api/sensor-data", json=dict(READING, device_id=f"ESP_{crop}")
    )
    assert response.status_code == 200
    irrigation = response.json()["irrigation"]
    assert irrigation["decision_source"] == "rule_fallback"
    assert client.calls[-1]["crop_type"] == crop


# --------------------------------- prediction is stored on the reading
def test_the_prediction_is_stored_with_the_reading(client):
    register(client, crop_type="tomato", device_id="ESP32_01")
    posted = client.post(
        "/api/sensor-data", json=dict(READING, device_id="ESP32_01")
    ).json()

    latest = client.get("/api/latest-sensor-data").json()
    assert latest["id"] == posted["id"]
    assert latest["device_id"] == "ESP32_01"
    assert latest["crop_type"] == "tomato"
    assert latest["irrigation"]["decision_source"] == "model"
    assert latest["irrigation"]["prediction_score"] == 0.75


def test_a_reading_is_kept_when_the_ai_service_is_down(
    client, monkeypatch
):
    register(client, crop_type="tomato", device_id="ESP32_01")

    def boom(*args, **kwargs):
        raise ai_client.AIServiceError("AI service unreachable: refused")

    monkeypatch.setattr("main.predict_irrigation", boom)

    response = client.post(
        "/api/sensor-data", json=dict(READING, device_id="ESP32_01")
    )
    assert response.status_code == 200
    body = response.json()
    # The measurement survives; the missing advice is reported, not hidden.
    assert body["success"] is True
    assert body["irrigation"] is None
    assert "unreachable" in body["irrigation_error"]

    latest = client.get("/api/latest-sensor-data").json()
    assert latest["soil_moisture"] == 22.0
    assert latest["irrigation"]["decision_source"] is None
