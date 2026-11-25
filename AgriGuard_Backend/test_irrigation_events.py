"""Phase 2B: pump-transition irrigation-event derivation and bounded history.

These cover the two Backend behaviours added in Phase 2B. The rest of the
suite (62 tests) is unchanged and still passes.

The point of `irrigation_events` is that the Phase 2A firmware posts nothing new:
it already sends `pump_status` and `water_flow` on every reading, and the event
is reconstructed from the transitions between consecutive readings. So these
tests drive the endpoint with the exact payload the firmware sends, including
the "no pump_status gap" case a real deployment hits when a device drops off.
"""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, Session, create_engine

sys.path.insert(0, str(Path(__file__).resolve().parent))

import irrigation_events  # noqa: E402
import models  # noqa: E402,F401  (registers the tables on SQLModel.metadata)
from main import app  # noqa: E402
from models import IrrigationEvent, SensorReading  # noqa: E402

BASE = {
    "soil_moisture": 30.0,
    "temperature": 25.0,
    "humidity": 60.0,
    "light": 18000.0,
}

T0 = datetime(2026, 3, 1, 12, 0, 0, tzinfo=timezone.utc)


def reading(offset_s, pump, flow):
    return dict(
        BASE,
        water_flow=flow,
        pump_status=pump,
        timestamp=(T0 + timedelta(seconds=offset_s)).isoformat(),
    )


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """TestClient on an isolated DB, with the AI call stubbed out.

    The stub returns a prediction so the endpoint takes its normal happy path;
    history derivation must work identically whether or not the AI answered.
    """
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

    monkeypatch.setattr(
        "main.predict_irrigation",
        lambda reading, crop_type, history=None: {
            "irrigation_needed": True,
            "prediction": "Irrigation Recommended",
            "prediction_score": 0.75,
            "decision_source": "model",
            "reason": "stubbed",
            "advisory_only": True,
            "model": "irrigation-lr",
            "version": "0.2.0",
        },
    )

    irrigation_events.reset_state()

    with TestClient(app) as test_client:
        test_client.engine = engine
        yield test_client

    app.dependency_overrides = {}
    irrigation_events.reset_state()


def register(client, device_id="ESP32_01", crop="tomato"):
    response = client.post(
        "/api/plots",
        json={"name": "Test Field", "crop_type": crop, "device_id": device_id},
    )
    assert response.status_code == 201
    return response.json()


def events(client):
    with Session(client.engine) as session:
        return session.exec(
            __import__("sqlmodel").select(IrrigationEvent)
        ).all()


# ---------------------------------------------------------------- limit


def test_history_is_bounded_by_limit(client):
    """`?limit=` returns the newest N rows and leaves the rest in the table."""
    register(client)
    for i in range(12):
        assert client.post(
            "/api/sensor-data",
            json=dict(
                reading(i * 15, False, 0.0),
                device_id="ESP32_01",
            ),
        ).status_code == 200

    limited = client.get("/api/sensor-data?limit=5").json()
    assert len(limited) == 5

    # Newest first, so the five returned are the last five posted.
    ids_full = [r["id"] for r in client.get("/api/sensor-data").json()]
    assert [r["id"] for r in limited] == ids_full[:5]

    # The row shape is unchanged; `limit` only bounds the count.
    assert set(limited[0].keys()) >= {
        "timestamp",
        "soil_moisture",
        "temperature",
        "humidity",
        "light",
        "water_flow",
        "pump_status",
    }


def test_history_without_limit_is_unchanged(client):
    """Omitting `?limit=` keeps the previous full-history behaviour."""
    register(client)
    for i in range(4):
        client.post(
            "/api/sensor-data",
            json=dict(reading(i * 15, False, 0.0), device_id="ESP32_01"),
        )
    assert len(client.get("/api/sensor-data").json()) == 4


def test_limit_is_validated(client):
    """A nonsensical limit is a client error, not a silent full dump."""
    assert client.get("/api/sensor-data?limit=0").status_code == 422
    assert client.get("/api/sensor-data?limit=-5").status_code == 422
    assert client.get("/api/sensor-data?limit=999999").status_code == 422
    assert client.get("/api/sensor-data?limit=abc").status_code == 422


# ------------------------------------------------- irrigation derivation


def test_a_pump_cycle_produces_one_event(client):
    """off -> on -> off creates exactly one IrrigationEvent."""
    register(client)

    # Pump off, then on (event opens), then on again, then off (event closes).
    client.post("/api/sensor-data", json=dict(reading(0, False, 0.0), device_id="ESP32_01"))
    client.post("/api/sensor-data", json=dict(reading(15, True, 2.0), device_id="ESP32_01"))
    client.post("/api/sensor-data", json=dict(reading(30, True, 2.0), device_id="ESP32_01"))
    client.post("/api/sensor-data", json=dict(reading(45, False, 0.0), device_id="ESP32_01"))

    rows = events(client)
    assert len(rows) == 1

    event = rows[0]
    assert event.start_time == T0 + timedelta(seconds=15)
    assert event.duration_seconds == 30
    # 2.0 L/min over the 15 s intervals at t=15..30 and t=30..45:
    #   2.0 * (15/60) + 2.0 * (15/60) = 1.0 L
    assert event.water_used == pytest.approx(1.0, abs=1e-6)


def test_no_event_while_the_pump_stays_off(client):
    """Steady idle readings must not manufacture irrigation history."""
    register(client)
    for i in range(5):
        client.post(
            "/api/sensor-data",
            json=dict(reading(i * 15, False, 0.0), device_id="ESP32_01"),
        )
    assert events(client) == []
    assert client.get("/api/water-usage").json()["total_consumption"] == 0
    assert client.get("/api/irrigation-history").json() == []


def test_an_unfinished_pump_cycle_stays_open(client):
    """A pump that never reports OFF has no completed event yet.

    Claiming a duration for a cycle that is still running, or that was never
    observed ending, would be an invention. The event appears on the OFF
    transition.
    """
    register(client)
    client.post("/api/sensor-data", json=dict(reading(0, False, 0.0), device_id="ESP32_01"))
    client.post("/api/sensor-data", json=dict(reading(15, True, 3.0), device_id="ESP32_01"))
    client.post("/api/sensor-data", json=dict(reading(30, True, 3.0), device_id="ESP32_01"))

    assert events(client) == []


def test_a_gap_longer_than_the_bound_does_not_credit_unmeasured_water(client):
    """A device that was offline must not have litres invented across the gap.

    The firmware posts every 15 s. A 6-hour gap means nothing was observed in
    between. Two things would both be inventions: carrying the 4 L/min seen
    before the outage across six hours (1 440 L), or trimming the interval to
    15 s and claiming 1 L, which asserts the pump stopped after 15 s when
    nobody observed that either. So the event closes with 0 L measured, and the
    duration still spans the outage, which is what really happened.
    """
    register(client)
    client.post("/api/sensor-data", json=dict(reading(0, False, 0.0), device_id="ESP32_01"))
    client.post("/api/sensor-data", json=dict(reading(15, True, 4.0), device_id="ESP32_01"))
    # 6 h later the pump is finally reported off.
    client.post(
        "/api/sensor-data",
        json=dict(reading(6 * 3600, False, 0.0), device_id="ESP32_01"),
    )

    rows = events(client)
    assert len(rows) == 1
    assert rows[0].water_used == 0.0
    # The duration still reflects reality, including the outage.
    assert rows[0].duration_seconds == 6 * 3600 - 15


def test_history_endpoints_are_populated_by_a_real_cycle(client):
    """The two dashboard cards read these endpoints, so assert their shape."""
    register(client)
    client.post("/api/sensor-data", json=dict(reading(0, False, 0.0), device_id="ESP32_01"))
    client.post("/api/sensor-data", json=dict(reading(15, True, 2.0), device_id="ESP32_01"))
    client.post("/api/sensor-data", json=dict(reading(30, False, 0.0), device_id="ESP32_01"))

    history = client.get("/api/irrigation-history").json()
    assert len(history) == 1
    assert set(history[0].keys()) == {"time", "timestamp", "duration", "water_used"}
    assert history[0]["duration"] == "0 min"

    usage = client.get("/api/water-usage").json()
    assert usage["total_consumption"] == pytest.approx(0.5, abs=1e-6)


def test_devices_are_tracked_independently(client):
    """Two plots must not merge into one another's irrigation event."""
    register(client, device_id="ESP32_01")
    register(client, device_id="ESP32_02", crop="rice")

    client.post("/api/sensor-data", json=dict(reading(0, True, 2.0), device_id="ESP32_01"))
    client.post("/api/sensor-data", json=dict(reading(5, False, 0.0), device_id="ESP32_02"))
    client.post("/api/sensor-data", json=dict(reading(15, False, 0.0), device_id="ESP32_01"))

    rows = events(client)
    assert len(rows) == 1
    assert rows[0].start_time == T0


def test_derivation_failure_never_fails_the_reading(client, monkeypatch):
    """A history write error must not turn a stored measurement into a 500.

    The reading is committed before history is derived. If the caller saw a
    500 it would retry, and the retry would insert a second row for one
    physical event.
    """
    register(client)

    def boom(*args, **kwargs):
        raise RuntimeError("simulated history failure")

    monkeypatch.setattr("main.observe_pump_state", boom)

    client.post("/api/sensor-data", json=dict(reading(0, False, 0.0), device_id="ESP32_01"))
    response = client.post(
        "/api/sensor-data",
        json=dict(reading(15, True, 2.0), device_id="ESP32_01"),
    )

    assert response.status_code == 200
    assert response.json()["success"] is True

    with Session(client.engine) as session:
        assert len(session.exec(
            __import__("sqlmodel").select(SensorReading)
        ).all()) == 2


def test_derivation_can_be_disabled(client, monkeypatch):
    """AGRIGUARD_IRRIGATION_EVENTS=0 restores the previous empty-history behaviour."""
    register(client)
    monkeypatch.setattr(irrigation_events, "ENABLED", False)

    client.post("/api/sensor-data", json=dict(reading(0, False, 0.0), device_id="ESP32_01"))
    client.post("/api/sensor-data", json=dict(reading(15, True, 2.0), device_id="ESP32_01"))
    client.post("/api/sensor-data", json=dict(reading(30, False, 0.0), device_id="ESP32_01"))

    assert events(client) == []
    assert client.get("/api/water-usage").json()["total_consumption"] == 0