from datetime import datetime, timezone
import os

from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from sqlmodel import Session, select

import ai_client
from ai_client import AIServiceError, predict_irrigation
from crops import KNOWN_CROPS, RULE_ONLY_CROPS, normalise_crop
from database import engine, create_db_and_tables, get_session
from irrigation_events import observe_pump_state


def irrigation_events_enabled() -> bool:
    """Whether pump-transition history derivation is switched on.

    Phase 2B, mirrored from `irrigation_events.ENABLED` so the endpoint can
    skip the extra SELECT entirely when derivation is disabled with
    AGRIGUARD_IRRIGATION_EVENTS=0. Read through a function rather than a bare
    import so a test can patch it.
    """
    import irrigation_events

    return irrigation_events.ENABLED

from models import (
    Plot,
    SensorReading,
    IrrigationEvent,
    AIPrediction,
    CropHealth
)

from schemas import (
    PlotCreate,
    PlotRead,
    SensorDataCreate,
    AIPredictionCreate,
    CropHealthCreate,
    IrrigationEventCreate
)


# ==================================================
# CREATE FASTAPI APP
# ==================================================

app = FastAPI(
    title="AgriGuard API",
    description="Backend API for AgriGuard Smart Agriculture System",
    version="1.0.0"
)


# ==================================================
# CORS
# ==================================================

#: Comma-separated allow-list for the dashboard, read once at import. Same
#: variable name and same parsing as the AI service (Agriguard_Ai/agriguard/
#: api.py) so one setting works for both.
CORS_ORIGINS_ENV = "AGRIGUARD_CORS_ORIGINS"


def parse_cors_origins(raw: str) -> list:
    """Split a comma-separated origin allow-list, dropping blanks.

    Blank entries are dropped rather than kept as "", because CORSMiddleware
    treats "" as a real origin and answers a null origin, which is exactly what
    an allow-list is meant to exclude.
    """
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def configure_cors(application: FastAPI, origins) -> FastAPI:
    """Attach CORS only when a non-wildcard allow-list is configured.

    The previous setting was allow_origins=["*"] with allow_credentials=True.
    That combination is rejected by browsers outright, so it was not a working
    permissive policy - it was a broken one that also advertised the API as
    open to every origin on the LAN, which matters because this service can
    write: anyone reachable could register a device->crop plot.

    CORS is opt-in here for the same reason as in the AI service. If the
    dashboard is served from the same origin (a reverse proxy in front of both)
    no header is needed. If it is hosted separately, the browser blocks the
    preflight until its origin is listed, and there is no safe implicit default,
    so none is invented here. To allow the common local dev servers:

        set AGRIGUARD_CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173

    A "*" entry is dropped rather than honoured. It arrives from an environment
    variable, so it is an operator's decision, but honouring it would silently
    restore exactly the hole this replaces; dropping it leaves no CORS
    middleware at all, which is the same-origin-safe outcome. Blank entries go
    too, because CORSMiddleware treats "" as a real origin and would answer a
    null origin, which is what an allow-list exists to exclude.
    """
    origins = [origin.strip() for origin in (origins or []) if origin and origin.strip()]
    origins = [origin for origin in origins if origin != "*"]
    if not origins:
        return application
    application.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        # No cookies or Authorization headers, so allowing credentials would
        # only widen the preflight for no benefit.
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )
    return application


CORS_ORIGINS = parse_cors_origins(os.environ.get(CORS_ORIGINS_ENV, ""))
configure_cors(app, CORS_ORIGINS)


# ==================================================
# STARTUP
# ==================================================
@app.on_event("startup")
def startup():
    create_db_and_tables()
    # Printed rather than only reachable through /api/health: a silently wrong
    # AI URL is the failure that produced a null `irrigation` on every sensor
    # POST during Phase 2A, and it looks from the dashboard exactly like the AI
    # deciding that no irrigation is needed.
    print(f"[agriguard] backend listening on port 8000")
    print(f"[agriguard] AI service URL: {ai_client.AI_URL}{ai_client.IRRIGATION_PATH}")
    print(
        "[agriguard] CORS origins: "
        f"{', '.join(CORS_ORIGINS) if CORS_ORIGINS else 'none (same-origin only)'}"
    )


# ==================================================
# ROOT
# ==================================================


@app.get("/")
def root():
    return {
        "project": "AgriGuard",
        "message": "AgriGuard Backend is running",
        "status": "online",
        "ai_url": ai_client.AI_URL,
    }


# ==================================================
# HEALTH CHECK
# ==================================================


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "service": "AgriGuard Backend",
        # Added in Phase 2B. The frontend does not read it; it exists so a
        # team member can confirm at a glance which AI service this backend will
        # call, instead of discovering a misconfiguration from a null prediction.
        "ai_url": ai_client.AI_URL,
        "crops": list(KNOWN_CROPS),
    }

# ==================================================
# PLOT CONFIGURATION
# THE CROP LIVES HERE, NOT ON THE ESP32
# ==================================================

@app.post("/api/plots", response_model=PlotRead, status_code=201)
def create_plot(data: PlotCreate, session: Session = Depends(get_session)):
    """Register a plot and bind it to the device that reports for it.

    crop_type is validated here rather than at prediction time, so a
    misconfigured plot is rejected when it is created instead of being scored
    silently later.
    """
    crop = normalise_crop(data.crop_type)
    if crop not in KNOWN_CROPS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"unknown crop_type '{data.crop_type}'. Valid crops: "
                f"{', '.join(KNOWN_CROPS)}."
            ),
        )

    device_id = data.device_id.strip()
    if not device_id:
        raise HTTPException(
            status_code=400, detail="device_id must not be blank"
        )

    existing = session.exec(
        select(Plot).where(Plot.device_id == device_id)
    ).first()
    if existing:
        raise HTTPException(
            status_code=409,
            detail=(
                f"device_id '{device_id}' is already registered to plot "
                f"'{existing.name}' (crop_type={existing.crop_type})"
            ),
        )

    plot = Plot(
        name=data.name.strip(),
        crop_type=crop,
        device_id=device_id,
    )

    try:
        session.add(plot)
        session.commit()
        session.refresh(plot)
    except Exception as e:
        session.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Plot creation error: {type(e).__name__}: {str(e)}"
        )

    return plot


@app.get("/api/plots", response_model=list[PlotRead])
def get_plots(session: Session = Depends(get_session)):
    statement = select(Plot).order_by(Plot.id)
    return session.exec(statement).all()


# ==================================================
# ESP32 → BACKEND
# POST SENSOR DATA
# ==================================================

@app.post("/api/sensor-data")
def create_sensor_data(
    data: SensorDataCreate,
    session: Session = Depends(get_session)
):
    try:
        # The device identifies the plot; the plot identifies the crop. A
        # reading with no device_id cannot be scored crop-aware, and the model
        # must never be handed a default crop, so this is a client error
        # rather than something to paper over.
        if not data.device_id or not data.device_id.strip():
            raise HTTPException(
                status_code=422,
                detail=(
                    "device_id is required: the backend resolves crop_type "
                    "from the plot this device is registered against. "
                    "Register the device with POST /api/plots first."
                ),
            )

        device_id = data.device_id.strip()

        plot = session.exec(
            select(Plot).where(Plot.device_id == device_id)
        ).first()

        if not plot:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"no plot configured for device_id '{device_id}'. "
                    f"Register it with POST /api/plots (name, crop_type, "
                    f"device_id) before sending sensor data. No crop-aware "
                    f"prediction is made without a crop."
                ),
            )

        timestamp = data.timestamp or datetime.now(timezone.utc)

        sensor = SensorReading(
            timestamp=timestamp,

            soil_moisture=data.soil_moisture,

            temperature=data.temperature,

            humidity=data.humidity,

            light=data.light,

            water_flow=data.water_flow,

            pump_status=data.pump_status,

            nitrogen=(
                data.npk.nitrogen
                if data.npk
                else None
            ),

            phosphorus=(
                data.npk.phosphorus
                if data.npk
                else None
            ),

            potassium=(
                data.npk.potassium
                if data.npk
                else None
            ),

            device_id=device_id,

            # Snapshot of the resolved crop, stored so this row stays
            # explainable if the plot is reconfigured later.
            crop_type=plot.crop_type
        )

        # Phase 2B: capture the previous reading for this device BEFORE the new one
        # is inserted, so that (previous, current) is unambiguous. Querying
        # after the insert would return the new row as its own predecessor.
        if irrigation_events_enabled():
            previous = session.exec(
                select(SensorReading)
                .where(SensorReading.device_id == device_id)
                .order_by(
                    SensorReading.timestamp.desc(),
                    SensorReading.id.desc(),
                )
            ).first()
        else:
            previous = None

        # Store the measurement before asking for advice, so a prediction
        # failure never loses the reading itself.
        session.add(sensor)

        session.commit()

        session.refresh(sensor)

        # Phase 2B: derive the irrigation event from the pump transition this
        # reading represents, so /api/irrigation-history and /api/water-usage
        # have a producer without the Phase 2A firmware posting anything new.
        if previous is not None or bool(sensor.pump_status):
            try:
                transition = observe_pump_state(
                    session, device_id, previous, sensor
                )
                if transition and transition["action"] == "closed":
                    print(
                        "[agriguard] irrigation event closed: "
                        f"device={device_id} "
                        f"duration={transition['duration_seconds']}s "
                        f"water_used={transition['water_used']}L "
                        f"(flow integrated: {transition['flow_integrated']})"
                    )
            except Exception as e:  # noqa: BLE001
                # The measurement is already committed. Failing to record
                # history is a reporting gap, not a failed submission, and the
                # caller must not retry and duplicate the reading because of it.
                print(
                    "[agriguard] WARN: irrigation event derivation failed "
                    f"({type(e).__name__}); the reading was still stored"
                )

        prediction = None

        try:
            prediction = predict_irrigation(
                {
                    "timestamp": timestamp.isoformat(),
                    "device_id": device_id,
                    # str(Plot.id) so the AI's plot_id means the same thing
                    # here as it does in the response below. It used to carry
                    # the plot *name* here and the numeric id there, so a
                    # caller correlating AI history with a plot could not
                    # tell which one it had been sent.
                    "plot_id": str(plot.id),
                    "soil_moisture": data.soil_moisture,
                    "temperature": data.temperature,
                    "humidity": data.humidity,
                    "light": data.light,
                    "water_flow": data.water_flow,
                    "pump_status": int(bool(data.pump_status)),
                    "crop_type": plot.crop_type,
                },
                crop_type=plot.crop_type,
            )
        except AIServiceError as e:
            prediction = None
            ai_error = str(e)
        except Exception as e:  # noqa: BLE001
            # The reading is already committed at this point. Turning an
            # unexpected failure here into a 500 would tell the client the
            # whole submission failed, and a retrying client would insert a
            # second reading for the same event. Degrade to "no prediction",
            # exactly like an unreachable AI, and report only the exception
            # type: the message may contain internal detail.
            prediction = None
            ai_error = f"AI service call failed ({type(e).__name__})"
        else:
            ai_error = None

        if prediction:
            sensor.irrigation_needed = prediction["irrigation_needed"]
            sensor.prediction_score = prediction["prediction_score"]
            sensor.decision_source = prediction["decision_source"]
            sensor.prediction_reason = prediction["reason"]
            session.add(sensor)
            session.commit()
            session.refresh(sensor)

        response = {
            "success": True,
            "message": "Sensor data stored successfully",
            "id": sensor.id,
            "device_id": device_id,
            "plot_id": plot.id,
            "crop_type": plot.crop_type,
            "irrigation": prediction,
        }

        if ai_error:
            # The measurement is safe; only the advice is missing. Reported
            # explicitly so no caller reads a missing prediction as
            # "no irrigation needed".
            response["irrigation_error"] = ai_error

        return response

    except HTTPException:
        session.rollback()

        raise

    except Exception as e:
        session.rollback()

        raise HTTPException(
            status_code=500,
            detail=f"Sensor data error: {type(e).__name__}: {str(e)}"
        )


# ==================================================
# GET SENSOR HISTORY
# BACKEND <-> DASHBOARD
# ==================================================
#
# Phase 2B: optional `?limit=`, newest-first, capped.
#
# The dashboard refetches this endpoint on every refresh, and the ESP32 posts
# every 15 s, so the table grows by 5 760 rows a day. Without a bound the
# dashboard downloaded the entire history every 15 s and rebuilt all three
# charts from it. `limit` is deliberately optional so existing callers that
# want the full history keep working unchanged.
#
# The response shape is untouched: still a JSON array of SensorReading rows,
# newest first.

@app.get("/api/sensor-data")
def get_sensor_data(
    limit: int | None = Query(
        default=None,
        ge=1,
        le=5000,
        description=(
            "Maximum number of readings to return, newest first. "
            "Omit for the full history."
        ),
    ),
    session: Session = Depends(get_session)
):

    statement = (
        select(SensorReading)
        .order_by(
            SensorReading.timestamp.desc(),
            SensorReading.id.desc(),
        )
    )

    if limit is not None:
        statement = statement.limit(limit)

    results = session.exec(statement).all()

    return results


# ==================================================
# GET LATEST SENSOR DATA
# BACKEND → DASHBOARD
# ==================================================

@app.get("/api/latest-sensor-data")
def get_latest_sensor_data(
    session: Session = Depends(get_session)
):

    statement = (
        select(SensorReading)
        .order_by(
            SensorReading.timestamp.desc()
        )
    )

    latest = session.exec(statement).first()

    if not latest:
        raise HTTPException(
            status_code=404,
            detail="No sensor data available"
        )

    irrigation_status = (
        "Irrigating"
        if latest.pump_status
        else "Not Irrigating"
    )

    return {
        "id": latest.id,

        "timestamp": latest.timestamp,

        "soil_moisture": latest.soil_moisture,

        "temperature": latest.temperature,

        "humidity": latest.humidity,

        "light": latest.light,

        "light_intensity": latest.light,

        "water_flow": latest.water_flow,

        "pump_status": latest.pump_status,

        "irrigation_status": irrigation_status,

        "npk": {
            "nitrogen": latest.nitrogen,

            "phosphorus": latest.phosphorus,

            "potassium": latest.potassium
        },

        "device_id": latest.device_id,

        "crop_type": latest.crop_type,

        "irrigation": {
            "irrigation_needed": latest.irrigation_needed,

            "prediction_score": latest.prediction_score,

            "decision_source": latest.decision_source,

            "reason": latest.prediction_reason,
        }
    }


# ==================================================
# AI → BACKEND
# POST AI PREDICTION
# ==================================================

@app.post("/api/ai-prediction")
def create_ai_prediction(
    data: AIPredictionCreate,
    session: Session = Depends(get_session)
):

    prediction = AIPrediction(
        irrigation_needed=data.irrigation_needed,

        prediction=data.prediction,

        confidence=data.confidence,

        recommended_action=data.recommended_action
    )

    session.add(prediction)

    session.commit()

    session.refresh(prediction)

    return {
        "success": True,

        "message": "AI prediction stored successfully",

        "id": prediction.id
    }


# ==================================================
# GET LATEST AI PREDICTION
# ==================================================

@app.get("/api/ai-prediction")
def get_ai_prediction(
    session: Session = Depends(get_session)
):

    statement = (
        select(AIPrediction)
        .order_by(
            AIPrediction.timestamp.desc()
        )
    )

    prediction = session.exec(statement).first()

    if not prediction:
        return {
            "prediction": "No prediction available",

            "confidence": None,

            "recommended_action":
                "Waiting for AI prediction."
        }

    return {
        "id": prediction.id,

        "timestamp": prediction.timestamp,

        "irrigation_needed":
            prediction.irrigation_needed,

        "prediction":
            prediction.prediction,

        "confidence":
            prediction.confidence,

        "recommended_action":
            prediction.recommended_action
    }


# ==================================================
# AI → BACKEND
# POST CROP HEALTH
# ==================================================

@app.post("/api/crop-health")
def create_crop_health(
    data: CropHealthCreate,
    session: Session = Depends(get_session)
):

    health = CropHealth(
        crop_condition=data.crop_condition,

        confidence=data.confidence,

        alert=data.alert,

        message=data.message
    )

    session.add(health)

    session.commit()

    session.refresh(health)

    return {
        "success": True,

        "message":
            "Crop health result stored successfully",

        "id": health.id
    }


# ==================================================
# GET LATEST CROP HEALTH
# ==================================================

@app.get("/api/crop-health")
def get_crop_health(
    session: Session = Depends(get_session)
):

    statement = (
        select(CropHealth)
        .order_by(
            CropHealth.timestamp.desc()
        )
    )

    health = session.exec(statement).first()

    if not health:
        return {
            "result": "Unavailable",

            "crop_condition": "Unavailable",

            "confidence": None,

            "alert": False,

            "message":
                "No crop-health result available."
        }

    return {
        "id": health.id,

        "timestamp": health.timestamp,

        "result":
            health.crop_condition,

        "crop_condition":
            health.crop_condition,

        "confidence":
            health.confidence,

        "alert":
            health.alert,

        "message":
            health.message
    }


# ==================================================
# POST IRRIGATION EVENT
# ==================================================

@app.post("/api/irrigation-events")
def create_irrigation_event(
    data: IrrigationEventCreate,
    session: Session = Depends(get_session)
):

    event = IrrigationEvent(
        start_time=(
            data.start_time
            or datetime.now(timezone.utc)
        ),

        duration_seconds=
            data.duration_seconds,

        water_used=
            data.water_used
    )

    session.add(event)

    session.commit()

    session.refresh(event)

    return {
        "success": True,

        "message":
            "Irrigation event stored successfully",

        "id": event.id
    }


# ==================================================
# GET IRRIGATION HISTORY
# ==================================================

@app.get("/api/irrigation-history")
def get_irrigation_history(
    session: Session = Depends(get_session)
):

    statement = (
        select(IrrigationEvent)
        .order_by(
            IrrigationEvent.start_time.desc()
        )
    )

    events = session.exec(statement).all()

    return [
        {
            "time":
                event.start_time.strftime("%H:%M"),

            "timestamp":
                event.start_time,

            "duration":
                f"{event.duration_seconds // 60} min",

            "water_used":
                event.water_used
        }

        for event in events
    ]


# ==================================================
# WATER USAGE
# ==================================================

@app.get("/api/water-usage")
def get_water_usage(
    session: Session = Depends(get_session)
):

    statement = select(IrrigationEvent)

    events = session.exec(statement).all()

    total = sum(
        event.water_used
        for event in events
    )

    return {
        "current_consumption":
            round(total, 2),

        "total_consumption":
            round(total, 2)
    }