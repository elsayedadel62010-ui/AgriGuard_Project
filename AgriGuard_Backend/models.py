from datetime import datetime, timezone
from typing import Optional

from sqlmodel import SQLModel, Field


class Plot(SQLModel, table=True):
    """A growing plot, and the device that reports readings for it.

    Crop identity is plot configuration, never an ESP32 sensor value: the
    firmware sends `device_id` and the backend resolves the crop from here, so
    a plot's crop can be corrected without reflashing hardware.

    `device_id` is unique so that device_id -> Plot -> crop_type resolves to
    exactly one crop. A second plot claiming the same device is rejected
    rather than silently shadowing the first, because that ambiguity would
    quietly change which crop every reading is scored against.
    """

    id: Optional[int] = Field(default=None, primary_key=True)

    name: str

    crop_type: str

    device_id: str = Field(unique=True, index=True)

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class SensorReading(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    soil_moisture: float
    temperature: float
    humidity: float
    light: float
    water_flow: float
    pump_status: bool

    nitrogen: Optional[float] = None
    phosphorus: Optional[float] = None
    potassium: Optional[float] = None

    # Resolved from the Plot the reporting device is registered against, and
    # stored on the reading as a snapshot. The Plot remains the source of
    # truth; this is a copy of what the prediction was actually made against,
    # so a row stays explainable after a plot's crop is reconfigured.
    device_id: Optional[str] = None
    crop_type: Optional[str] = None

    # The irrigation prediction this reading produced. Advisory only: the
    # backend records what the AI said and never acts on it here.
    irrigation_needed: Optional[bool] = None
    prediction_score: Optional[float] = None
    decision_source: Optional[str] = None
    prediction_reason: Optional[str] = None


class IrrigationEvent(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)

    start_time: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    duration_seconds: int
    water_used: float


class AIPrediction(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    irrigation_needed: bool
    prediction: str
    confidence: float
    recommended_action: Optional[str] = None


class CropHealth(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    crop_condition: str
    confidence: float
    alert: bool = False
    message: Optional[str] = None