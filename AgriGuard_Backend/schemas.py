from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class NPKData(BaseModel):
    nitrogen: Optional[float] = None
    phosphorus: Optional[float] = None
    potassium: Optional[float] = None


class PlotCreate(BaseModel):
    name: str = Field(min_length=1, description="human-readable plot name")
    crop_type: str = Field(
        min_length=1, description="crop grown in the plot, e.g. 'tomato'"
    )
    device_id: str = Field(
        min_length=1,
        description="ESP32 node identifier that reports for this plot, e.g. ESP32_01",
    )


class PlotRead(BaseModel):
    id: int
    name: str
    crop_type: str
    device_id: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class SensorDataCreate(BaseModel):
    timestamp: Optional[datetime] = None
    soil_moisture: float
    temperature: float
    humidity: float
    light: float
    water_flow: float
    pump_status: bool
    npk: Optional[NPKData] = None
    device_id: str = Field(
        description="ESP32 node identifier. Required for a crop-aware prediction: "
        "the backend resolves crop_type from the plot this device is registered "
        "against. The firmware does not send crop_type.",
    )


class AIPredictionCreate(BaseModel):
    irrigation_needed: bool
    prediction: str
    confidence: float
    recommended_action: Optional[str] = None


class CropHealthCreate(BaseModel):
    crop_condition: str
    confidence: float
    alert: bool = False
    message: Optional[str] = None


class IrrigationEventCreate(BaseModel):
    start_time: Optional[datetime] = None
    duration_seconds: int
    water_used: float