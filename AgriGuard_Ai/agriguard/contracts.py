"""JSON contract schemas for AgriGuard AI predictions.

These field names match the data the ESP32 actually collects and the integration
format agreed with Member 3 (backend/dashboard).

Raw ESP32 fields -> name used here:

    ESP32 sensor data *must* be mapped to these names before calling the AI.

INPUT (from ESP32/backend):
    timestamp        ISO-8601 string
    device_id        ESP32 node id, e.g. "ESP32_01"
    plot_id          agricultural plot id, e.g. "PLOT_A"
    soil_moisture    sensor scale used by the project (see dataset_structure.md)
    temperature      ambient temperature
    humidity         relative humidity
    light            light-intensity reading (raw sensor scale, NOT assumed lux)
    npk_n / npk_p / npk_k   optional, null when no NPK sensor is fitted
    water_flow       water-flow reading (raw sensor scale), 0 when the pump is off
    pump_status      0 = off, 1 = on
    history          optional recent readings used to build the historical features

OUTPUT:
    irrigation_needed         bool
    prediction                "Irrigation Recommended" / "Irrigation Not Needed"
    prediction_score          0..1 model score (NOT a calibrated probability)
    irrigation_duration_min   null - AgriGuard has no defensible duration model
    reason                    short human-readable explanation
    history_used              true / false / null - see the field description
    advisory_only             always true - the AI never commands hardware

The model is a *recommendation* system. The backend / IoT layer decides whether
and how to act on it; see docs/integration_guide.md and TASK 14 of the brief.
"""

from typing import List, Optional

from pydantic import BaseModel, Field

#: light and water_flow are intentionally unbounded above: the ESP32 firmware
#: may report a raw ADC count, a lux value or a pulse count depending on the
#: sensor that Member 1 fitted. We only reject negative values.
MAX_LIGHT = 200000.0
MAX_WATER_FLOW = 100000.0
MAX_TEMPERATURE = 80.0
MIN_TEMPERATURE = -40.0


class HistoryReading(BaseModel):
    """One earlier reading for the same device_id/plot_id pair."""

    timestamp: str = Field(description="ISO-8601 timestamp of the earlier reading")
    soil_moisture: float = Field(
        ge=0, le=100, description="soil moisture recorded at that time"
    )


class IrrigationHistory(BaseModel):
    """Optional sequential context.

    The irrigation model uses three historical features
    (soil_moisture_change, soil_moisture_avg_30min, time_since_last_irrigation_min).
    The backend may either send `recent_soil_moisture` and `last_irrigation_at`
    so the AI service can build them, or leave this block out - in which case the
    service substitutes neutral placeholders (current reading for the average, 0
    for the change, "nothing recent" for the time since) and reports
    `history_used: false` plus an explanation in `reason`.

    Send the history. Without it the model recommends water far too often:
    measured recall 1.00 at precision 0.16 on the test split.
    """

    recent_soil_moisture: Optional[List[HistoryReading]] = Field(
        default=None,
        description="Earlier readings for the same plot, oldest first",
    )
    last_irrigation_at: Optional[str] = Field(
        default=None,
        description="ISO-8601 time of the last observed pump run (pump_status=1)",
    )


class IrrigationRequest(BaseModel):
    timestamp: Optional[str] = Field(
        default=None, description="ISO-8601 timestamp from the ESP32"
    )
    device_id: Optional[str] = Field(
        default=None, description="ESP32 node identifier, e.g. ESP32_01"
    )
    plot_id: Optional[str] = Field(
        default=None, description="agricultural plot identifier, e.g. PLOT_A"
    )
    soil_moisture: float = Field(ge=0, le=100, description="soil moisture, 0-100")
    temperature: float = Field(
        ge=MIN_TEMPERATURE,
        le=MAX_TEMPERATURE,
        description="ambient temperature, Celsius",
    )
    humidity: float = Field(ge=0, le=100, description="relative humidity, 0-100 %")
    light: float = Field(
        ge=0, le=MAX_LIGHT, description="light intensity in the ESP32 sensor scale"
    )
    npk_n: Optional[float] = Field(
        default=None, ge=0, description="nitrogen reading, null when no NPK sensor"
    )
    npk_p: Optional[float] = Field(
        default=None, ge=0, description="phosphorus reading, null when no NPK sensor"
    )
    npk_k: Optional[float] = Field(
        default=None, ge=0, description="potassium reading, null when no NPK sensor"
    )
    water_flow: float = Field(
        default=0.0, ge=0, le=MAX_WATER_FLOW, description="water flow, 0 when the pump is off"
    )
    pump_status: int = Field(
        default=0, ge=0, le=1, description="pump state: 0 = off, 1 = on"
    )
    crop_type: str = Field(
        default="default",
        description="crop grown in the plot, e.g. 'tomato'. Plot configuration, "
        "not an ESP32 sensor: the backend resolves it from the plot the reading "
        "came from. 'tomato', 'rice' and 'pepper' are the crops the trained model "
        "has indicator columns for and are the only ones scored by the model; a "
        "crop with no column is answered by the rule-based fallback and reports "
        "decision_source 'rule_fallback'. A crop name that is not recognised never "
        "reaches the model, so it cannot be scored against an all-zero crop block",
    )
    history: Optional[IrrigationHistory] = Field(
        default=None,
        description="optional earlier readings for the same plot, used to build the "
        "historical features",
    )


class IrrigationPrediction(BaseModel):
    irrigation_needed: bool = Field(
        description="True when the AI recommends irrigation"
    )
    prediction: str = Field(
        description='"Irrigation Recommended" or "Irrigation Not Needed"'
    )
    prediction_score: float = Field(
        ge=0,
        le=1,
        description="estimated strength of the evidence for 'irrigation needed', "
        "on the same scale on every path. A backend can re-derive the decision as "
        "`irrigation_needed == (prediction_score >= decision_threshold)`, and that "
        "identity holds whether the score came from the model or from the rule "
        "fallback. This is a score, not a calibrated probability of correctness.",
    )
    decision_source: str = Field(
        description="what produced this decision: 'model' (the trained logistic "
        "regression answered), 'pump_guard' / 'water_guard' (a safety guard, so "
        "no second irrigation is requested), 'rule_fallback' (no model file was "
        "available), or 'rule_fallback_out_of_range' (a client-supplied reading "
        "fell outside the range the model was fitted on, so its extrapolated "
        "score is quoted in `reason` for diagnosis only and the crop threshold "
        "rule decided instead). Read this to know how much the number is worth, "
        "not to re-derive the decision.",
    )
    irrigation_duration_min: Optional[int] = Field(
        default=None,
        description="always null - AgriGuard has no defensible irrigation-duration "
        "model. The backend owns any runtime policy.",
    )
    reason: str = Field(description="short human-readable explanation")
    history_used: Optional[bool] = Field(
        default=None,
        description="true when the three historical features came from real "
        "readings, false when they were placeholders, null when the model was "
        "not consulted (safety guard or rule fallback). A false score should not "
        "be shown as a number in the dashboard - see "
        "models/irrigation_metrics.json -> test_metrics_without_history.",
    )
    advisory_only: bool = Field(
        default=True,
        description="always true. The AI never switches hardware; the backend/IoT "
        "layer applies its own safety limits before acting.",
    )


class IrrigationResponse(BaseModel):
    model: str = "irrigation"
    version: str
    prediction: IrrigationPrediction


class CropHealthPrediction(BaseModel):
    class_: str = Field(
        alias="class",
        description="internal class key: healthy | possible_disease",
    )
    label: str = Field(description='API label: "Healthy" or "Possible Disease"')
    confidence: float = Field(
        ge=0,
        le=1,
        description="softmax score for the returned class. Not calibrated: it is a "
        "relative score, not a guarantee of correctness.",
    )
    method: str = Field(
        description="which model produced the answer, e.g. MobileNetV3-Small"
    )
    model_version: str = Field(description="version of the crop-health weights")


class CropHealthResponse(BaseModel):
    model: str = "crop-health"
    version: str
    prediction: CropHealthPrediction
