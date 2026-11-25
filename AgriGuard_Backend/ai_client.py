"""Thin client for the AgriGuard AI service.

The backend owns crop resolution: it sends the crop configured for the plot
rather than trusting the firmware to name it. Every prediction therefore
carries an explicit `crop_type`, and the AI's own `decision_source` is passed
back to the caller untouched, because it is the only thing that distinguishes an
in-domain model answer from a rule answer.

The water policy deliberately does not live here. This client returns advice;
acting on it is the ESP32/backend's job, and the AI service is advisory only.
"""

import os
from typing import Any, Dict, Optional

import httpx

#: Default AI service address. The port is deliberately 8001 and NOT 8000: this
#: backend itself is served on 8000, so an AI default of 8000 made the backend
#: POST its own port when AGRIGUARD_AI_URL was unset. That is a silent failure -
#: the sensor POST still returned 200, but `irrigation` was always null with an
#: `irrigation_error`, which reads exactly like "the AI decided no irrigation".
#: The main backend logs the resolved URL on startup so this is visible
#: immediately rather than inferred from a null prediction.
AI_URL = os.getenv("AGRIGUARD_AI_URL", "http://127.0.0.1:8001")
TIMEOUT = httpx.Timeout(30.0, connect=5.0)

IRRIGATION_PATH = "/v1/predict/irrigation"


class AIServiceError(RuntimeError):
    """The AI service could not be reached, or answered unusably."""


def predict_irrigation(
    reading: Dict[str, Any],
    crop_type: str,
    history: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Ask the AI for an irrigation recommendation for one reading.

    `reading` is the sensor payload; `crop_type` is injected here so it always
    comes from the plot configuration. Raises AIServiceError rather than
    returning a partial answer: a reading stored without a prediction must be
    visible as "no prediction", never as a silent zero.
    """
    payload = dict(reading)
    payload["crop_type"] = crop_type
    if history:
        payload["history"] = history

    try:
        response = httpx.post(
            f"{AI_URL.rstrip('/')}{IRRIGATION_PATH}",
            json=payload,
            timeout=TIMEOUT,
        )
        response.raise_for_status()
        body = response.json()
    except httpx.HTTPError as exc:
        raise AIServiceError(f"AI service unreachable: {exc}") from exc
    except ValueError as exc:
        raise AIServiceError(f"AI service returned invalid JSON: {exc}") from exc

    # The transport succeeding only proves *something* answered with 2xx and
    # parseable JSON. A reverse proxy, a different service on the configured
    # port, or a misrouted request can all return a JSON array, a bare string,
    # or a number, and calling .get() on one of those raises AttributeError.
    # That escapes as an unhandled 500 from the endpoint and, because the
    # reading is committed before the AI is called, the client is told the
    # whole submission failed when it was not. Check the type explicitly and
    # report it as the AI contract being violated.
    if not isinstance(body, dict):
        raise AIServiceError(
            "AI service returned "
            f"{type(body).__name__}, expected a JSON object"
        )

    prediction = body.get("prediction")
    if not isinstance(prediction, dict):
        raise AIServiceError("AI response has no 'prediction' object")

    return {
        "irrigation_needed": prediction.get("irrigation_needed"),
        "prediction": prediction.get("prediction"),
        "prediction_score": prediction.get("prediction_score"),
        "decision_source": prediction.get("decision_source"),
        "reason": prediction.get("reason"),
        "advisory_only": prediction.get("advisory_only"),
        "model": body.get("model"),
        "version": body.get("version"),
    }
