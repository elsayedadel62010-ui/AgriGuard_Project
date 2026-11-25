AgriGuard Backend API
=====================

FastAPI service that receives ESP32 sensor readings, resolves the crop from the
plot each device is registered against, asks the AI service for an irrigation
prediction, and stores both.

Run
---
    pip install -r requirements.txt
    uvicorn main:app --reload            # listens on 8000

Configuration (environment variables, all optional)
--------------------------------------------------
    AGRIGUARD_AI_URL        Base URL of the AI service. The backend calls
                            {AGRIGUARD_AI_URL}/v1/predict/irrigation. Both
                            services default to port 8000, so if you are running
                            the AI on its own port you must set this, e.g.
                            AGRIGUARD_AI_URL=http://127.0.0.1:8001. If it is
                            wrong, readings are still stored and reported with
                            no prediction plus an irrigation_error; nothing is
                            lost.
    AGRIGUARD_CORS_ORIGINS  Comma-separated CORS allow-list for the dashboard,
                            e.g. "http://localhost:5173,http://127.0.0.1:5173".
                            Unset (the default) means no CORS middleware is
                            added, which is correct when the dashboard is served
                            from the same origin behind a reverse proxy. A "*"
                            entry is ignored rather than honoured; there is no
                            wildcard mode.

Interactive API docs: http://localhost:8000/docs


How a reading gets its crop
---------------------------
The ESP32 never sends a crop. Resolution is one hop, server-side:

    ESP32 device_id  ->  Plot row  ->  Plot.crop_type  ->  AI payload

`device_id` identifies the physical device, the Plot row is the configuration
an operator registers once, and `crop_type` is read from that row on every
reading. Consequences worth knowing:

* A `crop_type` field in a device payload is ignored. `SensorDataCreate` has no
  such field and Pydantic drops unknown keys, so firmware cannot spoof the crop
  and cannot accidentally change it.
* A reading whose `device_id` is not registered is rejected rather than scored
  against a default crop.
* `crop_type` is snapshotted onto each stored reading, so history stays correct
  even if a plot is reconfigured later.
* `plot_id` means the numeric Plot id in both places it appears: the `plot_id`
  field of the AI payload (as a string, because the AI contract types it as
  one) and the `plot_id` field of this API's response (as an integer).


Crops
-----
The crop vocabulary is split by what the model can actually answer for. The
backend keeps its own copy in `crops.py`, and `test_crops.py` fails if it drifts
from the shipped model.

  Model crops - the model answers, `decision_source` is "model":
      tomato, rice, pepper

  Rule-only crops - the model has no indicator column for these, so the AI
  answers from the rule threshold and `decision_source` is "rule_fallback":
      chili, maize, onion, potato, soybean, wheat

Any other value is rejected at plot creation with 400. The backend never sends
"default"; that value only exists inside the AI service for direct callers who
omit the field.


Endpoints
---------

GET /api/health
    Liveness check. No parameters, no side effects.

POST /api/plots
    Register a plot and bind it to a device. One device per plot.

    Request:
        name       string, required, 1+ characters
        crop_type  string, required. Case- and whitespace-insensitive; stored
                   lowercased. Must be one of the crops listed above.
        device_id  string, required, 1+ characters, not blank after stripping

    201 Created
        {"id": 1, "name": "North Field", "crop_type": "maize",
         "device_id": "ESP32_01", "created_at": "2026-10-01T19:52:08.708271Z"}

    400  unknown crop_type. The message lists every valid crop.
    400  device_id is blank after stripping.
    422  a required field is missing or the wrong type.

    A device may be registered only once. There is a unique index on
    `plot.device_id`, so the database rejects a duplicate even if two requests
    race past the pre-check.

    The crop cannot be changed after creation; there is no PATCH or DELETE
    endpoint yet.

GET /api/plots
    All registered plots, ordered by id. Returns a list, empty if none.

POST /api/sensor-data
    The ESP32 ingestion endpoint. This is the main one.

    Request (all required unless noted):
        device_id      string, required
        soil_moisture  number
        temperature    number
        humidity       number
        light          number
        water_flow     number, optional (defaults to 0)
        pump_status    boolean, optional (defaults to false)
        timestamp      ISO-8601 string, optional (defaults to now, UTC)

        crop_type is NOT accepted here. The backend resolves it from the Plot.

    200 OK
        {
          "success": true,
          "message": "Sensor data stored successfully",
          "id": 12,
          "device_id": "ESP32_01",
          "plot_id": 1,
          "crop_type": "maize",
          "irrigation": { ...the AI prediction, or null... }
        }

        When the AI could not be reached or answered with something unusable,
        `irrigation` is null and an `irrigation_error` key is added. The
        reading itself is still stored: the row is committed before the AI is
        called, so an AI outage costs you the advice, never the measurement,
        and a retrying device does not create a duplicate.

    422  device_id is missing or blank.
    422  no Plot is registered for that device_id. The message names the device
          and says to register it first.
    422  a required reading is missing or the wrong type.

GET /api/sensor-data
    All stored readings, newest first. Takes no query parameters.

GET /api/latest-sensor-data
    The most recent reading, for the dashboard. Takes no query parameters.

    404  no readings have been stored yet.

    The response is the stored snapshot, not a live AI call, so it works while
    the AI service is down. It includes the resolved `crop_type` and the stored
    `irrigation` block:

        {
          "id": 12, "timestamp": "...", "soil_moisture": 22.0,
          "temperature": 25.0, "humidity": 60.0, "light": 20000.0,
          "light_intensity": 20000.0, "water_flow": 0.0,
          "pump_status": false, "irrigation_status": "Not Irrigating",
          "npk": {"nitrogen": null, "phosphorus": null, "potassium": null},
          "device_id": "ESP32_01", "crop_type": "maize",
          "irrigation": {"irrigation_needed": true, "prediction_score": 0.82,
                         "decision_source": "model", "reason": "..."}
        }


Other endpoints
---------------
Unchanged by the crop integration and not documented here in detail:
`/api/ai-prediction`, `/api/crop-health`, `/api/irrigation-events`,
`/api/irrigation-history`, `/api/water-usage`. See /docs for their schemas.


Tests
-----
    python -m pytest -q

Runs against temporary SQLite files, so the development database is never
touched and the AI service does not have to be running.

  test_crop_flow.py  Plot configuration and the ingestion flow, with the AI
                     call stubbed.
  test_ai_client.py  The real AI payload construction, with only httpx stubbed.
  test_ai_seam.py    The whole chain end to end, with only the network mocked.
  test_crops.py      The backend's crop vocabulary against the shipped model.
  test_cors.py       The CORS allow-list policy.
