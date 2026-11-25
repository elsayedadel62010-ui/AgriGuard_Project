# Integration guide (for Member 3 - backend and dashboard)

This guide is written so that you can integrate the AgriGuard AI service
**without reading a single line of the ML training code**. Everything you need
is here: the flow, the exact request format, the exact response format, error
handling, and a working Python example you can copy.

- Field-by-field API reference: `docs/api_contract.md`
- Feature definitions and units: `docs/dataset_structure.md`
- What the models can and cannot do: `docs/limitations.md`

---

## 0. What the AI service is (and is not)

The AI service is a small FastAPI process in this repository. It:

- accepts one sensor reading, or one image;
- runs a trained model;
- returns a **recommendation**.

It is **not**:

- a database;
- a scheduler;
- a pump controller;
- a dashboard.

You already own all of those. Do not build them again inside this service, and
do not expect it to remember anything between requests - it is stateless on
purpose, so you can restart, scale or deploy it freely.

---

## 1. Irrigation flow

```text
ESP32
  |  (JSON over Wi-Fi, every 10 minutes)
  v
Backend  ---> POST /v1/predict/irrigation  ----> AI service
  |                                            | (logistic regression)
  |  <---- {"irrigation_needed": true, ...} ---|
  |
  |  apply YOUR limits:
  |    - min minutes between waterings
  |    - max minutes per day
  |    - max consecutive runtime
  |    - flow sensor says water is moving -> skip
  |
  +---> send a command to the ESP32 relay   (only if all limits pass)
  |
  +---> Dashboard: show the alert + the reason text
```

### Step 1 - receive from the ESP32

`plot_id` is an opaque label. This service never interprets it beyond keying
per-plot history, so `"PLOT_A"` is as valid as any other string, and the examples
below use it for readability. If you are calling through the AgriGuard backend
rather than directly, the backend fills it with the numeric plot id as a string
(`"1"`) and resolves the crop from the plot it has registered; see
`AgriGuard_Backend/README.txt`.

Nothing changes about your ESP32 payload. The AI service already uses the exact
field names Member 1's firmware produces. The only thing to do is rename nothing
and forward the object as it is:

```json
{
  "timestamp": "2026-09-27T18:30:00Z",
  "device_id": "ESP32_01",
  "plot_id": "PLOT_A",
  "soil_moisture": 35,
  "temperature": 29,
  "humidity": 58,
  "light": 700,
  "npk_n": null,
  "npk_p": null,
  "npk_k": null,
  "water_flow": 0,
  "pump_status": 0,
  "crop_type": "tomato"
}
```

If your firmware sends `null` for NPK, forward `null`. If it omits the NPK keys
entirely, that is also fine - they are optional.

### Step 2 - call the AI service

```bash
curl -X POST http://localhost:8000/v1/predict/irrigation \
  -H "Content-Type: application/json" \
  -d '{
        "timestamp": "2026-09-27T18:30:00Z",
        "device_id": "ESP32_01",
        "plot_id": "PLOT_A",
        "soil_moisture": 20,
        "temperature": 33,
        "humidity": 42,
        "light": 68000,
        "npk_n": null, "npk_p": null, "npk_k": null,
        "water_flow": 0, "pump_status": 0, "crop_type": "tomato"
      }'
```

### Step 3 - read the response

```json
{
  "model": "irrigation",
  "version": "0.2.0",
  "prediction": {
    "irrigation_needed": true,
    "prediction": "Irrigation Recommended",
    "prediction_score": 0.917,
    "irrigation_duration_min": null,
    "history_used": false,
    "decision_source": "model",
    "reason": "logistic-regression score 0.92 >= 0.35; soil_moisture=20, temperature=33, humidity=42, light=68000 | NOTE: no usable history was supplied, so the three historical features are placeholders and the score is not trustworthy (see models/irrigation_metrics.json -> test_metrics_without_history) | no history supplied; soil_moisture_avg_30min is a placeholder; ...",
    "advisory_only": true
  }
}
```

What to do with each field:

| Field | What you do with it |
| ----- | ------------------- |
| `irrigation_needed` | Your decision input. `true` = the AI suggests watering. |
| `prediction` | Display string, verbatim. |
| `prediction_score` | Threshold/ordering signal only. **Never** render it as "83 % sure" - it saturates at 0.0/1.0 on this model. |
| `irrigation_duration_min` | Ignore. Always `null` by design. |
| `history_used` | `false` means the historical features were placeholders, so hide the score. `null` means the model was not consulted. |
| `decision_source` | **Check this before trusting the score.** `model` = the trained model answered. `rule_fallback_out_of_range` = a reading was outside the range the model was trained on, so the crop threshold rule decided instead; treat the score as untrustworthy and show `reason`. `pump_guard`/`water_guard` = hardware already busy. |
| `reason` | Show under the alert; it is human-readable English. |
| `advisory_only` | Ignore, but assert on it in your code review. |

#### The one rule that is about safety, not display

Member 3 owns the code that actually opens a valve, so this is the backend's
rule to implement, not the dashboard's:

> **Never act on `irrigation_needed: true` when `history_used` is `false`.**

The AI deliberately keeps answering a history-free request rather than
refusing it, so a backend that treats `irrigation_needed` as the whole decision
will water on the placeholder path. Measured on the test split, a request
without history recommends irrigation on 25.4 % of rows and only 33.3 % of
those recommendations are correct - against 8.6 % base rate. With real history
the same model is at 65.0 % precision. The `false` flag is how the service says
"this answer came from invented values", and the row it marks is exactly the row
the numbers say not to trust.

Two ways to implement it, both one line:

```python
# Preferred: refuse to act on a placeholder-backed recommendation.
if body["prediction"]["irrigation_needed"] and body["prediction"]["history_used"] is False:
    log("suppressed: irrigation advised without history")
    return  # fall through to your own moisture policy instead

# Equivalent, and easier to reason about: never send history-free requests.
# Build `history` from your readings table and the flag is always true.
```

The second option is better if you can do it. The service already has
`last_irrigation_at` and the recent moisture readings it needs; `history_used`
becomes `false` only when the backend cannot supply them, which is also the
moment the backend most likely has stale data and a broken context and should
be suspicious of *every* reading from that plot.

This is a limitation of the model, not a bug you can patch downstream: the
placeholder is a "nothing recent" value, and 24 h of nothing recent genuinely
does look like a dry plot.

### Step 3a - send the history, and the score gets much better

The request above sends no `history`, so all three historical features are
placeholders and the model is guessing. Add a `history` block and the same plot
gets a far more trustworthy answer:

```bash
curl -X POST http://localhost:8000/v1/predict/irrigation \
  -H "Content-Type: application/json" \
  -d '{
        "timestamp": "2026-09-27T18:30:00Z",
        "device_id": "ESP32_01",
        "plot_id": "PLOT_A",
        "soil_moisture": 20,
        "temperature": 33, "humidity": 42, "light": 68000,
        "water_flow": 0, "pump_status": 0, "crop_type": "tomato",
        "history": {
          "recent_soil_moisture": [
            {"timestamp": "2026-09-27T18:00:00Z", "soil_moisture": 24},
            {"timestamp": "2026-09-27T18:15:00Z", "soil_moisture": 22}
          ],
          "last_irrigation_at": "2026-09-27T10:00:00Z"
        }
      }'
```

`history_used` comes back `true` and the score is the model's real opinion. The
measured difference is large: on the test split, precision rises from 0.33 to
0.65 and F1 from 0.50 to 0.66. Send the history whenever you have it.

`last_irrigation_at` is the time of the last observed pump run
(`pump_status=1`). It needs your request `timestamp` to be turned into
"minutes since" - send both, or the field is dropped and `reason` says so.

### Step 4 (strongly recommended) - send history

The model has three historical features and they matter. Keep the last few
readings per plot in your database (you already store them) and forward them. The
request in step 3a is a working example; the rules are:

- `recent_soil_moisture`: the previous 3-6 readings of the **same**
  `device_id`/`plot_id`, oldest first. The moving-average window is 30 minutes,
  and only readings inside it are used.
- `last_irrigation_at`: when the pump was last seen running
  (`pump_status == 1` or `water_flow > 0`).

If you omit this the endpoint still returns 200, but the model falls back to
placeholders. The measured cost of that is large - see
`test_metrics_without_history` in `models/irrigation_metrics.json`.

### Step 5 - handle `decision_source` before you trust a score

Four readings reach the model, and all four have to be inside the range it was
trained on for its answer to mean anything. If one is not, the service refuses to
report the model's opinion and the per-crop threshold rule answers instead:

```json
{
  "prediction": {
    "irrigation_needed": true,
    "prediction_score": 0.53,
    "history_used": false,
    "decision_source": "rule_fallback_out_of_range",
    "reason": "OUT OF TRAINING RANGE: temperature=-10 (trained 12.83..35.22), humidity=0 (trained 38.49..97.05), light=200000 (trained 0..84562.1). The model extrapolates outside the readings it was fitted on; it would have scored 0.08 against a threshold of 0.35, and that is reported here for diagnosis only. prediction_score (0.53) and the decision both come from the crop threshold rule. | ..."
  }
}
```

Treat this as "the AI does not know here, so use the conservative default". Do not
render `prediction_score` as a confidence in this case: the model is
extrapolating in a direction it was never fitted, and the fitted `light`
coefficient is negative, so a bright reading it has never seen pushes the score
*down*. The example above is soil moisture 16 % on a tomato plot - well past the
20.8 % trigger - and the model scored it 0.08, i.e. "do not irrigate". That is
the failure this guard exists to stop, and it is why the rule, not the model,
gets the last word out here.

The complete set of `decision_source` values, and the recorded training ranges
they are checked against, are in `docs/api_contract.md` and
`models/irrigation_model.json -> training_feature_ranges`.

### Step 5 - act (or do not)

The service is advisory. Before you send a command to a relay:

```text
1. if prediction.advisory_only is not true            -> stop, do not act
2. if pump_status == 1 or water_flow > 0              -> stop, already watering
3. if now - last_irrigation < YOUR_MIN_INTERVAL        -> stop
4. if minutes_watered_today + YOUR_MAX_DAILY > LIMIT  -> stop
5. if consecutive_runtime + YOUR_MAX_RUN > MAX_RUN     -> stop
6. otherwise send your own runtime decision to the ESP32
```

Steps 2-5 are **your** policy. This project deliberately ships no duration model
so that the water policy stays with the team that owns the hardware.

---

## 2. Crop-health flow

```text
ESP32-CAM
  |  (JPEG, e.g. every 30-60 minutes or on demand)
  v
Backend  ---> POST /v1/predict/crop-health  ---> AI service
  |          (multipart/form-data, field "image")  |  (MobileNetV3-Small)
  |  <---- {"class": "possible_disease", ...} ----|
  |
  +---> Dashboard: amber/red "inspect this plant" alert + the image
```

### Step 1 - get the image off the ESP32-CAM

```bash
curl -X POST http://192.168.1.50/capture -o capture.jpg
```

The AI service accepts JPEG, PNG, BMP, WEBP and TIFF up to 10 MB. Do **not**
resize on the ESP32 - send the original and let the service handle it, so the
training-time and inference-time preprocessing stay identical.

### Step 2 - forward it

```bash
curl -X POST http://localhost:8000/v1/predict/crop-health \
  -F "image=@capture.jpg"
```

In Python:

```python
with open("capture.jpg", "rb") as fh:
    files = {"image": ("capture.jpg", fh.read(), "image/jpeg")}
response = httpx.post(f"{AI_URL}/v1/predict/crop-health", files=files, timeout=60)
body = response.json()
```

The multipart field name must be exactly `image`.

### Step 3 - read the response

```json
{
  "model": "crop-health",
  "version": "0.2.0",
  "prediction": {
    "class": "possible_disease",
    "label": "Possible Disease",
    "confidence": 0.87,
    "method": "MobileNetV3-Small",
    "model_version": "0.2.0"
  }
}
```

| Field | What you do with it |
| ----- | ------------------- |
| `class` | `healthy` or `possible_disease` - use this as the machine key |
| `label` | `Healthy` or `Possible Disease` - show this to the user |
| `confidence` | relative score, not a guarantee. Show as a percentage if you like, but label it a score. |
| `method` | log it; it tells you which model answered |
| `model_version` | store it next to the image so old results stay explainable |

### Step 4 - present it honestly

```text
The model classifies images into Healthy or Possible Disease within the scope
of the training dataset. It is a prototype screening/classification model and is
not a universal disease diagnostic system.
```

A `Possible Disease` result should trigger a human inspection prompt, not an
automatic spray order. The model was trained on laboratory leaf photographs, so
its behaviour on a real plot is unknown - see `docs/limitations.md`.

---

## 3. Complete backend example (FastAPI, copy-paste ready)

```python
# backend/ai_client.py
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import httpx

AI_URL = os.getenv("AGRIGUARD_AI_URL", "http://127.0.0.1:8000")
TIMEOUT = httpx.Timeout(30.0, connect=5.0)

# ---- the water policy lives HERE, not in the AI service -------------------
MIN_MINUTES_BETWEEN_WATERINGS = 60
MAX_MINUTES_PER_DAY = 45
MAX_MINUTES_PER_RUN = 15


class AgriGuardAI:
    def __init__(self, base_url: str = AI_URL) -> None:
        self._client = httpx.Client(base_url=base_url, timeout=TIMEOUT)
        self._last_watering: Dict[str, float] = {}
        self._minutes_today: Dict[str, float] = {}

    def health(self) -> Dict[str, Any]:
        response = self._client.get("/v1/health")
        response.raise_for_status()
        return response.json()

    def ready(self) -> bool:
        """Both models loaded? Do not show a result if not."""
        try:
            return self.health()["status"] == "ok"
        except Exception:
            return False

    def predict_irrigation(
        self,
        reading: Dict[str, Any],
        recent_soil_moisture: Optional[List[Dict[str, Any]]] = None,
        last_irrigation_at: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Returns the AI prediction, or None if the service is unreachable."""
        payload = dict(reading)
        if recent_soil_moisture or last_irrigation_at:
            payload["history"] = {
                "recent_soil_moisture": recent_soil_moisture or [],
                "last_irrigation_at": last_irrigation_at,
            }
        try:
            response = self._client.post("/v1/predict/irrigation", json=payload)
        except httpx.HTTPError:
            return None
        if response.status_code != 200:
            return None
        return response.json()

    def should_water(
        self,
        prediction: Dict[str, Any],
        plot_id: str,
        runtime_minutes: int,
    ) -> Dict[str, Any]:
        """Apply the team's own safety policy to an AI recommendation."""
        body = prediction.get("prediction", {})
        verdict = {
            "irrigation_needed": bool(body.get("irrigation_needed")),
            "run_pump": False,
            "runtime_minutes": 0,
            "reason": body.get("reason", ""),
        }
        if not body.get("advisory_only", True):
            verdict["reason"] = "AI response was not advisory; refusing to act."
            return verdict
        if not verdict["irrigation_needed"]:
            return verdict

        now = time.time()
        last = self._last_watering.get(plot_id)
        if last is not None and (now - last) / 60 < MIN_MINUTES_BETWEEN_WATERINGS:
            verdict["reason"] += " | blocked: too soon after the last watering"
            return verdict
        if self._minutes_today.get(plot_id, 0.0) + runtime_minutes > MAX_MINUTES_PER_DAY:
            verdict["reason"] += " | blocked: daily watering cap reached"
            return verdict
        if runtime_minutes > MAX_MINUTES_PER_RUN:
            verdict["reason"] += " | blocked: runtime above the per-run limit"
            return verdict

        verdict["run_pump"] = True
        verdict["runtime_minutes"] = min(runtime_minutes, MAX_MINUTES_PER_RUN)
        self._last_watering[plot_id] = now
        self._minutes_today[plot_id] = (
            self._minutes_today.get(plot_id, 0.0) + verdict["runtime_minutes"]
        )
        return verdict

    def predict_crop_health(self, image_bytes: bytes, filename: str) -> Optional[Dict[str, Any]]:
        files = {"image": (filename, image_bytes, "image/jpeg")}
        try:
            response = self._client.post("/v1/predict/crop-health", files=files)
        except httpx.HTTPError:
            return None
        if response.status_code != 200:
            return None
        return response.json()
```

Usage inside your existing endpoint:

```python
ai = AgriGuardAI()

@router.post("/api/readings")
def ingest(reading: dict, db=Depends(get_db)):
    save_reading(db, reading)                       # your table, your job
    result = ai.predict_irrigation(
        reading,
        recent_soil_moisture=last_readings(db, reading["plot_id"], limit=6),
        last_irrigation_at=last_pump_run(db, reading["plot_id"]),
    )
    if result is None:
        return {"status": "ai_unavailable", "watered": False}

    body = result["prediction"]
    verdict = ai.should_water(result, reading["plot_id"], runtime_minutes=10)
    if verdict["run_pump"]:
        send_relay_command(reading["device_id"], verdict["runtime_minutes"])
    return {
        "status": "ok",
        "watered": verdict["run_pump"],
        "needed": body["irrigation_needed"],
        # only surface a score when the historical features were real
        "score": body["prediction_score"] if body["history_used"] else None,
        "history_used": body["history_used"],
        "reason": body["reason"],
        **verdict,
    }
```

```python
@router.post("/api/capture")
def capture(device_id: str, plot_id: str, db=Depends(get_db)):
    image = download_from_esp32_cam(device_id)       # your ESP32-CAM call
    store_image(db, plot_id, image)                  # keep the evidence
    result = ai.predict_crop_health(image, f"{device_id}.jpg")
    if result is None:
        return {"status": "ai_unavailable"}

    body = result["prediction"]
    if body["label"] == "Possible Disease":
        create_alert(db, plot_id=plot_id, kind="possible_disease",
                     image_id=..., score=body["confidence"])
    return {"status": "ok", "label": body["label"],
            "score": body["confidence"], "model": body["method"]}
```

---

## 4. Operating notes

| Topic | Recommendation |
| ----- | -------------- |
| Timeouts | 5 s connect, 30 s read. Crop-health inference is a single 224x224 forward pass and is fast. |
| Retries | Do not retry a `422` or a `503` immediately. Retry once on a network error only. |
| Caching | Do not cache irrigation predictions - they depend on live sensor values. You may cache a crop-health result for the same stored image id. |
| Concurrency | The service is stateless and thread-safe enough for a small team demo. Scale with uvicorn workers if needed: `uvicorn agriguard.api:app --workers 2`. Note that PyTorch is used inside the process, so keep the worker count small on a laptop. |
| Health monitoring | Poll `/v1/health`. Alert when `status != "ok"`; the dashboard should show "AI model unavailable" rather than a stale number. |
| Model updates | When the team retrains, `version` and `model_version` change. Log them with every stored prediction so old results remain explainable. |
| CORS | The service adds a CORS middleware when `AGRIGUARD_CORS_ORIGINS` is set, and no CORS headers at all when it is not. Set it to the dashboard's exact origins, comma-separated: `AGRIGUARD_CORS_ORIGINS="http://localhost:5173"`. Leave it unset when a reverse proxy serves both from one origin. The allow-list is never `*`, so a mistyped origin fails closed. |

---

## 5. Troubleshooting

| Symptom | Cause | Fix |
| ------- | ----- | --- |
| `422` on irrigation with `"field": "humidity"` | ESP32 sent a value outside 0-100 | Fix the firmware or clamp in the backend before forwarding |
| `422` on crop-health, `"invalid image"` | The ESP32-CAM returned an error page or a truncated JPEG, not an image | Check the camera HTTP response; log the first bytes |
| `415` on crop-health | Wrong `Content-Type` | Use `-F "image=@file.jpg"`; the field name must be `image` |
| `503` on crop-health | `models/crop_health_model.pth` is missing | Run `python scripts/train_crophealth.py` on that host |
| `/v1/health` shows `"status": "degraded"` | One of the two model files is missing or corrupt | Read `models.<name>.load_error` - it says exactly why |
| Predictions look identical for two plots | You are not sending `device_id`/`plot_id`, so the AI cannot separate the histories | Add both fields |
| `prediction_score` is always about 0.5 | You are sending a reading that is out of the training distribution, or the model is the rule fallback | Check `GET /v1/health` -> `models.irrigation.trained` |
