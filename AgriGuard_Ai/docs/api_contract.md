# API contract for backend integration (for Member 3)

Version `0.2.0`. The AI layer is a separate HTTP service. It receives readings
and returns a **recommendation**. It stores nothing, owns no database and never
switches hardware.

```text
Base URL: http://<ai-host>:8000
```

Interactive documentation (Swagger UI) is served at `/docs`.

## Endpoints at a glance

| Method | Path | Body | Returns |
| ------ | ---- | ---- | ------- |
| `GET`  | `/` | - | service name, version, endpoint list |
| `GET`  | `/v1/health` | - | service status and per-model load status |
| `GET`  | `/v1/classes/crop-health` | - | the two class keys, labels and dashboard hints |
| `POST` | `/v1/predict/irrigation` | JSON | irrigation recommendation |
| `POST` | `/v1/predict/crop-health` | `multipart/form-data` | `Healthy` or `Possible Disease` |

## Changes in 0.2.0 (read before upgrading Member 3's code)

| In 0.1.0 | In 0.2.0 | Why |
| -------- | --------- | --- |
| `prediction.confidence` | `prediction.prediction_score` | The number is a logistic-regression score, not a calibrated probability. Renaming it stops the dashboard from presenting it as "91 % sure". |
| `irrigation_duration_min: <number>` | `irrigation_duration_min: null` | There is no defensible irrigation-duration model, so the service no longer guesses one. |
| response echoed the whole request under `features` | removed | It leaked NPK and plot metadata back to every caller for no benefit. |
| no `advisory_only` flag | `prediction.advisory_only: true` | Makes it impossible to mistake this service for something allowed to drive a pump. |
| `device_id` / `plot_id` not accepted | accepted and used | Needed to keep per-plot history separate. |
| no `history` block | optional `history` block | Lets the backend supply the sequential readings the historical features need. |
| no `history_used` flag | `prediction.history_used` | A score computed from placeholder features is not trustworthy, and the dashboard has to be able to tell that difference. |
| no `decision_source` flag | `prediction.decision_source` | The service can decline to use the model when a reading is outside the range it was fitted on. Without this field a client could not tell a real prediction from a fallback and would have displayed the fallback's score as the model's. |

---

## 1. Irrigation prediction

### Request

```http
POST /v1/predict/irrigation
Content-Type: application/json
```

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

Only four fields are mandatory: `soil_moisture`, `temperature`, `humidity`,
`light`. Everything else is optional and defaults sensibly.

| Field | Type | Required | Range | Notes |
| ----- | ---- | -------- | ----- | ----- |
| `timestamp` | string | no | ISO-8601 | Used to place the reading on the plot's timeline |
| `device_id` | string | no | - | e.g. `ESP32_01` |
| `plot_id` | string | no | - | e.g. `PLOT_A` |
| `soil_moisture` | float | **yes** | 0 - 100 | sensor scale |
| `temperature` | float | **yes** | -40 - 80 | Celsius |
| `humidity` | float | **yes** | 0 - 100 | % |
| `light` | float | **yes** | 0 - 200000 | whatever the sensor reports |
| `npk_n`, `npk_p`, `npk_k` | float or null | no | >= 0 | `null` when no NPK probe is fitted |
| `water_flow` | float | no | 0 - 100000 | 0 while the pump is off |
| `pump_status` | int | no | 0 or 1 | 0 = off, 1 = on |
| `crop_type` | string | no | `default` | plot configuration, not a sensor. Only `tomato`, `rice` and `pepper` are scored by the model |
| `history` | object | no | - | see below |

#### Optional `history` block

The model uses three historical features. Sending this block lets the service
compute them from your real readings instead of falling back to placeholders.

```json
{
  "history": {
    "recent_soil_moisture": [
      { "timestamp": "2026-09-27T18:10:00Z", "soil_moisture": 36.4 },
      { "timestamp": "2026-09-27T18:20:00Z", "soil_moisture": 35.6 }
    ],
    "last_irrigation_at": "2026-09-27T15:00:00Z"
  }
}
```

Keep the last 3-6 readings of the same `device_id`/`plot_id`, oldest first. The
moving-average window is 30 minutes. **If you omit this block the endpoint still
returns 200** - the service substitutes placeholders, reports
`"history_used": false` and says so in `prediction.reason`. The measured cost of
that is in `models/irrigation_metrics.json` under
`test_metrics_without_history`; it is significant, so a real integration should
send history.

> **Send `timestamp` as well.** `last_irrigation_at` is an absolute time, so the
> service has to subtract it from something. The request's own `timestamp` is
> that reference. If you send `history` with a `last_irrigation_at` but no
> top-level `timestamp`, that one field is **dropped**, `time_since_last_irrigation_min`
> falls back to a placeholder, and `history_used` comes back `false` - the
> response says exactly that in `prediction.reason` rather than ignoring your
> value silently. The `recent_soil_moisture` entries are used either way, since
> each carries its own timestamp. In short: **omit `last_irrigation_at`, or send
> `timestamp` with it.**

### Response

```json
{
  "model": "irrigation",
  "version": "0.2.0",
  "prediction": {
    "irrigation_needed": true,
    "prediction": "Irrigation Recommended",
    "prediction_score": 0.624,
    "decision_source": "model",
    "irrigation_duration_min": null,
    "history_used": true,
    "reason": "logistic-regression score 0.62 >= 0.35; soil_moisture=20, temperature=33, humidity=42, light=68000",
    "advisory_only": true
  }
}
```

| Field | Type | Meaning for the dashboard |
| ----- | ---- | ------------------------- |
| `irrigation_needed` | bool | `true` = the AI recommends watering. Show the alert. |
| `prediction` | string | `"Irrigation Recommended"` or `"Irrigation Not Needed"`. Show verbatim. |
| `prediction_score` | float 0-1 | The strength of the evidence for "irrigate", on the same scale on every path. `irrigation_needed` is exactly `prediction_score >= decision_threshold` for **all** of them, including the guards and the fallbacks. Never render it as "85 % sure the crop needs water" - it is not a calibrated probability. If you badge it, label it "model score". |
| `decision_source` | string | What produced the advice. Read it to know how much the number is worth; you do **not** need it to re-derive the decision. One of `model`, `pump_guard`, `water_guard`, `rule_fallback`, `rule_fallback_out_of_range`. |
| `irrigation_duration_min` | null | Always `null`. The service has no duration model. Any runtime policy belongs in the backend. |
| `history_used` | bool / null | `true` = the three historical features came from real readings. `false` = they were placeholders, so **the score is not trustworthy** - hide the number and just show the recommendation. `null` = the model was not consulted at all (safety guard, or no model file), so there is no model score to show. |
| `reason` | string | English explanation, safe to display under the alert. |
| `advisory_only` | bool | Always `true`. |

### `decision_source` values

| Value | When | What to trust |
| ----- | ---- | ------------- |
| `model` | The trained logistic regression answered and every client-supplied reading was inside the range it was fitted on. | The score, subject to `history_used`. |
| `pump_guard` | `pump_status == 1`. | Nothing - the answer is a certainty, not a judgement. `prediction_score` is 0.0. |
| `water_guard` | `water_flow > 0`. | As above. |
| `rule_fallback` | `models/irrigation_model.json` is missing or unreadable. | The crop threshold rule. `history_used` is `null`. |
| `rule_fallback_out_of_range` | A client-supplied reading (`soil_moisture`, `temperature`, `humidity` or `light`) lies outside the recorded training range. | The crop threshold rule. `prediction_score` is the **rule's** score on the model's scale, not the model's; the model's own extrapolated score is quoted in `reason` for diagnosis only, and must not be displayed. |

The recorded bounds are in `models/irrigation_model.json ->
training_feature_ranges` and are also printed by `GET /v1/health`, so you can
read the current values instead of copying them:

| Reading | Recorded training range |
| ------- | ----------------------- |
| `soil_moisture` | 13.76 .. 77.31 |
| `temperature` | 12.83 .. 35.22 C |
| `humidity` | 38.49 .. 97.05 % |
| `light` | 0 .. 84 562 lux |

The out-of-range case is not hypothetical. The fitted `light` coefficient is
negative - in synthetic data the label marks the instant a threshold scheduler
fired, so bright readings correlate with "already watered" - so a linear model
extrapolates a brighter or colder reading hard in the wrong direction. A tomato
plot at 16 % soil moisture, well past its trigger, was told "do not irrigate" at
-10 C / 0 % RH / 200 000 lux. See
[`limitations.md`](limitations.md) for the measurement and the exact bounds.

**The guard is deliberately per-reading and client-supplied only.** The three
derived historical features are not range-checked, because
`soil_moisture_change` moves whenever the current reading moves against a fixed
history, and range-checking it made the documented trigger collapse to the crop's
reference threshold regardless of what the model thought. That history is the
backend's to supply and the backend owns its validity; see
[`limitations.md`](limitations.md).

### The decision threshold is 0.35, not 0.50

`irrigation_needed` is `prediction_score >= decision_threshold`, where the
threshold is stored in `models/irrigation_model.json` (currently **0.35**) and
also reported by `GET /v1/health`. It was chosen as the F1-optimal cut-off on a
held-out slice of the *training* timeline; the test set was never used to pick
it.

The backend must **not** re-derive the decision with its own `0.5` cut-off. Use
the `irrigation_needed` boolean that the service already computed. Re-thresholding
the raw score yourself at 0.5 changes the measured behaviour from recall 0.66 /
precision 0.65 to recall 0.42 / precision 0.81.

### All four readings reach the model

The model uses `soil_moisture`, `temperature`, `humidity` and `light`, the three
derived historical features, and the crop indicators - ten inputs. All four
readings are required and validated on the request, and all four are model
inputs in v2.0.0. A hot, dry, bright reading does move the recommendation, and
not always upwards: see
[`limitations.md`](limitations.md) for the measured sensitivity and the
`decision_source` guard that exists because of it.

`crop_type` is a real input, and the model is only trained on `tomato`, `rice`
and `pepper`. The trigger level is crop specific, so a request naming any other
crop is **not** scored by the model: it is answered by the rule-based threshold
and `decision_source` is `rule_fallback`, with the reason naming the crops the
model does cover. Check `decision_source` before acting on the recommendation -
`model` means the prediction is in-domain, anything else does not.

### Re-deriving the decision from `prediction_score`

`prediction_score >= decision_threshold` reproduces `irrigation_needed` exactly
on **every** path - the model, both safety guards, and both fallbacks. This used
not to be true, in two separate ways, and both are regression-tested:

- The "Irrigation Not Needed" branch returned `1 - score`, so a soaked plot came
  back as `prediction_score: 1.0` beside `irrigation_needed: false`. A backend
  following the documented check would have watered it.
- The safety guards reported 0.75 beside "do not water", which would have started
  a second pump while one was already running. They report 0.0 instead.
- The rule fallback scored *both* of its branches between 0.5 and 0.95, so the
  check was true even while reporting "do not water". That was invisible while
  the fallback was only reachable with no model file on disk; it became a live
  contract violation once out-of-range requests started routing through it. The
  fallback now produces a monotone score that crosses `decision_threshold`
  exactly at the crop's reference moisture.

There is a test that sweeps both outcomes across every branch and asserts the
identity, and another that asserts the number quoted in `reason` and the number
in `prediction_score` agree.

The one thing to be careful about is *interpretation*, not arithmetic: in the
`rule_fallback_out_of_range` case `prediction_score` is the threshold rule's
score, and the model's own extrapolated score is in `reason` for diagnosis only.
Do not display that number. `decision_source` tells you which is which.

The simplest correct integration is to read `irrigation_needed` and never
threshold the score yourself. The threshold rule above exists so that you *can*
re-derive the decision if you need to.

### The score saturates - read this before showing a number

`prediction_score` is a logistic-regression score, and the synthetic training
data is close to linearly separable, so the fitted weights are large and the
sigmoid saturates. A dry reading can come back as `1.0` and a wet one as `0.0`.
That is not a calibrated 100 % confidence; it means "far past the threshold".

Practical rule for the dashboard: use `prediction_score` to cross the threshold
and to order two predictions, not as a percentage. If you want a percentage
anyway, clamp it away from the extremes or display it as
"model score: high / medium / low". And if `history_used` is `false`, do not
display a score at all - the measured accuracy in that mode is in
`models/irrigation_metrics.json` -> `test_metrics_without_history`, and it is
poor.

### Safety behaviour you can rely on

- If `pump_status == 1` the response is **always** `Irrigation Not Needed`.
- If `water_flow > 0` the response is **always** `Irrigation Not Needed`.

This makes it impossible to talk the backend into a second irrigation while the
pump is already running. The service still returns 200 - it is a recommendation
that happens to be negative, not an error.

- **`history_used: false` means do not act on it.** The two rules above are
  enforced by the service, so a caller cannot talk the backend into a second
  irrigation. This one is not: the service still returns
  `irrigation_needed: true` on the placeholder path, so *the caller* has to
  refuse to act on it. Measured, a history-free request recommends irrigation
  on 25.4 % of test rows at 33.3 % precision, versus 65.0 % with history.

```python
prediction = body["prediction"]
if prediction["irrigation_needed"] and prediction["history_used"] is False:
    ...  # do not open a valve; send `history` next time
```

See `docs/integration_guide.md` for the full reasoning and the alternative of
never sending history-free requests at all.


### Error codes

| Code | When | Body |
| ---- | ---- | ---- |
| `422` | missing required field, out-of-range number, wrong JSON type | `{"detail": "request validation failed", "errors": [{"field": "...", "message": "..."}]}` |
| `500` | unexpected internal error | `{"detail": "internal AI service error"}` - no stack trace, ever |

---

## 2. Crop-health classification

### Request

```http
POST /v1/predict/crop-health
Content-Type: multipart/form-data
```

The image must be in a field literally named `image`. JPEG, PNG, BMP, WEBP and
TIFF are accepted, up to 10 MB. The service decodes and resizes the image
itself - do not resize on the ESP32.

```bash
curl -X POST http://localhost:8000/v1/predict/crop-health \
  -F "image=@esp32_capture.jpg"
```

### Response

```json
{
  "model": "crop-health",
  "version": "0.2.0",
  "prediction": {
    "class": "healthy",
    "label": "Healthy",
    "confidence": 0.91,
    "method": "MobileNetV3-Small",
    "model_version": "0.2.0"
  }
}
```

| Field | Type | Meaning |
| ----- | ---- | ------- |
| `class` | string | internal key: `healthy` or `possible_disease` |
| `label` | string | `Healthy` or `Possible Disease` - show this |
| `confidence` | float 0-1 | uncalibrated softmax score. A relative score, not a guarantee. |
| `method` | string | which model answered, e.g. `MobileNetV3-Small` |
| `model_version` | string | version of the crop-health weights |

### `class` values and dashboard mapping

| `class` | `label` | Dashboard |
| ------- | ------- | --------- |
| `healthy` | `Healthy` | green "no action" card |
| `possible_disease` | `Possible Disease` | amber/red "inspect this plant" alert |

`GET /v1/classes/crop-health` returns this list as JSON so the dashboard legend
never drifts out of sync.

### What this model does not do

```text
The model classifies images into Healthy or Possible Disease within the scope
of the training dataset. It is a prototype screening/classification model and is
not a universal disease diagnostic system.
```

It does not name a disease, it does not measure severity, and it was trained on
laboratory leaf photographs. A `Possible Disease` result means **go and look at
the plant** - it does not mean "apply fungicide X".

### Error codes

| Code | When | Body |
| ---- | ---- | ---- |
| `400` | the uploaded file is empty | `{"detail": "empty upload"}` |
| `413` | the file exceeds 10 MB | `{"detail": "image larger than 10 MB"}` |
| `415` | `Content-Type` is not `image/*` | `{"detail": "unsupported content type ..."}` |
| `422` | the field `image` is missing, or the bytes are not a decodable image | `{"detail": "invalid image: ..."}` |
| `503` | the trained weights are not installed or failed to load | `{"detail": "crop-health model unavailable: ..."}` |
| `500` | unexpected internal error | `{"detail": "crop-health inference error: <ExceptionType>"}` |

Note the ordering: a broken image returns `422` even when the model is missing,
because the upload is validated before the model is consulted. `503` therefore
means "your image is fine, our server is not ready".

---

## 3. Service check

```http
GET /v1/health
```

```json
{
  "status": "ok",
  "version": "0.2.0",
  "advisory_only": true,
  "models": {
    "irrigation": {
      "trained": true,
      "model_version": "0.2.0",
      "features": ["soil_moisture", "soil_moisture_change",
                   "soil_moisture_avg_30min",
                   "time_since_last_irrigation_min",
                   "crop_is_tomato", "crop_is_rice", "crop_is_pepper"],
      "decision_threshold": 0.35,
      "trained_at": "...",
      "training_dataset": "irrigation_data.csv",
      "load_error": null
    },
    "crop_health": {
      "trained": true,
      "method": "MobileNetV3-Small",
      "model_version": "0.2.0",
      "classes": ["healthy", "possible_disease"],
      "test_metrics": { },
      "load_error": null
    }
  }
}
```

`status` is `ok` only when **both** models loaded. It is `degraded` when either
fell back to rules or is missing - monitor this, and do not show a crop-health
result if `crop_health.trained` is `false`.

`load_error` is `null` on a healthy service and carries a short human-readable
reason otherwise.

---

## Integration flows

```text
Irrigation
ESP32 sensor node -> Backend (log + forward JSON) -> POST /v1/predict/irrigation
   -> { irrigation_needed, prediction, prediction_score, reason }
   -> Backend applies its own min-interval and max-daily-runtime limits
   -> Backend -> Dashboard (alert card)

Crop health
ESP32-CAM -> Backend (store + forward multipart) -> POST /v1/predict/crop-health
   -> { class, label, confidence, method }
   -> Backend -> Dashboard alert
```

Step-by-step instructions, including a working backend example, are in
`docs/integration_guide.md`.

## The one rule to share with the team

**The AI layer recommends; the backend and the IoT layer decide.** Nothing in
this service switches a pump, opens a valve or triggers a spray. Before the
backend acts on a recommendation it must apply its own limits - minimum interval
between waterings, maximum daily runtime, and a physical flow/current cutoff on
the ESP32 itself. `advisory_only` is in the response so this is impossible to
miss in code review.


