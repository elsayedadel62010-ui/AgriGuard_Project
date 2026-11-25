# AgriGuard AI

AI/Data Science components for the AgriGuard smart-agriculture project
(Stage 3): **irrigation prediction** and **crop-health image classification**,
exposed as a small FastAPI service so Member 3's backend and dashboard can
consume them.

> **Status: working prototype.** The irrigation model is trained on clearly
> labelled **synthetic** data. The crop-health model is trained on real but
> **laboratory** images (PlantVillage). Neither number is a field-accuracy
> claim. Read [`docs/limitations.md`](docs/limitations.md) before you quote
> anything from this repository.

## What this service is, and is not

```text
The AI layer is advisory: it produces recommendations.
It is not a direct control layer.
```

Nothing here switches a pump, opens a valve or triggers a spray. Every
irrigation response carries `"advisory_only": true`, and the service refuses to
recommend a second watering whenever `pump_status == 1` or `water_flow > 0`.
The backend and the ESP32 keep the authority over hardware.

## Team integration (do NOT build a separate system)

```text
ESP32 sensor node -> Backend -> POST /v1/predict/irrigation -> Dashboard
ESP32-CAM image   -> Backend -> POST /v1/predict/crop-health  -> Dashboard alert
```

- Member 1 (ESP32) produces the field values defined in `agriguard/contracts.py`.
- Member 3 (backend/dashboard) calls the two POST endpoints.
- Start here for the integration: [`docs/integration_guide.md`](docs/integration_guide.md)
- Field-by-field API reference: [`docs/api_contract.md`](docs/api_contract.md)

## Measured results

Every number below is produced by a script in this repository and stored in a
JSON file. Nothing is estimated or typed in by hand.

### Irrigation - logistic regression, chronological 80/20 split

The model consumes all four live sensor readings plus three derived historical
features and three crop indicators - ten inputs in all.

| | With history | No history (neutral defaults) |
| --- | --- | --- |
| Accuracy | 0.9406 | 0.8287 |
| Balanced accuracy | **0.8144** | 0.9002 |
| Precision (irrigate) | 0.6504 | 0.3318 |
| Recall (irrigate) | 0.6622 | 0.9865 |
| F1 | 0.6562 | 0.4966 |
| Majority baseline | 0.9144 | 0.9144 |

12,960 rows, 1,109 irrigation events (8.6 %). Train `2026-03-01`..`2026-03-24`,
test `2026-03-25`..`2026-03-30`, 222 positives in the test window, confusion
matrix TN=2291 FP=79 FN=75 TP=147.

Two things about these numbers are worth stating plainly.

First, they are much lower than an earlier revision reported, and that is the
point. The first version of the synthetic generator reported `pump_status=1` on
the same row as `irrigation_event=1`, which made
`time_since_last_irrigation_min == 0` a perfect giveaway for the positive class
and produced a headline 0.93 balanced accuracy with no real skill behind it. The
generator now reports the pump only from the *next* reading onwards, which is
both physically correct and leak-free. See
[`docs/limitations.md`](docs/limitations.md).

Second, adding `temperature`, `humidity` and `light` to the feature set - which
the brief requires, and which the model now does - buys most of this accuracy,
but it also makes the model extrapolate. The fitted `light` coefficient came out
**negative**: in synthetic data the label marks the instant a threshold
scheduler fired, so bright readings are correlated with "already watered". Hand
the model a brighter or colder reading than any in the training set and it will
extrapolate that trend into a confident "do not irrigate" on dry soil. So the
service refuses to use the model outside the recorded training range and falls
back to the crop threshold rule, reporting which one answered in
`decision_source`. The measured failure and the exact ranges are in
[`docs/limitations.md`](docs/limitations.md).

`prediction_score` is compared against a stored operating point
(`decision_threshold`, 0.35), not against 0.50. The threshold is tuned on a
held-out slice of the *training* timeline; the test set was not used to pick it.
The identity `irrigation_needed == (prediction_score >= decision_threshold)`
holds on every code path, including the fallbacks, so a backend can re-derive
the decision from the score alone.

The second column is the honest answer to "what happens if the backend sends one
reading and no history?" - without history the model has no idea and recommends
water for almost everything, which is why the response reports
`history_used: false` in that case. Full detail:
[`docs/model_results_and_tests.md`](docs/model_results_and_tests.md).

### Crop health - MobileNetV3-Small, 70/15/15 split

| | Test set (846 unseen images) |
| --- | --- |
| Accuracy | 0.9657 |
| Balanced accuracy | **0.9668** |
| Macro F1 | 0.9606 |
| Healthy: precision / recall / F1 | 0.9242 / 0.9697 / 0.9464 |
| Possible Disease: precision / recall / F1 | 0.9859 / 0.9639 / 0.9748 |
| Majority-class baseline | 0.6879 |
| Colour-heuristic baseline (same images) | 0.4208 |

5,639 PlantVillage images (1,739 healthy / 3,900 possible disease) collapsed
from 38 original classes into two; confusion matrix rows = true, cols =
predicted `[Healthy, Possible Disease]`: `[[256, 8], [21, 561]]`. The saved
weights come from the best validation epoch (epoch 3, 0.9823), not the last one.
See [`docs/crop_health_results.md`](docs/crop_health_results.md) and
`models/crop_health_model.json` for the measured test metrics, and
[`docs/limitations.md`](docs/limitations.md) for why that number is **not** an
estimate of performance on ESP32-CAM field photos.

## Quickstart

```bash
pip install -r requirements.txt
uvicorn agriguard.api:app --reload
```

Interactive docs (Swagger UI): http://127.0.0.1:8000/docs

If the dashboard is served from a **different origin** than the API, list it
before starting the service or the browser will block every call:

```bash
# Windows PowerShell
$env:AGRIGUARD_CORS_ORIGINS="http://localhost:5173,https://dash.example.com"
uvicorn agriguard.api:app --reload
```

```bash
# bash
AGRIGUARD_CORS_ORIGINS="http://localhost:5173,https://dash.example.com" \
  uvicorn agriguard.api:app --reload
```

Unset (the default) adds no CORS headers at all, which is correct when a reverse
proxy serves the dashboard and the API from the same origin. The allow-list is
never `*` - see [`docs/limitations.md`](docs/limitations.md).

### Optional: API key and rate limit

Both are **off by default**, because the deployment this project has - a laptop
on a LAN behind a reverse proxy - is protected by its network, and turning them
on by default would break every existing client for no gain. Turn them on if the
service is ever reachable from anything you do not control:

```bash
export AGRIGUARD_API_KEY="<long random string>"    # clients send X-AgriGuard-Key
export AGRIGUARD_RATE_LIMIT="30/minute"            # per client, per process
```

A client then sends the key on every request:

```bash
curl -X POST http://localhost:8000/v1/predict/irrigation \
  -H "Content-Type: application/json" \
  -H "X-AgriGuard-Key: $AGRIGUARD_API_KEY" \
  -d '{"soil_moisture": 20, "temperature": 33, "humidity": 42, "light": 68000}'
```

A missing or wrong key gets `401`; exceeding the budget gets `429` with a
`Retry-After` header. The key is checked before the limit, so an unauthenticated
flood cannot lock a legitimate client out. An unparseable
`AGRIGUARD_RATE_LIMIT` **refuses to start** rather than quietly running
unlimited. `agriguard/security.py` lists what these guards do not cover -
notably TLS and multi-worker deployments, where the counters are per process.

Check both models are loaded before trusting any output:

```bash
curl http://127.0.0.1:8000/v1/health
```

`"status": "ok"` means both loaded. `"degraded"` means one fell back to rules or
is missing - read `models.<name>.load_error`.

## Reproducing everything from scratch

```bash
pip install -r requirements.txt

# 1. synthetic irrigation data (sketchy but deterministic; re-runs give a new series)
python scripts/generate_irrigation_data.py --days 30 --period-min 10

# 2. download + prepare the PlantVillage subset (~15 min first time, then cached)
python scripts/prepare_crop_dataset.py --per-class 150

# 3. train (irrigation: seconds; crop-health: ~35 min on 4 CPU cores)
python scripts/train_irrigation.py
python scripts/train_crophealth.py --epochs 6

# 4. re-measure both models from the saved weights
python scripts/evaluate_models.py

# 5. print the per-crop moisture boundaries quoted in the docs
python scripts/decision_boundaries.py

# 6. prove it end to end
python -m pytest
python scripts/api_demo.py --endpoint both

# 7. regenerate examples/ and the evidence transcripts
python scripts/make_examples.py
python scripts/make_evidence.py

# 8. audit every Stage 3 requirement against the shipped artefacts (exit 0 = pass)
python scripts/audit_stage3.py
```

Neither training script touches the test split while fitting, and
`train_crophealth.py` selects the saved weights on **validation** accuracy
before the test split is opened. Every number in this README, in `docs/` and in
`evidence/` is printed by one of these commands, so you can regenerate and check
any of them.

## Handing this over to a reviewer

```bash
python scripts/package_release.py
```

Writes `dist/AgriGuard_AI_<version>.zip` and re-opens it to prove the exclusions
held. Do not zip the folder by hand: an archive built that way is what shipped
a 1 GB, hard-coded-path `.venv/` last time, because dragging a folder into
WinRAR never reads `.gitignore`.

The dataset is **included** (it is regenerable, but a reviewer cannot check any
claimed number without it). Pass `--with-dataset-ignored` for a slim pack when
the recipient is expected to run the preparation scripts first.

Unpack it, then check it before trusting it - both must be green:

```bash
python -m pytest -q          # 150 passed
python scripts/audit_stage3.py
```

`tests/test_docs_consistency.py` is the check that matters for a handover: it
fails if the documentation quotes different metrics from the model file that
actually ships, which is the exact defect a hand-built archive drifts into.

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/` | service name, version, endpoint list |
| `GET` | `/v1/health` | status and per-model load state |
| `GET` | `/v1/classes/crop-health` | the two class keys, labels and dashboard hints |
| `POST` | `/v1/predict/irrigation` | JSON in, irrigation recommendation out |
| `POST` | `/v1/predict/crop-health` | `multipart/form-data` image in, `Healthy` / `Possible Disease` out |

### POST /v1/predict/irrigation

```bash
curl -X POST http://127.0.0.1:8000/v1/predict/irrigation \
  -H "Content-Type: application/json" \
  -d '{"timestamp":"2026-09-27T18:30:00Z","device_id":"ESP32_01","plot_id":"PLOT_A",
       "soil_moisture":20,"temperature":33,"humidity":42,"light":68000,
       "water_flow":0,"pump_status":0,"crop_type":"tomato",
       "history":{"recent_soil_moisture":[{"timestamp":"2026-09-27T18:10:00Z","soil_moisture":23.4},
                                          {"timestamp":"2026-09-27T18:20:00Z","soil_moisture":21.9}],
                  "last_irrigation_at":"2026-09-27T15:00:00Z"}}'
```

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

Use the `irrigation_needed` boolean the service computed. Do not re-threshold
`prediction_score` yourself at 0.5 - the operating point is 0.35 and is reported
by `GET /v1/health`. If you do re-derive it, use
`irrigation_needed == (prediction_score >= decision_threshold)`, which holds on
every code path including the fallbacks.

`decision_source` says what produced the advice: `model`, one of the two safety
guards, `rule_fallback` when no model file is present, or
`rule_fallback_out_of_range` when a reading fell outside the range the model was
fitted on and the crop threshold rule decided instead. Read it to know how much
the number is worth, not to work out the decision.

`prediction_score` saturates near 1.0 on a dry reading. It is a
logistic-regression score, **not** a calibrated probability - do not render it as
a percentage. The
`history` block is optional; without it the service substitutes placeholders,
returns `"history_used": false` and warns you in `reason`.

### POST /v1/predict/crop-health

```bash
curl -X POST http://127.0.0.1:8000/v1/predict/crop-health -F "image=@capture.jpg"
```

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

```text
The model classifies images into Healthy or Possible Disease within the scope
of the training dataset. It is a prototype screening/classification model and is
not a universal disease diagnostic system.
```

It returns **503**, never a colour heuristic, when the weights are missing.

## How the irrigation model works

1. Ten features: the four live sensor readings (`soil_moisture`, `temperature`,
   `humidity`, `light`), the three derived historical features
   (`soil_moisture_change`, `soil_moisture_avg_30min`,
   `time_since_last_irrigation_min`) computed per `(device_id, plot_id)` from
   sequential readings only, and three crop indicators.
2. `irrigation_event` is the label and is never an input - the historical
   features are built from `pump_status` / `water_flow` so nothing leaks.
3. If `pump_status == 1` or `water_flow > 0` it always answers
   `Irrigation Not Needed`, whatever the model says, with
   `prediction_score: 0.0`, `history_used: null` and
   `decision_source: pump_guard` / `water_guard`.
4. If any client-supplied reading falls outside the range the model was fitted
   on, the model's answer is not used: the crop threshold rule decides and
   `decision_source` is `rule_fallback_out_of_range`. The model score is still
   computed and quoted in `reason` for diagnosis. This exists because the fitted
   `light` coefficient is negative, so the model extrapolates confidently in the
   wrong direction on weather it has never seen - see
   [`docs/limitations.md`](docs/limitations.md).
5. If `models/irrigation_model.json` is missing entirely, the service falls back
   to crop-specific moisture thresholds and reports
   `decision_source: rule_fallback`.
6. `irrigation_duration_min` is always `null`: no duration model was built,
   because a fake number here is worse than none.
7. Use the `irrigation_needed` boolean. If you must re-derive it,
   `prediction_score >= decision_threshold` gives the same answer on every path,
   guard and fallback included - see
   [`docs/api_contract.md`](docs/api_contract.md#re-deriving-the-decision-from-prediction_score).

## How the crop-health model works

1. MobileNetV3-Small pretrained on ImageNet, 2-class head.
2. Stage 1 trains the head with a frozen backbone, stage 2 unfreezes the last
   two backbone blocks at a low learning rate.
3. Training, validation, test and live inference all share one transform in
   `agriguard/imageutils.py`, so the network can never be served a differently
   preprocessed image.
4. 38 PlantVillage classes are mapped to two categories by an exact rule:
   names ending in `healthy` -> `Healthy`, all 26 named diseases ->
   `Possible Disease`. The mapping is machine-readable in
   `data/crop_health/class_mapping.json`.
5. Near-duplicate images are detected with a 64-bit dHash and kept inside a
   single split, so a re-encoded twin of a test photo cannot be in training.

## Repository layout

```text
agriguard/            service code
  api.py              FastAPI endpoints, validation, error contract
  contracts.py        Pydantic request/response models
  cropdata.py         PlantVillage URLs, class order, API labels
  crophealth.py       MobileNetV3-Small loader + inference
  imageutils.py       the one shared image transform
  irrigation.py       logistic-regression loader, rule fallback, guards
  preprocess.py       cleaning, causal historical features, chronological split
  config.py           shared filesystem anchors
  version.py          service and model versions
data/
  irrigation_data.csv synthetic irrigation dataset
  crop_health/        prepared images, splits.csv, class_mapping.json, stats
models/               trained weights and measured metrics (JSON)
docs/                 contract, dataset, results, limitations, integration
evidence/             generated transcripts of the real commands
examples/             real-model example requests and responses
scripts/              data generation, training, evaluation, examples, evidence
  audit_stage3.py     requirement-by-requirement audit of the shipped artefacts
  eda.py              correlations, cross-validation, hypothesis tests, outliers
  package_release.py  builds a handover archive with the local junk excluded
tests/                pytest suite
```

## Tests

```bash
pytest
```

Covers the historical features (including a no-look-forward assertion), the
chronological split, the safety guards, the API contract and error codes, the
shared image transform, the real-model crop evaluation on held-out PlantVillage
images, and the offline dataset-preparation helpers. Some crop-model tests skip
with an explicit reason if the weights or the dataset are not built yet - they
never silently pass.

## Honesty checklist

| Claim | Allowed? |
| --- | --- |
| "0.8144 balanced accuracy on our synthetic irrigation test split" | yes, with "synthetic" in the sentence |
| "our irrigation model is 93 % accurate in the field" | no - the 0.93 figure came from a leak that has been fixed, and nothing here was measured in a field |
| "our irrigation model is 94 % accurate" | no without the "synthetic test split" qualifier |
| "our irrigation model uses temperature, humidity and light" | yes, inside the range it was trained on |
| "classifies leaf images as Healthy or Possible Disease" | yes |
| "diagnoses plant diseases" | no - it does not name the disease |
| "N % accurate on real ESP32-CAM images" | no - never tested on them |
| "the AI decides when to water" | no - it recommends |
| "the model runs the pump" | no - nothing here touches hardware |

Longer version, with reasons: [`docs/limitations.md`](docs/limitations.md).

## Documentation index

| Document | What it is for |
| --- | --- |
| [`docs/api_contract.md`](docs/api_contract.md) | exact request/response format for Member 3 |
| [`docs/integration_guide.md`](docs/integration_guide.md) | step-by-step wiring, copy-paste backend client, troubleshooting |
| [`docs/dataset_structure.md`](docs/dataset_structure.md) | irrigation columns, units, features, split rules |
| [`docs/crop_health_dataset.md`](docs/crop_health_dataset.md) | PlantVillage source, class mapping, counts, preprocessing |
| [`docs/model_results_and_tests.md`](docs/model_results_and_tests.md) | all measured metrics and the test suite |
| [`docs/crop_health_results.md`](docs/crop_health_results.md) | crop-health metrics, confusion matrix, limitations |
| [`docs/eda_report.md`](docs/eda_report.md) | correlations, cross-validation, hypothesis tests, outlier scan |
| [`docs/limitations.md`](docs/limitations.md) | what these numbers do not mean |
| [`evidence/`](evidence) | transcripts generated by `scripts/make_evidence.py` |
| [`examples/`](examples) | real request/response pairs from `scripts/make_examples.py` |
| `scripts/audit_stage3.py` | one-line-per-requirement audit of the shipped artefacts; prints PASS/FAIL and exits non-zero on any failure |

## Versions

| Component | Version |
| --- | --- |
| AI service | 0.2.0 |
| Irrigation model | 0.2.0 |
| Crop-health model | 0.2.0 |
| API contract | 0.2.0 |

## Licence / attribution

PlantVillage images come from
[spMohanty/PlantVillage-Dataset](https://github.com/spMohanty/PlantVillage-Dataset)
(Mohanty, Hughes & Salathe, [arXiv:1511.08060](https://arxiv.org/abs/1511.08060)),
fetched from the Hugging Face mirror `GVJahnavi/PlantVillage_dataset`. Check the
upstream licence before any commercial or public redistribution. See
`docs/limitations.md` for the full attribution.


