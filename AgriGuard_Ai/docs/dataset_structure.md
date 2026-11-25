# AgriGuard irrigation dataset structure (v2)

This is the canonical schema for everything the ESP32 sends and everything the
AI layer stores. The same field names are used in the CSV on disk and at the
HTTP boundary, so nothing has to be renamed in between.

Source of truth in code: `agriguard/preprocess.py`
(`CANONICAL_COLUMNS`, `REQUIRED_COLUMNS`, `SENSOR_RANGES`).

## Column reference

| # | Column | Type | Nullable | Meaning |
| - | ------ | ---- | -------- | ------- |
| 1 | `timestamp` | ISO-8601 string | no | When the reading was recorded (ESP32 RTC). The AI layer parses it and orders every computation by it. |
| 2 | `device_id` | string | filled if missing | Which ESP32 node produced the reading, e.g. `ESP32_01`. Historical features are computed per device. |
| 3 | `plot_id` | string | filled if missing | Which agricultural plot the node sits in, e.g. `PLOT_A`. Historical features are computed per (device, plot). |
| 4 | `soil_moisture` | float | no | Soil moisture reading on the scale the ESP32 firmware reports. 0 = dry, 100 = saturated. This is the single most important feature. |
| 5 | `temperature` | float | no | Ambient temperature in degrees Celsius. Higher temperature means faster evaporation. |
| 6 | `humidity` | float | no | Relative humidity, 0-100 %. Lower humidity means faster evaporation. |
| 7 | `light` | float | no | Light intensity in whatever scale the fitted sensor reports. See the warning below. |
| 8 | `npk_n` | float | **yes** | Soil nitrogen. `null` when no NPK probe is fitted. |
| 9 | `npk_p` | float | **yes** | Soil phosphorus. `null` when no NPK probe is fitted. |
| 10 | `npk_k` | float | **yes** | Soil potassium. `null` when no NPK probe is fitted. |
| 11 | `water_flow` | float | fills to 0.0 | Water-flow reading in the sensor's own scale. 0 while the pump is off. |
| 12 | `pump_status` | int 0/1 | fills to 0 | Pump state: `0` = off, `1` = on. |
| 13 | `irrigation_event` | int 0/1 | no | **Label.** `1` = an irrigation action started at this reading, `0` = none. Training only; the ESP32 never sends it. |
| 14 | `crop_type` | string | optional | Crop grown in the plot, e.g. `tomato`. Plot configuration, not a sensor. Supplies three of the model's ten features via `crop_is_*` indicators; a crop with no indicator (anything other than tomato, rice, pepper) is answered by the rule-based fallback instead. |

### Unit warnings - read before interpreting any number

- **`light` is not claimed to be lux.** It is whatever the ESP32 firmware
  reports. Many BH1750-style modules are configured for raw ADC counts, and some
  cheap sensors output a 0-1000 "brightness" index. AgriGuard stores the value
  as given and only rejects negative numbers. The cleaning range
  `0 - 200 000` is a sanity guard against garbage, not a sensor specification.
- **`soil_moisture` is not claimed to be calibrated volumetric water content.**
  A capacitive probe reports a raw ADC level that is only proportional to water
  content until it is calibrated against soil and soil type. The bundled
  thresholds are placeholders. See `docs/limitations.md`.
- **`water_flow` is not claimed to be L/min.** It is the sensor's own scale.
  The model only uses it as a boolean "is water moving right now" signal.
- **NPK has no unit declared** because the NPK probe in this project has not
  been calibrated. The fields exist so the schema is complete and so real NPK
  data can be added later without a breaking change.

## Required vs optional

```text
REQUIRED : timestamp, soil_moisture, temperature, humidity, light, irrigation_event
OPTIONAL : device_id, plot_id, npk_n, npk_p, npk_k, water_flow, pump_status, crop_type
```

Missing optional columns are created with defaults
(`device_id=ESP32_UNKNOWN`, `plot_id=PLOT_UNKNOWN`, `water_flow=0`,
`pump_status=0`, `crop_type=default`, NPK left empty). A missing **required**
column raises `MissingColumnsError` immediately - the training run stops rather
than silently training on garbage.

## Cleaning rules

`agriguard/preprocess.py::clean_irrigation_frame` applies exactly this order:

1. Validate that every required column exists (else `MissingColumnsError`).
2. Add missing optional columns with their defaults.
3. Parse `timestamp` to UTC; drop rows with an unparsable timestamp.
4. Convert every numeric column with `errors="coerce"`, so junk becomes `NaN`.
5. Drop rows missing any of the four live sensor readings.
6. Drop rows whose `irrigation_event` is not 0 or 1.
7. Drop rows whose `pump_status` is not 0 or 1.
8. Clip out-of-range numeric values to the documented range and count them.
9. Fill empty NPK with the column median (kept only for a tidy stored dataset;
   NPK is not a model input).
10. Sort by `timestamp` with a **stable** sort, because several devices share a
    timestamp and the order of tied rows must be identical between runs.

| Column | Acceptable range | Rationale |
| ------ | ---------------- | --------- |
| `soil_moisture` | 0 - 100 | sensor scale is 0-100 by construction |
| `temperature` | -40 - 80 °C | wider than any plausible field value |
| `humidity` | 0 - 100 % | physical limit |
| `light` | 0 - 200 000 | sanity bound, not a lux spec |
| `water_flow` | 0 - 100 000 | sanity bound, not a flow spec |
| `npk_n/p/k` | 0 - 5 000 | sanity bound |

The cleaning report (rows in, rows dropped, values clipped) is printed by the
training script and stored inside `models/irrigation_model.json` under
`cleaning_report`.

## Model features

The trained model uses **ten** columns, listed here in the exact order stored
in `models/irrigation_model.json`. The order matters: the coefficients, the
scaler means/stds and the medians are all aligned to this one vector, and the
loader rejects a model file that names a different set rather than mis-scoring
it.

| # | Feature | Source | Client-supplied? |
| - | ------- | ------ | ---------------- |
| 1 | `soil_moisture` | current reading | yes |
| 2 | `soil_moisture_change` | `soil_moisture[t] - soil_moisture[t-1]`, same device/plot | derived |
| 3 | `soil_moisture_avg_30min` | mean of `soil_moisture` over the window `(t-30min, t]` | derived |
| 4 | `time_since_last_irrigation_min` | minutes since the last row where `pump_status=1` or `water_flow>0`, capped at 1440 | derived |
| 5 | `temperature` | current reading | yes |
| 6 | `humidity` | current reading | yes |
| 7 | `light` | current reading | yes |
| 8 | `crop_is_tomato` | one-hot over the trained crops | derived |
| 9 | `crop_is_rice` | one-hot over the trained crops | derived |
| 10 | `crop_is_pepper` | one-hot over the trained crops | derived |

`npk_*`, `device_id` and `plot_id` are **not** model inputs. They are kept in
the dataset and in the API contract so the schema stays complete and future
models can use them.

The "client-supplied?" column is not cosmetic: it is the set the out-of-range
guard checks against the ranges recorded at training time
(`training_feature_ranges` in the model file). Only the four current readings
are range-checked. The three historical features are derived - `change` in
particular moves whenever the current reading moves relative to a fixed
history - so range-checking them turned the per-crop trigger into an artifact of
that arithmetic rather than a property of the model. `CLIENT_SUPPLIED_FEATURES`
in `agriguard/irrigation.py` documents the reasoning in full, and
`docs/limitations.md` records what remains unguarded.

### The score saturates, on purpose of the fit

The fitted coefficients are large - `soil_moisture_avg_30min` has a coefficient
of about +15 and `time_since_last_irrigation_min` about -15 - because the
simulated labels are close to linearly separable in those two features. The
sigmoid therefore saturates: a dry reading scores `1.0`, a wet one `0.0`.

Weakening `C` to shrink the coefficients was tried and rejected: on a
chronological validation split the balanced accuracy falls monotonically
(0.71 at `C=1.0`, 0.53 at `C=0.1`, 0.50 at `C=0.01`). The default `C=1.0` is
therefore kept, and the saturation is documented instead of hidden. The
practical consequence is in `docs/api_contract.md`: use the score to cross the
threshold and to order predictions, never as a percentage.

### The three historical features are leakage-free

- Every window looks **backwards only**. A row's features depend on that row and
  earlier rows of the same device/plot, never on a later one. There is a test
  for exactly this: `test_historical_features_never_look_forward` truncates the
  timeline and asserts the surviving feature values are unchanged.
- `time_since_last_irrigation_min` is built from `pump_status` and
  `water_flow`, which are **observable**. It deliberately never reads
  `irrigation_event`, because that column is the target and using it would be
  target leakage.
- Features are computed per `(device_id, plot_id)`, so one plot's history never
  leaks into another's.

### They need sequential real readings

These three features are only meaningful when the CSV is a continuous log from a
running ESP32. The bundled synthetic CSV is a simulation, so its historical
features are plausible but not field measurements. The feature code is written
for the real thing: give it a real ESP32 CSV with the columns above and it will
work unchanged.

`models/irrigation_model.json` reports what happens without history. Measured on
the test split using the neutral placeholders a real history-less request
produces (`avg30` = current reading, `change` = 0, `time_since` = 1440) - the
situation when the backend sends a single reading and no history:

| Variant | Accuracy | Balanced accuracy | Precision | Recall | F1 |
| ------- | -------- | ----------------- | --------- | ------- | -- |
| With historical features | 0.9406 | 0.8144 | 0.6504 | 0.6622 | 0.6562 |
| No history (neutral placeholders) | 0.8287 | 0.9002 | 0.3318 | 0.9865 | 0.4966 |

The gap between those two rows is the honest answer to "do the historical
features matter here?" - they matter. Without them the model errs towards
recommending water (recall 0.99, precision 0.33). Note that balanced accuracy
*rises* in the no-history row, which looks like an improvement and is not: the
model answers almost everything "irrigate", and the class weighting rewards that.
Accuracy, precision and F1 all get worse. Read `prediction_score` and precision
together, never balanced accuracy alone. The response says which case you are in
at runtime through `prediction.history_used`. Read the same numbers from
`models/irrigation_metrics.json` -> `test_metrics` and
`test_metrics_without_history`; they are also printed by
`python scripts/train_irrigation.py`.

## What the model actually uses

Ten features, in this order:

| # | Feature | Why |
| - | ------- | --- |
| 1 | `soil_moisture` | The current reading; the dominant term |
| 2 | `soil_moisture_change` | Is the soil drying faster than the last reading? |
| 3 | `soil_moisture_avg_30min` | 30-minute mean, i.e. how far the soil has already dipped |
| 4 | `time_since_last_irrigation_min` | Has a run just finished, and will water be wasted? |
| 5 | `temperature` | Hotter air raises evaporative demand |
| 6 | `humidity` | Drier air raises evaporative demand |
| 7 | `light` | Sun raises evaporative demand - but see below |
| 8 | `crop_is_tomato` | The trigger level is crop specific |
| 9 | `crop_is_rice` | Rice is kept near 55-60 %, tomato and pepper near 25-26 % |
| 10 | `crop_is_pepper` | Without the crop the two moisture clusters are inseparable |

`temperature` and `humidity` fit the expected signs: hotter and drier raise the
irrigation score. `light` does not - its fitted coefficient is **negative**, so
bright sun makes the model *less* likely to recommend water on dry soil. That is
a real property of this synthetic dataset rather than a bug: the label is the
instant a threshold scheduler fired, so daytime readings are correlated with
"already watered" and the weather acts as a proxy for scheduler timing.

The brief requires all four current readings to be model inputs, so they are in
the model. The cost is that the model can now extrapolate confidently in the
wrong direction on weather it has never seen, which is why the service
range-checks the four readings and defers to the crop threshold rule outside the
recorded training bounds. The measurement, the exact ranges, and the remaining
unguarded case are in `docs/limitations.md`.
`UNUSED_WEATHER_COLUMNS` in `agriguard/preprocess.py` is kept as an alias for
that section's cross-references; the columns are now used, and the name is
misleading.

## The label must not be visible in the inputs

An earlier revision reported the pump as running on the same row as
`irrigation_event`, which made `time_since_last_irrigation_min == 0` a perfect
predictor of the label and produced a meaningless 0.93 balanced accuracy. The
generator now reports the pump from the next reading onwards, and
`tests/test_irrigation.py` asserts that no positive row carries a running pump.
This is the single most important rule in this dataset contract.

## Train/test split

**Chronological, never shuffled.**

```text
first 80 % of the timeline  ->  training
last 20 % of the timeline   ->  testing
```

Implemented in `preprocess.chronological_split`. A random split on a time series
would let the model be scored on readings from *before* its training readings,
which is data leakage and would make the reported score meaningless. The scaler
is fitted on training rows only, for the same reason.

The chosen boundary, the row counts and the two time ranges are stored in
`models/irrigation_model.json` under `split`.

## Example row

```csv
timestamp,device_id,plot_id,soil_moisture,temperature,humidity,light,npk_n,npk_p,npk_k,water_flow,pump_status,irrigation_event,crop_type
2026-03-01T00:00:00Z,ESP32_01,PLOT_A,33.13,13.8,90.8,0.0,17.2,21.7,30.9,0.0,0,0,tomato
2026-03-01T00:00:00Z,ESP32_02,PLOT_B,71.36,15.23,89.37,439.4,,,,0.0,0,0,rice
2026-03-01T00:00:00Z,ESP32_03,PLOT_C,27.29,15.34,92.25,136.3,,,,0.0,0,0,pepper
```

The empty NPK cells are the documented "no NPK sensor on this node" case.

## The bundled CSV is synthetic

```text
This dataset is synthetic prototype data generated for development and testing.
It is not real agricultural field data.
Model performance on this dataset must not be interpreted as real-world field
accuracy.
```

It is produced by `scripts/generate_irrigation_data.py`, which simulates three
plots minute by minute: soil moisture dries at a rate driven by temperature,
humidity and light plus a hidden wind term, a controller switches the pump on
when the *true* moisture drops below the crop threshold, and the CSV records the
*noisy sensor* reading. The reported pump state covers the interval that just
ended, so the pump becomes visible one reading **after** the decision - putting
it on the label row itself would leak the answer. Because the label also depends
on hidden quantities, no model can separate the classes perfectly - which is why
the reported metrics are believable rather than a suspicious 100 %.

To replace it with real data, overwrite `data/irrigation_data.csv` with a real
ESP32 export using these column names and re-run
`python scripts/train_irrigation.py`. Nothing else has to change.
