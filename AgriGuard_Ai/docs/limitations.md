# Known limitations (v2)

Read this before quoting any number from this project, and before showing any
result to a user, a marker or a farmer. Everything here is a real constraint of
the current implementation, not a disclaimer for form's sake.

---

## 1. The irrigation dataset is synthetic

```text
This dataset is synthetic prototype data generated for development and testing.
It is not real agricultural field data.
Model performance on this dataset must not be interpreted as real-world field
accuracy.
```

`data/irrigation_data.csv` was produced by
`scripts/generate_irrigation_data.py`, which simulates soil drying and a
threshold-based irrigation controller. The label depends on hidden variables
(per-plot compaction, a wind term, sensor noise), so the classes are genuinely
not separable and the metrics are plausible rather than suspiciously perfect.

**Consequences:**

- A good score on this data proves the **pipeline** works: cleaning, feature
  engineering, chronological splitting, training, serving.
- It proves nothing about real crops. Expect the real field score to be lower.
- The soil-moisture thresholds embedded in the rule fallback are simulator
  constants. They are not agronomically validated.
- **The simulation has no geography and no season.** There is no location, no
  planting calendar, no rainfall, no irrigation quota, and no growing season
  anywhere in `data/irrigation_data.csv`. "Drying" is a smooth function of hours
  since watering, which is roughly true in a hot arid greenhouse and badly wrong
  in a humid one, on clay that holds water for days, or during a rainy week when
  the crop is watered by the sky. The learned per-crop triggers (~21 % for tomato
  and pepper, ~60 % for rice) are therefore properties of the simulator, and they
  are **not** transferable to a different climate, soil type, or substrate
  without re-measuring them on real plots. Treat them as a starting point for
  calibration, never as a validated number.

**To fix:** collect real ESP32 CSV exports with the documented column names and
re-run `python scripts/train_irrigation.py`. No code change is needed.

---

## 2. The model has no calibration, and no duration output

- `prediction_score` is a logistic-regression probability on a **synthetic**
  distribution. It is not calibrated, and it **saturates**: the simulated labels
  are close to linearly separable, so the fitted coefficients are large and dry
  readings score `1.0` while wet readings score `0.0`. Treat it as a
  threshold-crossing and ordering signal, never as a percentage.
- `history_used` tells you when that score is not trustworthy at all. It is
  `false` whenever the three historical features had to be filled with
  placeholders, and the accuracy in that mode is poor (see section 3).
- `irrigation_duration_min` is **always `null`**. There is no defensible
  duration model, so the service does not invent one. Water-volume policy belongs
  to the backend and the ESP32, and it must be enforced there.
- `confidence` on the crop side is an uncalibrated softmax score, and the
  classes are imbalanced, so it should be read as a *relative* score.

---

## 3. An earlier revision of this project reported a fake 0.93

The first released version of the synthetic generator reported the pump state on
the **same row** as the irrigation event:

```python
"water_flow":  water_flow if (pump_steps_left > 0 or irrigated_now) else 0.0,
"pump_status": 1 if (pump_steps_left > 0 or irrigated_now) else 0,
"irrigation_event": 1 if irrigated_now else 0,
```

`irrigation_now` is true exactly on the label row, so every positive row also had
`pump_status == 1`, `water_flow > 0`, and therefore
`time_since_last_irrigation_min == 0`. That feature is a **perfect giveaway for
the label**: 100 % of positives sat at 0 and no negative ever did. The model
learned nothing about irrigation at all, and it scored **0.9305 balanced
accuracy** on the test split.

The generator now reports the pump from the *next* reading onwards, which is both
physically correct (a decision taken at time *t* moves water during *t+1*) and
leak-free. `tests/test_irrigation.py` now asserts that no positive row carries a
running pump or a zero time-since-irrigation value, so the leak cannot come back.

Two real modelling bugs were found and fixed at the same time, and both are
covered by regression tests:

| Bug | Symptom | Fix |
| --- | ------- | --- |
| Target leak via the pump state | 0.9305 balanced accuracy with no skill | Pump reported one interval later |
| No crop context | The model had to satisfy "water below ~21" (tomato) and "water below ~60" (rice) with one shared linear term, so recall collapsed to 0.171 | `crop_type` one-hot indicators |
| Weather readings as model inputs | A bright afternoon reading of 20 % soil moisture on a tomato plot was scored "do not irrigate" | `temperature`/`humidity`/`light` are no longer features |

**The lesson to carry forward: a suspiciously high score on data you generated
yourself is a bug report, not a result.** Check that no input column is
determined by the label before believing any number.

---

## 4. The weather inputs are in the model, and that is a real risk

`irrigation_event` marks the instant a **threshold scheduler** fired, not the
moment a plant became water-stressed. That distinction decides what the weather
features can learn:

- The event fires when the simulator's *hidden* moisture crosses a per-crop
  threshold, so the recorded moisture at an event is pinned near that threshold
  regardless of weather.
- At a *fixed* soil moisture, a high evaporative demand means the soil is on a
  faster trajectory and has usually **already been pumped**. This makes the
  weather effect ambiguous in a way the labels cannot resolve.

The brief requires `soil_moisture`, `temperature`, `humidity` and `light` to be
model inputs, so they are. **This section previously documented the opposite
choice** - it described a shipped model that excluded the weather readings, and
justified it on the grounds that the weather variant scored worse. That was
retracted once already, because the claimed figure did not reproduce. The
re-measurement showed the weather variant is the *better* model, and the model
was switched to it. Do not reintroduce a variant claim here without re-running
`scripts/train_irrigation.py` and pasting the measured table.

### What the weather features actually learned

Standardised coefficients, read from `models/irrigation_model.json` in its own
feature order (an earlier draft of this file listed these shifted by one
position, which made `humidity` look like the dominant term):

| Feature | Coefficient | Sign agrees with agronomy? |
| ------- | ----------- | -------------------------- |
| `soil_moisture` | -10.72 | yes - drier soil, more water needed |
| `soil_moisture_change` | -0.77 | yes - soil dropping, more water needed |
| `soil_moisture_avg_30min` | -3.16 | yes - a wet recent average means it was already watered |
| `time_since_last_irrigation_min` | +0.18 | yes - long since a run, water will be used |
| `temperature` | +0.38 | yes - hotter air, more evaporative demand |
| `humidity` | -0.24 | yes - drier air, more evaporative demand |
| `light` | **-1.32** | **no** - see below |
| `crop_is_rice` | +8.79 | yes - rice is kept far wetter |
| `crop_is_tomato` | -4.35 | yes |
| `crop_is_pepper` | -4.45 | yes |

`light` has the wrong sign, and it is the second-largest weather term. In the
synthetic data the label marks the instant a scheduler fired, so bright readings
are correlated with "already watered"; the model learns "sunny means someone
already irrigated this plot". That is a property of the label, not a coding
error, and it is why the model must not be trusted on a reading brighter or
colder than any in the training set.

`test_bright_light_pulls_the_score_down_and_that_is_a_documented_weakness` pins
the sign so it cannot be rediscovered as a surprise, and fails loudly if real
field data ever flips it.

### The measured failure, and the guard

At the training median weather a tomato plot at 20 % soil moisture scores well
clear of the threshold. Move the weather to -10 C / 0 % RH / 200 000 lux - all
three readings outside the recorded training range - and the model alone scores:

| `soil_moisture` | Model score alone | Model's own answer | Served answer |
| --------------- | ----------------- | ------------------ | ------------- |
| 16 % | 0.079 | do not irrigate | **irrigate** (`rule_fallback_out_of_range`, 0.53) |
| 20 % | 0.004 | do not irrigate | **irrigate** (`rule_fallback_out_of_range`, 0.45) |
| 24 % | 0.000 | do not irrigate | **irrigate** (`rule_fallback_out_of_range`, 0.37) |
| 26 % | 0.000 | do not irrigate | do not water (`rule_fallback_out_of_range`, 0.33) |

The first three rows are the failure: soil at 16-24 %, well past that crop's
trigger, and the model says do not irrigate because the weather pushed it there.
The bundled training set contains humidity below 38.5 %, temperature below
12.8 C and light above 84 562 lux nowhere at all, and never in that
combination, so this is a linear model answering far outside its support. The
saturation to 0.000 at 24 % shows how far: the extrapolation is not a near miss
that happened to land on the wrong side, it is the model being fully confident.

The service therefore does not use the model outside the ranges it was fitted
on. `training_feature_ranges` in the model file records the per-feature bounds:

| Feature | Trained range |
| ------- | ------------- |
| `soil_moisture` | 13.76 .. 77.31 % |
| `temperature` | 12.83 .. 35.22 C |
| `humidity` | 38.49 .. 97.05 % |
| `light` | 0 .. 84 562 lux |

A request with any of those four outside its bounds gets
`decision_source: rule_fallback_out_of_range`: the crop threshold rule decides,
`prediction_score` is the rule's score, and the model's own extrapolated score is
quoted in `reason` for diagnosis only. The invariant
`irrigation_needed == (prediction_score >= decision_threshold)` still holds, so a
backend needs no special case. See
[`api_contract.md`](api_contract.md#decision_source-values).

### What the guard does not cover

- **The derived historical features are not range-checked.** `soil_moisture_change`
  is the gap between the current reading and the previous one, so any legitimate
  current reading paired with a slightly stale history pushes it outside the
  marginal range. Range-checking it turned the per-crop trigger into an artifact
  of that arithmetic - the boundary came out as the crop's reference threshold
  regardless of the model, and history sensitivity above ~26 % became
  unreachable. Since `change` and `avg_30min` both carry negative coefficients,
  their extrapolation directions raise the irrigation score or coincide with
  genuinely wetter soil, so the safe direction is the one left unguarded.
- **A client can still feed a bogus history.** Nothing validates that
  `recent_soil_moisture` is physically consistent with the current reading. A
  backend that forwards unvalidated history can drive the model somewhere it
  should not go, and the out-of-range guard will not catch it because it only
  inspects the four current readings.
- **The ranges are per feature, not joint.** A point inside all four marginal
  ranges can still be an unusual combination, and the model is then
  extrapolating in the joint space while reporting `decision_source: model`. A
  Mahalanobis-style joint test would be tighter; the per-feature test is the
  weaker claim that can actually be justified from a marginal histogram.
- **Real frost, heat and shade conditions are outside the range by
  construction**, because the synthetic generator never produced them. The guard
  is doing its job there - but "the rule answered" is not the same as "the answer
  is good".

### What the model does well

It learns the per-crop moisture trigger and the effect of history on it. Under a
realistic "recently watered" history (`avg30=35`, `change=-2`, `tsl=60`) at the
training median weather, it places the boundary at 20.8 % for tomato and 20.5 % for
pepper, against actual event medians of 20.9 % and 21.1 % - a close match, and the
clearest evidence that sending the history block is what makes the model
worthwhile. Rice's trigger sits above the range where that history context is
representable, so no boundary is quoted for it rather than reporting the search
limit. The trigger moves with the history and with the weather, which is the
behaviour the brief asks for and the reason the boundary must be quoted with its
context rather than as a constant. Regenerate every number in this section with
`python scripts/decision_boundaries.py`.

---

## 5. The historical features need real sequential data

`soil_moisture_change`, `soil_moisture_avg_30min` and
`time_since_last_irrigation_min` are the three features the model relies on
most. On the bundled synthetic CSV the drying curve is smooth, so these features
are almost perfectly informative - a real field with irregular watering,
mid-day evaporation cycles and sensor dropouts looks nothing like that.

`models/irrigation_metrics.json` reports `test_metrics_without_history`, which
feeds the neutral placeholders a real history-less request produces
(`avg30` = current reading, `change` = 0, `time_since` = 1440). That is the
honest estimate of what happens when the backend sends a single reading with no
history. Measured on the test split:

| Variant | Accuracy | Balanced accuracy | Precision | Recall | F1 |
| ------- | -------- | ----------------- | --------- | ------- | -- |
| With historical features | 0.9406 | 0.8144 | 0.6504 | 0.6622 | 0.6562 |
| No history (neutral placeholders) | 0.8287 | 0.9002 | 0.3318 | 0.9865 | 0.4966 |

Without history the model errs towards recommending water (recall 0.99,
precision 0.33). The response tells the backend when this is happening via
`prediction.history_used: false`. The mitigation is documented in
`docs/api_contract.md`: forward `history.recent_soil_moisture` and
`history.last_irrigation_at`.

**Read that balanced accuracy column carefully.** It goes *up* without history,
which looks like an improvement and is not: the model answers almost everything
"irrigate" and the class weighting rewards that. Accuracy, precision and F1 all
get worse. Balanced accuracy alone would make omitting history look like a win.

The failure direction was chosen deliberately. The neutral placeholder for
`soil_moisture_avg_30min` is the *current* reading rather than the training
median: substituting the median (much wetter than a dry reading) made the model
read a dry plot as "recently watered" and answer "do not irrigate". Erring
towards recommending water wastes some water; erring the other way can kill a
crop.

A second, uncomfortable observation from the same data: when the model is refit
on less of the timeline, performance falls sharply and has not plateaued at the
end of the learning curve. The synthetic drying pattern shifts over the 30 days,
so this model is more sensitive to its training window than a real dataset should
be. Treat 0.8144 balanced accuracy as optimistic even for synthetic data.


---

## 6. The rule-based fallback is a placeholder, not a validated model

If `models/irrigation_model.json` cannot be loaded, the service falls back to
moisture thresholds per crop type. Those thresholds are constants in
`agriguard/irrigation.py` chosen to be reasonable-looking. They have not been
validated against any real crop, soil or season.

`GET /v1/health` reports which path is live:

```json
"models": { "irrigation": { "trained": false, "load_error": "..." } }
```

Treat a `trained: false` irrigation model as a degraded service, not as a
working one.

---

## 8. Crop health: a two-class screening model, not a diagnostic tool

```text
The model classifies images into Healthy or Possible Disease within the scope
of the training dataset. It is a prototype screening/classification model and is
not a universal disease diagnostic system.
```

- It has **two** classes. 26 different PlantVillage diseases share the single
  label `possible_disease`. It cannot tell you which disease, how severe, or
  what to spray.
- It has **no** pest, nutrient-deficiency, water-stress, herbicide-damage or
  physical-damage class. It will happily call a drought-stressed leaf "Possible
  Disease".
- 14 crop species are covered. Anything else is out of distribution and will be
  scored confidently and wrongly.
- Only 12 of those 14 species have healthy training images. Orange and Squash
  appear **only** as diseases, so the model has never seen a healthy leaf of
  either. A healthy orange or squash leaf is not in the training distribution
  and the model has no way to represent it.

---

## 9. Crop health: the domain gap is the biggest risk

PlantVillage is **laboratory** data: a single detached leaf, filling the frame,
on a plain grey or black background, shot under controlled light. An ESP32-CAM in
a real plot produces something quite different:

| | PlantVillage (training) | ESP32-CAM (deployment) |
| - | ---------------------- | ---------------------- |
| Background | plain grey/black | soil, weeds, neighbouring leaves, plastic, sky |
| Framing | one leaf, centred | whatever the camera sees |
| Lighting | uniform, controlled | direct sun, hard shadows, low-light noise, colour-temperature shifts |
| Resolution | 256x256 originals | whatever the OV2640 gives at its JPEG quality setting |
| Motion | none | wind, a moving plant, a shaking camera |
| Symptoms | the disease fills the frame | a small lesion on one leaf among many |

A CNN trained on the left column and shown the right column commonly loses a
lot of accuracy. The training-time augmentation (random resized crop, flips,
rotation, colour jitter) helps but does not close the gap. **Do not present the
reported test accuracy as the expected field accuracy.**

**To fix this project properly:**

1. Collect real ESP32-CAM images from the actual plots, with labels confirmed by
   a plant pathologist, and fine-tune on them.
2. Photograph a single leaf against a plain background (or add a detachable
   white/black backdrop) to get closer to the training distribution.
3. Re-evaluate on a held-out set of *real* plot images before quoting any
   number.

### The model was never tested outside the crops and conditions it was trained on

This is a **geographic and agronomic** limit, not just an image-quality one.
Nothing in this project establishes that the model works anywhere it was not
trained:

| Scope | Trained on | Untested |
| ----- | ---------- | -------- |
| Crops | tomato, rice, pepper, and 11 other PlantVillage species | any locally grown crop, any greenhouse variety |
| Region / climate | none - the synthetic irrigation series has no location, and PlantVillage was assembled without a field calendar | this project's actual location, climate, and season |
| Soil and substrate | not represented at all; the irrigation simulator assumes a fixed soil-water curve | clay, sand, loam, saline, degraded soil |
| Growing system | none | greenhouse, hydroponic, vertical, shade-net |

The honest consequence: the measured `0.9657` test accuracy is a statement about
846 held-out *PlantVillage lab photographs*, and nothing more. It is not a
prediction of accuracy in this project's own greenhouse or field. A
`crop_type` the irrigation model was never trained on is never scored by that
model: it is answered by the rule-based threshold instead, and the response
reports `decision_source: "rule_fallback"` to say so. The same rule should be
applied by hand to any crop
image the training set does not cover.

### False positives and false negatives are expected, not exceptional

A two-class screen makes mistakes in both directions, and the two errors have
different costs:

- **False positive** (healthy leaf reported as `Possible Disease`): the grower
  inspects a healthy plant, may waste fungicide. Measured healthy-class
  precision is `0.9242`, so roughly 8 in 100 "Possible Disease" verdicts on
  lab-style images are healthy leaves.
- **False negative** (diseased leaf reported as `Healthy`): the grower does
  nothing. Measured healthy-class recall is `0.9697`, so roughly 3 in 100
  diseased lab-style images are cleared. **This is the more dangerous error,
  because it is silent** - there is no alarm, and the number is a plausible
  number.

Both rates are far better than chance but neither is a safety guarantee, and
both are measured on lab images, so under the domain gap in the previous section
they should be expected to be **worse in the field, not better**. The response
is a screening label, so the backend and the UI should present it as "inspect
this plant", never as a diagnosis and never as permission to skip inspection.

---

## 10. Class imbalance and shortcut learning

`Possible Disease` outnumbers `Healthy` in the dataset. Therefore:

- Report **balanced accuracy** and the confusion matrix, not just accuracy.
- `confidence` is skewed towards `possible_disease`.
- PlantVillage images share backgrounds, so a CNN can partly learn "this looks
  like a laboratory photo" instead of "this leaf has spots". A model with high
  test accuracy can still be useless in the field. The single-label-per-plant
  limitation below makes this worse.

**One image per class per source:** PlantVillage contains multiple near-identical
captures of the same plant. `prepare_crop_dataset.py` detects near-duplicates
with dHash and keeps them in one split so twins cannot leak across train and
test, but genuine near-twin photographs of the same leaf can still straddle the
split. This inflates the test score to an unknown degree.

---

## 11. Evaluation limits

- The irrigation test set is **the last 20 % of a synthetic timeline**. There is
  exactly one such period, so the numbers carry no confidence interval and no
  second chance.
- No cross-validation, no seed sweep, no hyperparameter search. One split, one
  seed (`42`), one run.
- The decision threshold **is** tuned, but only on the last 20 % of the
  *training* timeline, maximising F1. The test set was never used to pick it.
  Tuning against a real cost of a missed watering - rather than F1 - needs
  labelled field data and an agreed cost matrix.
- Only three crops (tomato, rice, pepper) are validated. A request for any other
  crop produces all-zero crop indicators and the response is flagged as
  extrapolation in `reason`; it should not be acted on.
- There is **no irrigation-duration model**, so no duration metric exists.
- The crop test set is used once, at the end. It never influenced training or
  the epoch count.

---

## 12. The AI layer is advisory only

```text
The AI layer is advisory: it produces recommendations.
It is not a direct control layer.
```

- Nothing in this repository switches a pump, opens a valve or triggers a spray.
- Every response carries `advisory_only: true`.
- The service returns `Irrigation Not Needed` whenever `pump_status == 1` or
  `water_flow > 0`, so it cannot recommend a second watering while water is
  already moving.
- The backend must still enforce minimum interval, maximum daily runtime,
  maximum per-run runtime and a hardware flow cutoff. That code is
  **Member 3's**, and a working example is in `docs/integration_guide.md`.

---

## 13. Engineering limitations

- Trained and tested on Windows + Python 3.11.5, CPU only, 4 cores. The crop
  training time quoted in the README is CPU-specific.
- `torchvision` downloads ImageNet weights on first training or first
  inference. Hosts without internet access need the weights pre-seeded in the
  Torch cache.
- The dataset preparation script fetches PlantVillage parquet row groups from
  Hugging Face. The first run needs internet; the cache in
  `.cache/plantvillage/` makes later runs offline. The `hf_hub` package is not a
  declared dependency precisely to avoid the ~2 GB download; the script uses
  HTTP range requests instead and reads the parquet bytes with `pyarrow`, which
  **is** declared in `requirements.txt`.
- No authentication, rate limiting or TLS **by default**. The service sends no
  CORS headers and accepts no key unless `AGRIGUARD_CORS_ORIGINS` /
  `AGRIGUARD_API_KEY` are set, and it rate-limits nothing unless
  `AGRIGUARD_RATE_LIMIT` is set. That is deliberate: this is a team prototype on
  a LAN behind a reverse proxy, and the perimeter is the network. It is a
  prototype; do not expose it to the public internet as-is.

  When enabled, the API key is compared in constant time and is checked *before*
  the rate limit, so an unauthenticated flood cannot consume a real client's
  budget. The limiter is a per-client token bucket held in process memory, which
  has three consequences worth stating plainly:

  - **It is not shared across workers.** Under `uvicorn --workers 4` each worker
    keeps its own counters, so the effective ceiling is 4x the configured one.
    A shared store (Redis) or an edge limiter is the fix.
  - **It resets on restart.** A crash clears everyone's budget.
  - **It is not authorisation.** One key means every holder sees every plot.
    Per-plot scoping belongs in Member 3's backend, which owns the data model.

  There is deliberately no lockout on repeated wrong keys: a lockout is a
  denial-of-service primitive, since anyone who knows the port could lock the
  service out with six guesses. The rate limit bounds the guessing instead.

  None of this replaces TLS. The key travels in a header in cleartext over HTTP,
  so it is only meaningful behind an HTTPS terminator.
- CORS is available but **off by default**. The service sends no CORS headers
  unless `AGRIGUARD_CORS_ORIGINS` is set to the dashboard's exact origins, which
  is the correct behaviour for a same-origin reverse-proxy deployment. The
  allow-list is never `*`, so a misconfigured origin fails closed rather than
  exposing the prediction endpoints to every site the user visits. CORS is a
  browser control, not an access control: it does nothing for a non-browser
  client, so the missing authentication above is the real limit.
- Storage: this repository keeps the prepared image dataset in
  `data/crop_health/` (~115 MB). It is reproducible, so it is regenerable and
  should not be committed to git.
- No model registry, no experiment tracking, no dataset versioning.

---

## 14. Licence and attribution

- **PlantVillage** images come from the public dataset by
  [spMohanty/PlantVillage-Dataset](https://github.com/spMohanty/PlantVillage-Dataset),
  described in Mohanty, Hughes & Salathe, *PlantVillage: In-the-Wild Plant
  Disease Detection and Classification Framework*, arXiv:1511.08060. The code
  downloads them from the Hugging Face mirror `GVJahnavi/PlantVillage_dataset`.
  **Check the upstream licence before any commercial or public redistribution.**
- Model weights in `models/` are trained in this repository and are part of the
  project.
- If you replace the bundled data with your own plot images, review the licence
  and consent status of those images separately.

---

## 15. Summary - what you may and may not claim

| Claim | Allowed? |
| ----- | -------- |
| "Our irrigation model reached 0.814 balanced accuracy on our synthetic test split" | yes, with "synthetic" in the sentence |
| "Our irrigation model is 78 % accurate in the field" | **no** - the 94 % accuracy figure is on a synthetic test split, and nothing here was measured in a field |
| "Our irrigation model found the right watering time for a tomato plot" | **no** - it reproduces a per-crop moisture trigger, it does not schedule optimally |
| "Our irrigation model uses temperature, humidity and light" | **yes**, inside the range it was trained on. Outside it the model extrapolates badly, so the service hands those readings to the crop threshold rule and says so in `decision_source` |
| "Our irrigation model is a safe, out-of-range-proof predictor" | **no** - it is the opposite: a negative `light` coefficient makes it confidently wrong at the edges, which is why the guard exists |
| "Our crop model classifies leaf images as Healthy or Possible Disease" | yes |
| "Our crop model diagnoses plant diseases" | **no** |
| "Our crop model is N % accurate on real ESP32-CAM images" | **no** - it was never tested on them |
| "The AI decides when to water" | **no** - it recommends; the backend and ESP32 decide |
| "The model runs the pump" | **no** - nothing here touches hardware |
