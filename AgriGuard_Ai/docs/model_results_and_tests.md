# Model Results & Test Documentation

Every number on this page is produced by a script in this repository and stored
in a JSON file under `models/`. Nothing is typed in by hand.

```bash
python scripts/generate_irrigation_data.py --days 30 --period-min 10
python scripts/train_irrigation.py          # -> models/irrigation_*.json
python scripts/evaluate_models.py           # re-reads the stored model files
python -m pytest -q                         # the test suite
```

The crop-health side is measured separately; see
[`crop_health_results.md`](crop_health_results.md).

---

## 1. Read this first: an earlier revision reported a fake 0.93

The first released version of `scripts/generate_irrigation_data.py` reported the
pump as running on the **same row** as the irrigation event:

```python
"water_flow":    water_flow if (pump_steps_left > 0 or irrigated_now) else 0.0,
"pump_status":   1 if (pump_steps_left > 0 or irrigated_now) else 0,
"irrigation_event": 1 if irrigated_now else 0,
```

`irrigated_now` is true exactly on the label row, so every positive row also had
`pump_status == 1`, `water_flow > 0` and therefore
`time_since_last_irrigation_min == 0`. That feature was a **perfect predictor**:
100 % of positives sat at zero and no negative ever did.

The model learned nothing about irrigation. It scored **0.9305 balanced
accuracy** and 0.9951 precision, and the number was printed in the README as a
result.

The generator now reports the pump from the *next* reading onwards, which is both
physically correct (a decision taken at time *t* moves water during *t+1*) and
leak-free. Two regression tests now lock this in:

- `test_irrigation_event_rows_do_not_leak_through_the_pump_state`
- `test_time_since_irrigation_alone_cannot_identify_the_label`

**Rule to carry forward: on data you generated yourself, a suspiciously high
score is a bug report, not a result.** Check that no input column is determined
by the label before believing a number.

Two real modelling bugs surfaced at the same time and are also covered by tests:

| Bug | Symptom | Fix | Test |
| --- | ------- | --- | --- |
| Target leak via pump state | 0.9305 balanced accuracy with no skill | Pump reported one interval later | `test_irrigation_event_rows_do_not_leak_through_the_pump_state` |
| No crop context | One shared moisture term had to serve "water below ~21" (tomato) and "water below ~60" (rice); recall fell to 0.171 | `crop_type` one-hot indicators | `test_rice_is_watered_at_a_much_higher_moisture_than_tomato` |
| Weather readings left out of the model | A 4-feature model scored 0.7788 balanced accuracy against 0.8144 with weather, so the brief's implied exclusion was a real loss, not a safety gain | Weather features added, and the extrapolation risk they create is guarded per-reading instead | `test_weather_actually_moves_the_score_in_the_expected_direction` |
| Training median as the no-history default | A dry reading looked "recently watered" and was waved through | Neutral defaults: `avg30` = current reading | `test_no_history_falls_back_to_neutral_values_not_the_training_median` |
| Best-state check outside the epoch loop (crop side) | Only the last epoch was ever considered for "best validation" | Moved inside the loop | covered by the restored-weights log line |

---

## 2. Irrigation model

Logistic regression, ten features, chronological 80/20 split, seed 42, CPU.

### Features, in order

| # | Feature | Coefficient | Why |
| - | ------- | ----------- | --- |
| 1 | `soil_moisture` | -10.72 | Dominant term: drier means irrigate |
| 2 | `soil_moisture_change` | -0.77 | Falling faster means irrigate sooner |
| 3 | `soil_moisture_avg_30min` | -3.16 | How far the soil has already dipped |
| 4 | `time_since_last_irrigation_min` | +0.18 | Long since a run means the soil has dried out again |
| 5 | `temperature` | +0.38 | Hotter air, more evaporative demand |
| 6 | `humidity` | -0.24 | Drier air, more evaporative demand |
| 7 | `light` | **-1.32** | **Wrong sign** - see `docs/limitations.md` section 4 |
| 8 | `crop_is_tomato` | -4.35 | Per-crop trigger offset |
| 9 | `crop_is_rice` | +8.79 | Rice is kept wetter |
| 10 | `crop_is_pepper` | -4.45 | Per-crop trigger offset |

Intercept -5.92. Coefficients are on standardized features, so they are not
directly comparable in magnitude. `temperature`, `humidity` and `light` **are**
model inputs - the brief requires all four current sensor readings to be, and the
earlier revision of this file that excluded them was corrected. The cost is that
the model can now extrapolate in the wrong direction on weather it has never
seen, which is why the service range-checks those readings and falls back to the
crop threshold rule outside them. Measured failure, guard behaviour and the
coverage gaps are in `docs/limitations.md` section 4.

### Split

| | Rows | Range | Positives |
| - | ---- | ----- | --------- |
| Train | 10,368 | 2026-03-01 .. 2026-03-24 | 887 |
| Test | 2,592 | 2026-03-25 .. 2026-03-30 | 222 |

The test window keeps the natural class balance (8.6 % positives overall), so the
majority-class baseline is **0.9144** accuracy. Report balanced accuracy, not
accuracy: accuracy alone is dominated by the negative class.

### Test results, at the stored operating point

`decision_threshold` = **0.35**, chosen as the F1-optimal cut-off on the last
20 % of the *training* timeline. The test set was not used to pick it.

| | Accuracy | Balanced accuracy | Precision | Recall | F1 |
| - | -------- | ----------------- | --------- | ------- | -- |
| **With history** | 0.9406 | **0.8144** | 0.6504 | 0.6622 | 0.6562 |
| Same model, cut-off 0.50 | 0.9444 | 0.7226 | 0.8145 | 0.4550 | 0.5838 |
| No history (neutral placeholders) | 0.8287 | 0.9002 | 0.3318 | 0.9865 | 0.4966 |
| Majority baseline | 0.9144 | 0.5000 | - | - | - |

Confusion matrices (rows = true, cols = predicted `[0, 1]`):

| Variant | TN | FP | FN | TP |
| ------- | -- | -- | -- | -- |
| With history | 2291 | 79 | 75 | 147 |
| Same model, cut-off 0.50 | 2347 | 23 | 121 | 101 |
| No history | 1929 | 441 | 3 | 219 |

Reading these honestly:

- Balanced accuracy **0.8144** against a 0.5000 chance line is a real signal, not
  a strong model. Precision 0.65 means roughly one in three "irrigate"
  recommendations is unnecessary.
- Without history the model is not much use in the useful direction: recall 0.99
  with precision 0.33 means it recommends water for most readings. This is the
  deliberate failure direction - over-watering wastes water, under-watering kills
  a crop - and it is why the response carries `history_used: false`.
- **The no-history row has the higher balanced accuracy, and that is not an
  improvement.** The model answers almost everything "irrigate" and the class
  weighting rewards it; accuracy, precision and F1 all get worse. Balanced
  accuracy on its own would make omitting history look like a win, so read it
  next to precision.
- The 0.50 cut-off has higher precision and much lower recall. F1 was the tuning
  objective because a false "water now" costs a farmer water and trust, while a
  missed event is still covered by the ESP32's own threshold rule.

### What the model learned

`scripts/decision_boundaries.py` measures the wettest soil moisture the **service**
recommends watering for, under three fixed history contexts, at the training
median weather of 24 C / 68 % RH / 3 653 lux. It asks the service rather than
re-deriving the decision from the coefficients, so the numbers describe a
decision the API would actually make.

| History context | tomato | rice | pepper |
| --- | --- | --- | --- |
| no history (`avg30`=current, `change`=0, `tsl`=1440) | 26.3 % | 61.6 % | 26.1 % |
| recently watered (`avg30`=35, `change`=-2, `tsl`=60) | **20.8 %** | > 51.5 % | **20.5 %** |
| drying fast (`avg30`=30, `change`=-6, `tsl`=1440) | 28.0 % | > 42.0 % | 27.6 % |

Where events actually happened in the training data (median soil moisture at
the 1,109 labelled rows): tomato 20.9 %, rice 56.4 %, pepper 21.1 %.

Two things in that table need explaining before the numbers can be quoted.

**The `> 51.5 %` entries are not measurements.** The two historical features are
derived, not supplied: the service includes the current reading in the 30-minute
window, and measures the trend from the newest prior reading. Holding
`avg30=35` and `change=-2` fixed while sweeping the current reading therefore
forces both prior readings to be functions of that reading, and they only stay
physically sensible in a window around it (1.5..51.5 % here). Rice's trigger sits
above that window, so the sweep ran out of representable context before it found
a crossing. The script prints `> N` for those cells rather than the window
ceiling, because a number taken from the search bounds is not a boundary.

**The `recently watered` row is the one to compare against the medians, and it
matches them almost exactly** - tomato 20.8 % against a 20.9 % event median,
pepper 20.5 % against 21.1 %. That is a much better result than the no-history
row, and it is the reason history is worth sending. The no-history row sits about
5 points above the medians, which is the same over-recommending behaviour
described earlier seen from the other direction: with no history the model cannot
see a dip, so it behaves as if the soil were steady and waters a little early.

Three caveats worth stating plainly:

- The boundary is **not** a property of the crop alone. The same tomato reading is
  judged differently at 28.0 % and 20.8 % depending on the drying trend, and the
  weather moves it further still: at 35 C / 39 % RH / 85 klux the recently watered
  tomato boundary falls to 16.5 %, and at 13 C / 97 % RH / 0 lux it falls to
  19.3 %. Quoting one number per crop as *the* trigger would be misleading.
- The honest summary of the model is **a learned per-crop moisture trigger,
  modulated by drying trend, time since watering and current weather** - closer to
  a scheduler heuristic than to a plant-stress model, and it should not be
  presented as more than that.
- Outside the recorded training range the model does not answer at all; the crop
  threshold rule does. So there is a hard edge in this table at the range
  boundary, and a reading beyond it will not follow any trend in it.

### Why the score is not a percentage

`prediction_score` is a raw logistic output and it saturates - dry readings score
near 1.0 and wet readings near 0.0 - because the fitted coefficients are large.
It is a threshold-crossing and ordering signal. Do not render it as "94 %
confidence" in a dashboard.

---

## 3. Test report

`python -m pytest -q` -> **201 passed, 1 warning** (the warning is a third-party
Starlette/httpx deprecation notice, not a failure in this code).

Every "Actual result" below was produced by running the shipped model files in
`models/` against the input described, not transcribed by hand. Scores are the
raw logistic score for irrigation and the class confidence for crop health.

| Test Case | Expected Result | Actual Result | Correct? | Notes |
| --------- | --------------- | ------------- | -------- | ----- |
| Dry soil, tomato: `soil_moisture=15`, no history | `irrigation_needed: true` | `true`, `prediction_score` `1.0`, `history_used: false`, `decision_source: model` | Yes | No history, so the score saturates and is flagged untrustworthy in `reason` |
| Wet soil, tomato: `soil_moisture=45`, no history | `irrigation_needed: false` | `false`, `prediction_score` `0.0` | Yes | Symmetric saturation, the other end of the sigmoid |
| Pump already running, `pump_status=1` | Never recommend a second start | `false`, `prediction_score` `0.0`, `history_used: null`, `decision_source: pump_guard` | Yes | Safety guard runs before the model; `history_used` is `null`, not `false`, because no model was consulted |
| Dry soil, rice: `soil_moisture=15` | `irrigation_needed: true` | `true`, `prediction_score` `1.0` | Yes | Rice is kept near 55-60 %, so 15 % is far past its trigger |
| Tomato just inside the documented `20.8 %` boundary, with the documented `avg30=35, change=-2, tsl=60` history | `irrigation_needed: true` | `true`, `prediction_score` `0.363` | Yes | The history context is what puts the trigger here: the same reading with no history is answered differently, see the boundary table above |
| Tomato just outside it, `20.9 %`, same history | `irrigation_needed: false` | `false`, `prediction_score` `0.349` | Yes | Confirms the boundary is a measured crossing, not a rounded label, and that `20.8 %` is tight against the 20.9 % training median |
| Same reading, weather outside the training range in the **dangerous** direction: -10 C / 0 % RH / 200 000 lux | Must not be answered by an extrapolating model | `true`, `prediction_score` `0.45`, `decision_source: rule_fallback_out_of_range` | Yes | The model alone scores **0.004** for this reading and would have said "do not irrigate" on soil at 20 %, well past the tomato trigger. That is the measured failure the guard exists for; the model's score is quoted in `reason` for diagnosis only |
| Same reading, weather outside the range in the **benign** direction: 45 C / 10 % RH / 120 000 lux | Guard is conservative, not a fix for a measured disagreement | `true`, `prediction_score` `0.45`, `decision_source: rule_fallback_out_of_range` | Yes | The model alone scores 0.768 here and would have recommended watering anyway, so the guard changes only who is credited with the decision, not the answer. Recorded so the guard is not mistaken for a targeted fix |
| Same reading, weather inside the training range: 24 C / 68 % RH / 3 653 lux | Answered by the model | `true`, `prediction_score` `0.988`, `decision_source: model` | Yes | The guard only fires where the model would actually be extrapolating, so ordinary readings keep the full model |
| Score/decision agreement across all branches | `irrigation_needed == (prediction_score >= 0.35)` | 0 disagreements in 132 cases (3 crops x 11 moisture levels x 4 weather conditions) | Yes | Regression-tested; catches the old `1 - score` inversion, the guards' 0.75-beside-"do not water", and the fallback that scored both branches 0.5-0.95 |
| `last_irrigation_at` sent with no request `timestamp` | Field is honoured, or its absence is explained | `history_used: false` and `reason` says `last_irrigation_at was supplied but ignored ... send timestamp to use it` | Yes | Previously dropped silently; a client could not tell a dropped field from a missing one |
| Real unseen healthy leaf, `healthy/apple___healthy__00004.jpg` | `label: "Healthy"` | `Healthy`, confidence `0.9755` | Yes | Test-split image, never trained on |
| Real unseen diseased leaf, `possible_disease/apple__apple_scab__04218.jpg` | `label: "Possible Disease"` | `Possible Disease`, confidence `0.9925` | Yes | Two-class screen only; it does not name the disease |
| Healthy leaf outside the training set of species | Not claimed as valid | `Possible Disease` or `Healthy` per the model, with the species noted as untrained | No (limitation) | A two-class screen over 14 species cannot be right everywhere; `docs/crop_health_results.md` reports the measured per-class rates instead |

Two rows are there to record a known failure rather than a success. The
out-of-range rows show the model is demonstrably wrong in that regime and that
the service declines to use it; the crop-health row is the honest failure of a
screening model, left visible on purpose. A screening model is expected to produce
false positives and false negatives, and the per-class precision and recall in
section 1 are the honest way to express that risk.

## 4. Test suite by area

```bash
python -m pytest -q
```

| Area | What is covered |
| ---- | --------------- |
| Leak guards | No positive row carries a running pump or a zero time-since value |
| Crop context | One-hot encoding, unknown crops, rice vs tomato at the same moisture |
| Weather features | All four current readings are model inputs; `temperature` and `humidity` move the score in the agronomically expected direction; the negative `light` coefficient is pinned as a known weakness rather than left to be rediscovered |
| Out-of-range guard | A reading outside the recorded training range hands the decision to the crop threshold rule, names the offending reading and its bounds, and is reported in `decision_source`; in-range readings still reach the model so history sensitivity is not flattened |
| Safety | Dry soil is always recommended across the whole accepted weather range; pump-running and flowing-water guards win |
| No-history defaults | Neutral placeholders, no NaN, `history_used: false` |
| Operating point | The stored threshold is used, and old files default to 0.50 |
| Stale artefacts | A v0.1.0 model file (trained on the four current readings) is rejected with a retrain message instead of being mis-scored |
| Documented boundaries | The per-crop moisture boundaries quoted in this file are re-measured by importing `scripts/decision_boundaries.py` itself, so the docs cannot describe a decision the service would not make |
| History contexts | The boundary script proves each context against the service's own feature builder before sweeping, and `feasible_window` bounds the sweep so a pinned context is never measured against an impossible history. A context the service cannot produce raises instead of printing a number |
| Crop metric provenance | The README's condensed honesty checklist is checked against `models/irrigation_model.json`, so a copy of the limitations table cannot drift back to the retracted leak-era figure |
| Crop checkpointing | `best_validation_accuracy` equals the maximum over the training history, and the evaluation reproduces the training metrics from the saved weights |
| Dataset paths | `CROP_HEALTH_DIR` is defined once as a `Path`; two scripts previously imported a shadowing string and crashed |
| Contract | Exact response keys, 422 on invalid input, advisory wording, and the score/decision identity on every branch including the fallbacks |
| Documented examples | The request/response quoted in the README and `docs/api_contract.md` is replayed against the live service, so the quoted score cannot drift |
| Crop preprocessing | Transform parity with training, `weights_only=True` loading |
| Dataset prep | Cache correctness, dHash grouping, no duplicate group across splits |
| Docs vs code | Every `test_...` name cited in the Markdown exists; current metrics are present and superseded ones are absent; the quoted test count matches what pytest collects |

The suite runs against the real trained models in `models/`, not against mocks,
so a model file that fails to load turns into a visible failure rather than a
silently degraded service. `tests/test_docs_consistency.py` is the part that
keeps this file honest: it reads the Markdown, resolves every test name it
cites, and compares the numbers here against `models/*.json`.

## 5. Requirement audit

```bash
python scripts/audit_stage3.py
```

The test suite proves the code behaves; this proves the deliverable is complete.
It loads the shipped artefacts, calls the real service, regenerates the boundary
numbers and runs the suite, then prints one line per Stage 3 requirement and
exits non-zero if any fail. Current state: **21/21 checks passed**. It exists
because "all the tests pass" and "everything asked for was delivered" are
different claims, and only the first one is automatic.
