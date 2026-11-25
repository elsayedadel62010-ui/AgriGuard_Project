"""Data preprocessing for AgriGuard.

This module is deliberately the ONLY place where raw readings are turned into
model inputs, so training, validation, testing and live inference can never
drift apart (TASK 8 of the brief).

Irrigation pipeline
-------------------
    load_irrigation_csv()          # load + validate + clean + add history
        -> chronological_split()   # 80 % train / 20 % test, no shuffling
        -> train_model()           # scaler fitted on TRAIN rows only

    build_features_from_request()  # the same feature builder for a live reading

Historical features
-------------------
    soil_moisture_change            soil_moisture[t] - soil_moisture[t-1]
    soil_moisture_avg_30min         mean of soil_moisture over (t-30min, t]
    time_since_last_irrigation_min  minutes since the last observed pump run

All three are computed with a backward-looking window only, so nothing that
happens *after* the prediction timestamp can leak into the input. They are
computed per (device_id, plot_id) group, exactly as they would be at runtime.
They need *sequential* readings, so with the synthetic prototype CSV they are
partly meaningless; with real ESP32 logs they become informative.
"""

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# dataset contract
# ---------------------------------------------------------------------------
CANONICAL_COLUMNS: List[str] = [
    "timestamp",
    "device_id",
    "plot_id",
    "soil_moisture",
    "temperature",
    "humidity",
    "light",
    "npk_n",
    "npk_p",
    "npk_k",
    "water_flow",
    "pump_status",
    "irrigation_event",
]

REQUIRED_COLUMNS: List[str] = [
    "timestamp",
    "soil_moisture",
    "temperature",
    "humidity",
    "light",
    "irrigation_event",
]

#: Columns the CSV must have; the optional ones are added if they are absent.
OPTIONAL_COLUMNS: List[str] = [
    "device_id",
    "plot_id",
    "npk_n",
    "npk_p",
    "npk_k",
    "water_flow",
    "pump_status",
    "crop_type",
]

TARGET_COLUMN = "irrigation_event"

#: Plausibility ranges used to drop obviously broken rows. These are the values
#: documented in docs/dataset_structure.md - they are NOT sensor calibration.
SENSOR_RANGES: Dict[str, Tuple[float, float]] = {
    "soil_moisture": (0.0, 100.0),
    "temperature": (-40.0, 80.0),
    "humidity": (0.0, 100.0),
    "light": (0.0, 200000.0),
    "water_flow": (0.0, 100000.0),
    "npk_n": (0.0, 5000.0),
    "npk_p": (0.0, 5000.0),
    "npk_k": (0.0, 5000.0),
}

HISTORICAL_FEATURES: List[str] = [
    "soil_moisture_change",
    "soil_moisture_avg_30min",
    "time_since_last_irrigation_min",
]

#: Backward-looking window for the moving average.
AVG_WINDOW = "30min"

#: A reading with a non-zero flow counts as "the pump ran".
FLOW_ACTIVE_MIN = 0.0

#: Cap for time_since_last_irrigation_min. Beyond this the feature is flat: it
#: only means "no recent irrigation". Keeps the cap explicit and documented.
MAX_MINUTES_SINCE_IRRIGATION = 1440.0

#: The three weather readings the brief requires the model to consume. They are
#: real model inputs, not decoration.
#:
#: An earlier revision kept them OUT of the model and justified that by
#: claiming it cost about 0.05 balanced accuracy. That was wrong in the
#: direction that mattered: re-measured on the same chronological split, adding
#: them made the model BETTER, not worse -
#:
#:     without weather: accuracy 0.9352  balanced 0.7788  recall 0.5901
#:     with weather:    accuracy 0.9406  balanced 0.8144  recall 0.6622
#:
#: and their standardised coefficients are agronomically correct in sign:
#: humidity -10.7 (drier air -> more water needed), light +0.38, temperature
#: +0.18, soil_moisture -3.2. A humidity sweep at 20 % soil moisture falls from
#: 0.7236 at 10 % RH to 0.4400 at 90 % RH, which is the direction an agronomist
#: would expect.
#:
#: The real cost of including them is EXTRAPOLATION, not accuracy, and the
#: coefficient signs explain why: the raw (standardised) light coefficient is
#: -1.32, so a brightness the model never saw drives the score DOWN - the wrong
#: direction for a dry plot. Measured failure: tomato at 16 % soil moisture with
#: -10 C / 0 % RH / 200 000 lux scores 0.08, i.e. "do not irrigate", when the
#: trigger is 20.8 %. :data:`CLIENT_SUPPLIED_FEATURES` in
#: :mod:`agriguard.irrigation` is range-checked against the ranges recorded at
#: training time, and the crop threshold rule answers instead. See
#: docs/limitations.md.
WEATHER_FEATURES: List[str] = ["temperature", "humidity", "light"]

#: Historical alias for :data:`WEATHER_FEATURES`. Nothing in the model is unused
#: any more, so the name is kept only so existing imports and reports do not
#: break; new code should use :data:`WEATHER_FEATURES`. "unused" is a claim about
#: a model this project no longer ships.
UNUSED_WEATHER_COLUMNS: List[str] = list(WEATHER_FEATURES)

#: The numeric part of the feature vector, in order. `soil_moisture` first
#: because it is the quantity that gates the pump, then the three historical
#: features, then the current weather readings.
NUMERIC_FEATURE_COLUMNS: List[str] = (
    ["soil_moisture"] + HISTORICAL_FEATURES + WEATHER_FEATURES
)

#: Crops the irrigation model was trained on, one-hot encoded.
#:
#: The soil-moisture trigger point is crop specific: rice is kept near 60 %
#: while tomato and pepper trigger near 21-25 %. A model that only sees
#: ``soil_moisture`` must satisfy "water below ~21" and "water below ~60" with
#: one shared linear term, which makes the two clusters of the same label
#: inseparable and tanks recall. The crop is known per plot on the ESP32, so
#: giving the model that context is legitimate - it is observable context, not
#: a leaked label.
CROP_FEATURES: List[str] = ["crop_is_tomato", "crop_is_rice", "crop_is_pepper"]

#: Full feature vector, in order. The order here is the contract between
#: training and inference; the model JSON stores it by name.
FEATURE_COLUMNS: List[str] = NUMERIC_FEATURE_COLUMNS + CROP_FEATURES


def crop_indicators(crop_type: Optional[str]) -> List[float]:
    """One-hot encode ``crop_type`` over :data:`CROP_FEATURES`.

    A crop that was not in the training set yields all zeros: nothing is
    invented, and the caller is expected to flag such predictions as
    extrapolation rather than trust them.
    """
    name = (crop_type or "").strip().lower()
    return [
        1.0 if name == column[len("crop_is_") :] else 0.0
        for column in CROP_FEATURES
    ]


HISTORICAL_DEFAULTS: Dict[str, float] = {
    "soil_moisture_change": 0.0,
    "soil_moisture_avg_30min": 0.0,
    "time_since_last_irrigation_min": MAX_MINUTES_SINCE_IRRIGATION,
}

GROUP_COLUMNS = ["device_id", "plot_id"]


class MissingColumnsError(ValueError):
    """Raised when the CSV does not contain the columns the model needs."""

    def __init__(self, missing: Iterable[str]):
        self.missing = sorted(set(missing))
        super().__init__(
            "CSV is missing required column(s): " + ", ".join(self.missing)
        )


# ---------------------------------------------------------------------------
# validation / cleaning
# ---------------------------------------------------------------------------
def validate_columns(df: pd.DataFrame) -> None:
    """Raise MissingColumnsError if a required column is absent."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise MissingColumnsError(missing)


def add_missing_optional_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add optional columns with sensible defaults so downstream code is uniform."""
    df = df.copy()
    defaults = {
        "device_id": "ESP32_UNKNOWN",
        "plot_id": "PLOT_UNKNOWN",
        "npk_n": np.nan,
        "npk_p": np.nan,
        "npk_k": np.nan,
        "water_flow": 0.0,
        "pump_status": 0,
        "crop_type": "default",
    }
    for column, default in defaults.items():
        if column not in df.columns:
            df[column] = default
    return df


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the numeric columns safely; anything unparsable becomes NaN."""
    df = df.copy()
    for column in list(SENSOR_RANGES) + ["pump_status", TARGET_COLUMN]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def parse_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    return df


def clean_irrigation_frame(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Drop unusable rows, clip implausible values, return (frame, report)."""
    validate_columns(df)
    df = add_missing_optional_columns(df)
    df = parse_timestamps(df)
    df = coerce_numeric(df)

    report = {
        "rows_in": int(len(df)),
        "rows_bad_timestamp": int(df["timestamp"].isna().sum()),
        "rows_dropped": 0,
        "values_clipped": 0,
    }

    df = df.dropna(subset=["timestamp"])
    df = df.dropna(subset=["soil_moisture", "temperature", "humidity", "light"])
    df = df.dropna(subset=[TARGET_COLUMN])

    # a boolean flag is only 0 or 1
    df = df[df[TARGET_COLUMN].isin([0, 1])]
    df = df[df["pump_status"].isin([0, 1])]
    report["rows_dropped"] = report["rows_in"] - int(len(df))

    for column, (low, high) in SENSOR_RANGES.items():
        if column not in df.columns:
            continue
        below = int((df[column] < low).sum())
        above = int((df[column] > high).sum())
        report["values_clipped"] += below + above
        df[column] = df[column].clip(low, high)

    df["pump_status"] = df["pump_status"].astype(int)
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    # NPK is optional: fill the gaps with the per-column median so the column is
    # usable, and remember the median so inference can do the same. NPK is not a
    # model input, so this only keeps the stored dataset tidy.
    for column in ("npk_n", "npk_p", "npk_k"):
        if column in df.columns and df[column].notna().any():
            df[column] = df[column].fillna(df[column].median())

    # kind="stable" matters: several devices share a timestamp and the order
    # of tied rows must not change between runs.
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    report["rows_out"] = int(len(df))
    return df, report


# ---------------------------------------------------------------------------
# historical features (backward looking only)
# ---------------------------------------------------------------------------
def _group_keys(df: pd.DataFrame) -> List[pd.Series]:
    keys = [df["device_id"].astype(str) + "|" + df["plot_id"].astype(str)]
    return keys


def add_historical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the three historical features using only information available at time t.

    The input frame must already be sorted by timestamp. Every window looks
    backwards only, so a row's features never depend on a later reading.
    """
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True).copy()
    df["soil_moisture_change"] = 0.0
    df["soil_moisture_avg_30min"] = np.nan
    df["time_since_last_irrigation_min"] = np.nan

    window = pd.Timedelta(AVG_WINDOW)
    for _, indices in df.groupby(_group_keys(df)[0], sort=False).groups.items():
        block = df.loc[indices].sort_values("timestamp", kind="stable")

        # 1. change since the previous reading of the same plot
        df.loc[block.index, "soil_moisture_change"] = (
            block["soil_moisture"].diff().fillna(0.0).clip(-100.0, 100.0).to_numpy()
        )

        # 2. moving average over (t - 30min, t]
        df.loc[block.index, "soil_moisture_avg_30min"] = (
            block.set_index("timestamp")["soil_moisture"]
            .rolling(window, closed="right", min_periods=1)
            .mean()
            .to_numpy()
        )

        # 3. minutes since the last observed pump run.
        #    NOTE: uses pump_status / water_flow (observable), NEVER the
        #    irrigation_event label - that would be target leakage.
        ran = (
            (block["pump_status"] == 1) | (block["water_flow"] > FLOW_ACTIVE_MIN)
        )
        stamps = block["timestamp"].where(ran)

        elapsed = pd.Series(np.nan, index=block.index, dtype=float)
        last_seen = stamps.ffill()
        previous = last_seen.shift(1)
        gap = (block["timestamp"] - previous).dt.total_seconds() / 60.0
        elapsed = gap.where(previous.notna(), MAX_MINUTES_SINCE_IRRIGATION)
        elapsed[ran.to_numpy()] = 0.0  # a pump running right now means "0 min ago"
        df.loc[block.index, "time_since_last_irrigation_min"] = (
            elapsed.fillna(MAX_MINUTES_SINCE_IRRIGATION)
            .clip(0.0, MAX_MINUTES_SINCE_IRRIGATION)
            .to_numpy()
        )

    df["soil_moisture_avg_30min"] = df["soil_moisture_avg_30min"].fillna(
        df["soil_moisture"]
    )
    df["time_since_last_irrigation_min"] = df[
        "time_since_last_irrigation_min"
    ].fillna(MAX_MINUTES_SINCE_IRRIGATION)
    return df


def load_irrigation_csv(path: Path) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Load, validate and clean an irrigation CSV. Returns (frame, report)."""
    df = pd.read_csv(path)
    frame, report = clean_irrigation_frame(df)
    frame = add_historical_features(frame)
    return frame, report


# ---------------------------------------------------------------------------
# split / scale
# ---------------------------------------------------------------------------
def chronological_split(
    df: pd.DataFrame, test_fraction: float = 0.20
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """First 80 % of the timeline is training, the last 20 % is testing.

    No shuffling: a random split would let the model be evaluated on readings
    that happen *before* its training readings, which is data leakage for a
    time series.
    """
    ordered = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    cut = int(len(ordered) * (1.0 - test_fraction))
    return ordered.iloc[:cut].copy(), ordered.iloc[cut:].copy()


def scale_features(
    x: np.ndarray, means: np.ndarray, stds: np.ndarray
) -> np.ndarray:
    """Standardize a feature matrix/vector with statistics from the train set."""
    return (np.asarray(x, dtype=float) - means) / (stds + 1e-9)


def build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """Training-time feature matrix: numeric block, then the crop indicators."""
    numeric = df[NUMERIC_FEATURE_COLUMNS].to_numpy(dtype=float)
    if "crop_type" in df.columns:
        crops = np.asarray([crop_indicators(v) for v in df["crop_type"]], dtype=float)
    else:
        crops = np.zeros((len(df), len(CROP_FEATURES)), dtype=float)
    return np.column_stack([numeric, crops])


# ---------------------------------------------------------------------------
# the SAME feature builder, used for a live API request
# ---------------------------------------------------------------------------
def build_features_from_request(
    req,
    previous_soil_moisture: Optional[float] = None,
    recent_soil_moisture: Optional[Iterable[Tuple[str, float]]] = None,
    last_irrigation_minutes_ago: Optional[float] = None,
    current_timestamp: Optional[pd.Timestamp] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Turn one live request into the model's feature vector.

    This is the inference-time twin of add_historical_features(): same feature
    order, same window, same cap, same defaults. That is what keeps training
    and inference honest.

    Returns (features, notes) where `notes` records any historical feature that
    had to fall back to a default because the backend sent no history.
    """
    now = current_timestamp
    if now is None:
        now = pd.Timestamp.utcnow().tz_localize(None)

    notes: List[str] = []

    if recent_soil_moisture is not None:
        history = [
            (pd.Timestamp(ts).tz_localize(None), float(value))
            for ts, value in recent_soil_moisture
        ]
        history.sort(key=lambda item: item[0])
        window_start = now - pd.Timedelta(AVG_WINDOW)
        window = [value for stamp, value in history if window_start < stamp <= now]
        if window:
            average = float(np.mean(window + [req.soil_moisture]))
        else:
            average = req.soil_moisture
            notes.append("no reading inside the 30-minute window; using current value")
    else:
        # Neutral, not a training median. "No history" means we cannot see a
        # dip, so the honest guess is "no dip": average = the current reading.
        # Substituting the training median instead pushed dry readings toward
        # "do not irrigate", because the median (~33 %) is much wetter than a
        # dry reading and the model reads a high average as "already watered".
        average = float(req.soil_moisture)
        notes.append("no history supplied; soil_moisture_avg_30min is a placeholder")

    if previous_soil_moisture is not None:
        change = float(req.soil_moisture) - float(previous_soil_moisture)
    else:
        # No previous reading means "no observed trend", not "the soil jumped".
        change = float(HISTORICAL_DEFAULTS["soil_moisture_change"])
        notes.append("no previous reading; soil_moisture_change is a placeholder")

    if last_irrigation_minutes_ago is not None:
        since = float(last_irrigation_minutes_ago)
    else:
        # Nothing recent is the safe assumption: the cap means "no recent run",
        # which never suppresses a recommendation.
        since = float(HISTORICAL_DEFAULTS["time_since_last_irrigation_min"])
        notes.append("no irrigation history; time_since_last_irrigation_min is a placeholder")

    values = {
        "soil_moisture": float(req.soil_moisture),
        "soil_moisture_change": change,
        "soil_moisture_avg_30min": average,
        "time_since_last_irrigation_min": since,
        "temperature": float(req.temperature),
        "humidity": float(req.humidity),
        "light": float(req.light),
    }
    ordered = [values[name] for name in NUMERIC_FEATURE_COLUMNS]
    ordered.extend(crop_indicators(getattr(req, "crop_type", None)))
    return np.asarray(ordered, dtype=float), notes


def fill_historical_defaults(
    features: np.ndarray, medians: Optional[Dict[str, float]]
) -> np.ndarray:
    """Replace NaN historical features with the neutral fallback values.

    This is a last-resort net for rows that reach inference still holding NaN;
    `build_features_from_request` normally fills them in with neutral
    placeholders earlier (see HISTORICAL_DEFAULTS). A stored training median is
    used only if one is supplied, which is not the behaviour we want by default -
    see the note in build_features_from_request for why.
    """
    filled = np.array(features, dtype=float, copy=True)
    for index, name in enumerate(FEATURE_COLUMNS):
        if np.isnan(filled[index]):
            default = (medians or {}).get(name, HISTORICAL_DEFAULTS.get(name, 0.0))
            filled[index] = float(default)
    return filled


def clamp(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))
