"""Derive `IrrigationEvent` rows from the pump state the firmware already sends.

Phase 2B. Why this module exists
--------------------------------
`IrrigationEvent` was reachable only through `POST /api/irrigation-events`, and
nothing in the project ever called it: not the ESP32 firmware, not the
dashboard. So `/api/irrigation-history` always returned `[]` and
`/api/water-usage` always returned zero, while the firmware was already
transmitting everything needed to fill them in - `pump_status` and `water_flow`
on every reading.

The alternative was a second POST from the firmware, which would have changed
the Phase 2A contract that has already been compile- and E2E-verified. Deriving
the event server-side keeps the firmware untouched.

How an event is derived
-----------------------
From the transitions already present in the readings a device sent:

    off -> on   open an event at the timestamp of the reading that started the pump
    on  -> off  close it: `duration_seconds` from the two timestamps,
                    `water_used` from the measured flow integrated over them

`water_used` is litres, integrated from the L/min the flow sensor actually
reported (`litres = L/min * minutes`) using the same reading timestamps as the
duration. It is therefore a measurement over the intervals that were really
observed, not an estimate of a quantity the device never transmitted.

Accuracy limits, stated plainly
-------------------------------
* Integration uses the PREVIOUS reading's flow rate over the interval that
  follows it. With the firmware's 15 s cadence the error is small but non-zero,
  and the first reading of a pump cycle contributes no interval.
* `MAX_GAP_SECONDS` bounds how far apart two consecutive readings may be before
  integration stops. The firmware posts every 15 s; a gap that large means the
  device was offline or restarting, and carrying the flow rate across it would
  credit litres nobody measured. The event is still closed, with the water
  accumulated up to that point.
* Open-event state is in memory. If the backend restarts mid-irrigation, the
  next OFF transition rebuilds the start from the previous ON reading's
  timestamp instead of dropping the event. A restart that leaves the pump ON
  forever leaves no event at all, which is the honest outcome: nothing observed
  the pump stopping.

Set `AGRIGUARD_IRRIGATION_EVENTS=0` to disable derivation entirely; the
endpoint keeps working and simply stays empty, exactly as it did before.
"""

import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from sqlmodel import Session

from models import IrrigationEvent

#: Two readings further apart than this are not integrated over the gap.
MAX_GAP_SECONDS = 300

#: Escape hatch for a live demo. Default on.
ENABLED = os.getenv("AGRIGUARD_IRRIGATION_EVENTS", "1").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
)

#: device_id -> {"start_time": datetime, "litres": float} for pumps currently on.
_open_events: Dict[str, Dict[str, Any]] = {}


def _aware(value: datetime) -> datetime:
    """UTC-aware copy of a timestamp.

    SQLite hands back naive datetimes, but the readings are stamped with
    `datetime.now(timezone.utc)`. Comparing a naive value against an aware one
    raises TypeError, so the offset is restored here rather than at every call
    site. A naive value is assumed to be UTC, which is what was written.
    """
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def reset_state() -> None:
    """Forget every open event. For tests and for a clean start."""
    _open_events.clear()


def observe_pump_state(
    session: Session,
    device_id: str,
    previous: Optional[Any],
    current: Any,
):
    """Record the effect of one reading's pump state on irrigation history.

    `previous` and `current` are `SensorReading` rows for the same device, or
    `None` for `previous` when this is the device's first reading. The new
    reading must already be committed before this is called, so that the two
    cannot be confused when history is read back.

    Returns a summary dict describing what changed, or `None` when nothing did.
    The caller is responsible for handling exceptions: a failure to write
    history must never turn a stored measurement into a failed request.
    """
    if not ENABLED:
        return None

    now = _aware(current.timestamp)
    state = _open_events.get(device_id)

    if state is None and previous is not None and bool(previous.pump_status):
        # Backend restarted (or state was cleared) while the pump was running.
        # Rebuild the start from the previous reading rather than losing the
        # event; the water is reconstructed from the readings that follow.
        state = {
            "start_time": _aware(previous.timestamp),
            "litres": 0.0,
        }

    if bool(current.pump_status):
        if state is None:
            _open_events[device_id] = {
                "start_time": now,
                "litres": 0.0,
            }
            return {
                "action": "opened",
                "device_id": device_id,
                "start_time": now.isoformat(),
            }
        # Pump still running. Keep the original start; accumulate the interval
        # that just ended so a crash before the OFF transition loses less.
        if previous is not None:
            state["litres"] += _integrated_litres(previous, now)
        _open_events[device_id] = state
        return None

    # Pump is off now. Nothing to close if it was never seen on.
    if state is None:
        return None

    litres = state["litres"]
    integrated = False
    if previous is not None:
        contribution = _integrated_litres(previous, now)
        if contribution is not None:
            litres += contribution
            integrated = True

    duration_seconds = int(round(max(0.0, (now - state["start_time"]).total_seconds())))
    _open_events.pop(device_id, None)

    event = IrrigationEvent(
        start_time=state["start_time"],
        duration_seconds=duration_seconds,
        water_used=round(litres, 3),
    )
    session.add(event)
    session.commit()
    session.refresh(event)

    return {
        "action": "closed",
        "device_id": device_id,
        "event_id": event.id,
        "start_time": state["start_time"].isoformat(),
        "duration_seconds": duration_seconds,
        "water_used": event.water_used,
        "flow_integrated": integrated,
    }


def _integrated_litres(previous, now: datetime) -> Optional[float]:
    """Litres passed between `previous` and `now`, or None if the gap is unusable."""
    span = (now - _aware(previous.timestamp)).total_seconds()
    if span <= 0 or span > MAX_GAP_SECONDS:
        return None
    try:
        flow = float(previous.water_flow or 0.0)
    except (TypeError, ValueError):
        return None
    if flow < 0:
        return None
    return flow * (span / 60.0)