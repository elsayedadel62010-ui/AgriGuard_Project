# AGRIGUARD HACKATHON QUICK REFERENCE (Final Repository State)

## Architecture (At-a-Glance)
**Sensors → ESP32 → Wi-Fi → Backend (8000) → Database → AI (8001) → Dashboard**

- ESP32 local safety is **authoritative** and independent of Backend/AI/Wi-Fi.
- Dashboard is monitoring/advisory only (not safety-critical).

## Component List (Key)
- ESP32 DevKit-C
- ESP32-CAM (optional)
- Soil (capacitive, analog)
- DHT22
- BH1750 (I²C)
- YF-S201 flow
- Relay + external pump supply (flyback/fuse required)
- LED (220 Ω)
- LCD1602 I²C

## GPIO Map (Authoritative)
| Signal | Pin |
|---|---|
| Soil AOUT | GPIO34 |
| YF-S201 OUT | GPIO18 |
| DHT22 DATA | GPIO27 |
| Relay IN | GPIO26 |
| LED | GPIO25 (via 220 Ω to GND) |
| SDA (BH1750+LCD) | GPIO21 |
| SCL (BH1750+LCD) | GPIO22 |

## Service Ports & URLs
| Service | Host/Port | Notes |
|---|---|---|
| Backend | http://0.0.0.0:8000 (or 127.0.0.1:8000) | `/api/health`, `/api/sensor-data`, `/api/latest-sensor-data`, `/api/irrigation-history`, `/api/water-usage`, `/api/plots` |
| AI | http://127.0.0.1:8001 | `/v1/predict/irrigation` |
| Dashboard | Open `AgriGuard_Frontend/dashboard.html` via local server | Supports `?api=...` override |

## Environment Variables (Backend)
- `AGRIGUARD_AI_URL` (default `http://127.0.0.1:8001`)
- `AGRIGUARD_CORS_ORIGINS` (include dashboard origin)
- `AGRIGUARD_IRRIGATION_EVENTS` (0 to disable; default enabled)

## Firmware Config (sketch.ino)
- `BACKEND_BASE_URL` = `http://192.168.1.4:8000` (change to laptop LAN IP)
- `DEVICE_ID` = `ESP32_01`
- `SENSOR_PATH` = `/api/sensor-data`
- `POST_INTERVAL_MS` = 15000
- Baud 115200

## Startup Sequence
1. Power on ESP32 (pump starts **OFF** - safe)
2. Start AI service (port 8001) if using live AI
3. Start Backend: `uvicorn main:app --host 0.0.0.0 --port 8000` (Backend dir)
4. Open Dashboard (Frontend)
5. Register Plot via Dashboard (device ESP32_01) — required (Backend resolves crop from plot)
6. ESP32 auto-connects Wi-Fi and POSTs every 15 s

## Safety Essentials
- Pump: **external supply only**, common ground, fuse/switch, flyback protection. **VERIFY PHYSICAL MODULE**.
- LCD 5 V in Wokwi: **do not copy**; use 3.3 V only if backpack verified, else bidirectional I²C level shifter.
- Local protections: dry-run ~3 s (no flow when ON) → disable; leak ~3 s (flow >0.5 L/min when OFF) → latch. Independent of network/AI.
- Backend/AI are **advisory only** — never control relay.
- Never fabricate readings; invalid readings not sent.

## Demo Flow
1. System startup
2. Sensor readings update
3. Backend ingests sensor POSTs
4. Dashboard updates (15 s refresh, bounded history)
5. AI decision/advisory shown (source + reason + score when present)
6. Irrigation behavior observed via local logic (safety)
7. Water flow measured (L/min)
8. Irrigation history & water usage update
9. Safety/fail-safe (only where safely demonstrable)

## Troubleshooting Quick
- No data on Dashboard: check Backend `/api/health`, CORS, ESP32 LAN IP
- Prediction unavailable: AI service not running or unreachable (correct, not fabricated)
- Pump doesn't run: check local threshold, dry-run latch, external supply
- Flow reads 0: YF-S201 orientation/pulses; physical calibration may be needed (sim 7.5 pulses/L)
- LCD blank: I²C address/voltage/level shifter

## Pre-Demo Checklist
- [ ] Backend + AI running
- [ ] Plot registered for DEVICE_ID
- [ ] Dashboard loads and shows REAL API DATA
- [ ] Sensor values updating
- [ ] Safety protections understood (local authoritative)
- [ ] Pump power isolated externally with protection
- [ ] LCD voltage compatibility confirmed
- [ ] Relay compatibility confirmed (**VERIFY PHYSICAL MODULE**)
- [ ] YF-S201/soil calibration noted as required if needed
