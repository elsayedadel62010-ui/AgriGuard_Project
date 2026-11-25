# AGRIGUARD PHYSICAL PROTOTYPE ASSEMBLY GUIDE (Final Repository State)

> This guide is derived directly from the AgriGuard repository. Every hardware value listed is traceable to actual files (`AgriGuard_Hardware/sketch.ino`, `diagram.json`, `libraries.txt`, `yf-s201.chip.*`). Any characteristic that cannot be verified from these files without physical measurement is explicitly marked **VERIFY PHYSICAL MODULE**.

## 1. PROJECT OVERVIEW

AgriGuard is a smart irrigation and crop-health monitoring prototype. The system architecture is:

**Sensors → ESP32 → Wi-Fi → Backend → Database → AI → Dashboard**

The ESP32's local safety protections remain independent and authoritative. The Dashboard is for monitoring/visualization and advisory information; it is **not** safety-critical.

## 2. SYSTEM ARCHITECTURE

| Layer | Role | Notes |
|---|---|---|
| Sensors | Soil, climate, light, water flow | Physical sensors connected to ESP32 as per firmware |
| ESP32 | Acquisition, local safety, LCD display, relay control | Local fail-safes always operate; Backend/AI are advisory only |
| Wi-Fi | Network transport | 2.4 GHz Wi-Fi network accessible to ESP32 and Backend |
| Backend (FastAPI) | Ingestion, persistence, AI call, events | Resolves crop from Plot; validates device_id |
| Database (SQLite) | Sensor readings, plots, irrigation events | Schema unchanged |
| AI Service | Crop-aware irrigation/advisory | Port 8001 by default |
| Frontend (Dashboard) | Visualization, monitoring, history, plot registration | Bounded history requests, escaped HTML |

## 3. COMPLETE HARDWARE COMPONENTS

| Component | Quantity | Verification Status |
|---|---|---|
| ESP32 DevKit-C | 1 | VERIFIED IN CODE |
| ESP32-CAM (optional/if present) | 1 | PHYSICAL VERIFICATION REQUIRED |
| Capacitive soil moisture sensor (analog) | 1 | PHYSICAL VERIFICATION REQUIRED |
| DHT22 | 1 | PHYSICAL VERIFICATION REQUIRED |
| BH1750 light sensor (I²C) | 1 | VERIFIED IN CODE |
| YF-S201 water flow sensor | 1 | PHYSICAL VERIFICATION REQUIRED |
| Relay module (suitable for pump) | 1 | **VERIFY PHYSICAL MODULE** (coil/contacts, voltage/current) |
| Water pump | 1 | **VERIFY PHYSICAL MODULE** (rating, power supply) |
| External power supply for pump | 1 | **VERIFY PHYSICAL MODULE** (fuse/protection, flyback) |
| LED (red) | 1 | VERIFIED IN CODE |
| 220 Ω resistor | 1 | VERIFIED IN CODE |
| LCD1602 I²C (16x2) | 1 | **VERIFY PHYSICAL MODULE** (I²C address/backpack voltage) |
| Jumper wires, breadboard, connectors | As needed | PHYSICAL VERIFICATION REQUIRED |

## 4. ESP32 PIN MAPPING (AUTHORITATIVE)

Source: `AgriGuard_Hardware/sketch.ino`. This mapping **must not be changed**.

| Signal | Pin | Notes |
|---|---|---|
| Soil moisture AOUT | **GPIO34** | Analog input (0–4095) |
| YF-S201 OUT | **GPIO18** | Interrupt-driven, INPUT_PULLUP |
| DHT22 DATA | **GPIO27** | Single-wire |
| Relay IN | **GPIO26** | Controls pump relay/driver |
| LED | **GPIO25** | Through 220 Ω to GND |
| I²C SDA | **GPIO21** | Shared by BH1750 and LCD |
| I²C SCL | **GPIO22** | Shared by BH1750 and LCD |

**No other GPIOs are re-assigned in firmware.**

## 5. SENSOR CONNECTIONS

| Sensor | Power | Data | Notes |
|---|---|---|---|
| Soil (capacitive) | 3.3V | AOUT → GPIO34 | Analog direction and calibration: **PHYSICAL VERIFICATION REQUIRED** |
| DHT22 | 3.3V (recommended) | DATA → GPIO27 | Pull-up requirements: **VERIFY PHYSICAL MODULE** |
| BH1750 | 3.3V | SDA→GPIO21, SCL→GPIO22 | I²C address handled by library |
| YF-S201 | 5V/3.3V per module | OUT → GPIO18 | K-factor in simulation 7.5 pulses/L; **physical calibration required** (not assumed) |

## 6. ACTUATOR / PUMP / RELAY CONNECTIONS

**Critical:** Pump **must not** be powered from ESP32.

- Relay IN → GPIO26 (active as per firmware; polarity verified in code behavior)
- LED → GPIO25 through 220 Ω → GND
- Pump: External supply with common ground, appropriate fuse/switch, and flyback protection. **VERIFY PHYSICAL MODULE** for voltage/current compatibility of relay contacts/coils.
- Separation from unsafe mains voltage work: do not provide unsafe mains wiring in this guide.

## 7. ESP32-CAM CONNECTIONS

If present, verify actual ESP32-CAM integration in repository. Capture, endpoint, model, and failure behavior must be confirmed from actual files if implemented. If incomplete, software completion documented; physical behavior is **PHYSICAL VERIFICATION REQUIRED**.

## 8. POWER SUPPLY AND POWER DISTRIBUTION

- ESP32: USB or regulated 5V via VIN/3.3V logic
- Sensors: 3.3V where specified
- Relay coil: per relay module spec (**VERIFY PHYSICAL MODULE**)
- Pump: **external supply only**; ensure common GND between logic and external supply domain as appropriate

## 9. COMPLETE WIRING DIAGRAM

Reference `AgriGuard_Hardware/diagram.json`. **Important:** The Wokwi diagram shows LCD VCC at **5 V**. **Do not copy this blindly to the physical prototype.** Use **3.3 V** only if the actual LCD I²C backpack is verified compatible; otherwise use an appropriate **bidirectional I²C level shifter**. Diagram and physical must remain synchronized with firmware (GPIO map unchanged).

## 10. FIRMWARE / ESP32 LOGIC

Source: `sketch.ino` (Phase 2A transport, advisory Backend).

- Loop cadence 1 Hz; climate read ~2.5 s; POST every 15 s
- Flow: pulse counting with interrupt permanently attached; window-based L/min calculation; total litres integrated
- **Local safety (authoritative):**
  - Temporary local rule: `soil_moisture < TEMP_LOCAL_DRY_THRESHOLD = 40` (test-only; crop decisions come from Backend/AI)
  - Dry-run protection: pump ON but no flow for ~3 s (DRY_RUN_LIMIT=3) → relay disabled
  - Leak protection: pump OFF but flow > LEAK_FLOW_THRESHOLD (0.5 L/min) for ~3 s (LEAK_LIMIT=3) → latch alarm
- Wi-Fi/Backend failures: do not disable local safety. Transport decoupled to Core 0; safety on Core 1
- Safe startup: relay and LED OFF; sensors initialized; Wi-Fi connects asynchronously
- Payload matches Backend schema; invalid readings not sent (never fabricated)
- Serial @115200 for diagnostics

## 11. BACKEND AND DASHBOARD DATA FLOW

- ESP32 POSTs to `BACKEND_BASE_URL + SENSOR_PATH` = `/api/sensor-data` (JSON)
- Backend persists SensorReading; derives irrigation events from pump transitions (bounded gaps MAX_GAP_SECONDS=300; no fabricated litres; kill switch `AGRIGUARD_IRRIGATION_EVENTS=0`)
- Dashboard polls: latest sensor, sensor history (bounded, limit 1..5000), irrigation history, water usage, plots, crop health; uses `Promise.allSettled`, 15 s refresh
- Water usage returns both `current_consumption` and `total_consumption`

## 12. AI / MACHINE LEARNING INTEGRATION

- AI URL default `http://127.0.0.1:8001` (non-colliding with Backend 8000); resolved path `/v1/predict/irrigation` in client
- Backend passes context; AI returns advisory; never controls relay
- Prediction source rendered: model / rule_fallback / water_running_guard / unavailable. Score shown only when present; reason shown when present
- If AI unavailable, dashboard displays prediction unavailable (no fabrication)

## 13. SAFETY AND FAIL-SAFE SYSTEM

- Local protections authoritative and independent of Backend/AI/Wi-Fi
- Dry-run and leak protections as above
- Safe boot; non-blocking timing for safety timers
- Pump external supply required; no unsafe mains instructions
- Backend/AI advisory only

## 14. PHYSICAL ASSEMBLY

1. Mount ESP32/ESP32-CAM securely
2. Wire sensors (I²C: SDA21/SCL22 shared; analog soil to GPIO34; flow GPIO18; DHT22 GPIO27)
3. Relay/driver + external pump supply with protection (**VERIFY PHYSICAL MODULE**)
4. LED+220 Ω to GPIO25
5. LCD (check backpack voltage; add level shifter if needed)
6. Verify common GNDs, power ratings
7. Insulate/highlight pump power isolation

## 15. TESTING AND TROUBLESHOOTING

| Check | Expected | If Failing |
|---|---|---|
| Wi-Fi | Connects to configured SSID | Check credentials, 2.4 GHz |
| Backend health | `/api/health` returns ai_url/crops | Verify Backend running, CORS/origins |
| Sensor readings | Finite values, no fabricated NaN | Check wiring, power |
| Flow | Pulses increase with flow | Calibrate K-factor (7.5 sim baseline) |
| Relay/pump | Safety OFF at boot; local rule applies | Verify coil voltage/polarity |
| LCD | Displays pages | Check I²C address, voltage |
| Dry-run | Disables pump after ~3 s no flow | Simulate blocked outlet |
| Leak | Latches after ~3 s flow when OFF | Simulate backflow/valve leak |

## 16. FINAL DEPLOYMENT CHECKLIST

### Hardware
- [ ] All components verified against pin map
- [ ] Pump on external supply, fuse/protection present
- [ ] Common grounds confirmed
- [ ] LCD voltage/backpack compatibility confirmed (level shifter if needed)
- [ ] Relay voltage/current compatibility confirmed
- [ ] YF-S201 calibration plan documented
- [ ] Soil sensor calibration plan documented

### Software/Integration
- [ ] Firmware builds; GPIO matches physical
- [ ] Backend tests pass (73 passed, 6 warnings)
- [ ] Frontend JS syntax valid; live smoke tests pass
- [ ] Health includes ai_url and crops
- [ ] Bounded sensor-data works (limit 1..5000)
- [ ] Plot registration works; crop resolved from plot
- [ ] Safety protections active in firmware
- [ ] Documentation synchronized with repository

### Safety
- [ ] Dry-run and leak protections verified where safely testable
- [ ] No unsafe mains-voltage instructions included
- [ ] Local safety remains authoritative (Backend/AI advisory only)
- [ ] Safe startup behavior confirmed
- [ ] Wi-Fi/Backend/AI failures do not disable local safety
