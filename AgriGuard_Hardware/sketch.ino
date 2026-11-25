/*
 * AgriGuard ESP32 - Phase 1: hardware bring-up.
 *
 * Scope of this phase: complete and reliable SENSOR SIMULATION ONLY.
 * There is deliberately no Wi-Fi, no HTTP, no JSON and no Backend or AI
 * communication in this file. Networking is Phase 2.
 *
 * IMPORTANT - the local soil-moisture threshold below is a TEMPORARY HARDWARE
 * TEST mechanism so the relay and LED can be verified in Wokwi. It is NOT the
 * AgriGuard irrigation decision and must not be described as one. The real
 * decision is crop-aware and arrives from the Backend/AI in a later phase; a
 * single crop-blind threshold cannot serve, because the trigger point is per
 * crop (the AI's own rule fallback uses wheat 22%, rice 60%, tomato 25%).
 *
 * Phase 2A adds ONLY the transport that ships one reading to the Backend:
 * Wi-Fi, one HTTP POST, and logging of the result. It adds no new measurement,
 * no new threshold and no new control path. The relay is still driven from
 * applyTemporaryRule() and nothing else; the Backend's reply is advisory and is
 * logged, never acted on.
 */

#include <math.h>
#include <string.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include <DHT.h>
#include <BH1750.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <freertos/semphr.h>

// ============================================================ PIN ASSIGNMENT
// Mirrors diagram.json. Keep the two in step.
const int SOIL_PIN = 34;        // capacitive soil moisture, analogue out
const int FLOW_SENSOR_PIN = 18; // YF-S201 pulse output (interrupt driven)
const int DHT_PIN = 27;         // DHT22 single-wire data
const int RELAY_PIN = 26;       // pump relay
const int PUMP_LED = 25;        // pump indicator LED through 220 ohm

// I2C moved off GPIO 4/2 onto the ESP32's own default pins. GPIO 2 is a
// boot-strapping pin and is no longer used by this board at all.
const int I2C_SDA = 21;
const int I2C_SCL = 22;

LiquidCrystal_I2C lcd(0x27, 16, 2);
DHT dht(DHT_PIN, DHT22);
BH1750 lightSensor;

// ============================================================ CONSTANTS
// TEMPORARY / PROTOTYPE hardware test only. See the file header.
const int TEMP_LOCAL_DRY_THRESHOLD = 40;

// YF-S201: pulses per litre. This is already the convention baked into the
// shipped Wokwi custom chip yf-s201.chip.c, which emits freq = 7.5 * L/min,
// so simulation and firmware agree with no new constant and no new calibration.
const float YF_S201_PULSES_PER_LITRE = 7.5;

// Main cycle: 1 Hz, drives flow/soil/light/relay/display.
const unsigned long MEASURE_INTERVAL_MS = 1000;

// The DHT22 cannot be sampled faster than about once every 2 s, so climate
// gets its own slower cadence and the 1 Hz cycle keeps its previous values.
// The blocking that would otherwise occur is prevented by the force=true
// arguments in readClimate(), not by this interval.
const unsigned long CLIMATE_INTERVAL_MS = 2500;

// Safety thresholds for the two hardware protections.
const int DRY_RUN_LIMIT = 3;       // seconds of "pump on, no water"
const int LEAK_LIMIT = 3;          // seconds of "pump off, still flowing"
const float LEAK_FLOW_THRESHOLD = 0.5f;  // L/min counted as real flow

// ============================================================ NETWORK (PHASE 2A)
// HACKATHON CONFIGURATION - THESE ARE THE ONLY LINES TO EDIT.
//
// BACKEND_BASE_URL is the laptop's address on the SAME LAN as the ESP32.
// The ESP32 is a separate device, so "localhost" / "127.0.0.1" would point at
// the ESP32 itself and is never correct here. Change the LAN IP below when the
// laptop gets a new address from the router; nothing else needs to change, and
// the sensor path is appended to this one base, never written out in full
// anywhere else.
//
//   >>> THIS IS THE ONE LINE TO CHANGE AT THE HACKATHON <<<
//   Currently 192.168.1.4, the laptop's Wi-Fi address on this network.
//   Check it with:  ipconfig   (look for IPv4 Address under Wi-Fi)
//
// The Backend must be started so it accepts connections from the LAN:
//     uvicorn main:app --host 0.0.0.0 --port 8000
// The default of 127.0.0.1 would make the backend reachable only from the
// laptop itself, never from the ESP32.
const char *BACKEND_BASE_URL = "http://192.168.1.4:8000";

// Must match the device_id of a row created by POST /api/plots. The Backend
// resolves crop_type from that row, so an unregistered id is rejected with
// 422 and NO reading is stored.
const char *DEVICE_ID = "ESP32_01";

// The Backend path, appended to BACKEND_BASE_URL to form the endpoint.
const char *SENSOR_PATH = "/api/sensor-data";

// Wokwi cannot reach a private LAN address. If the project is later run in
// Wokwi, point BACKEND_BASE_URL at a publicly reachable endpoint (a tunnel to
// this laptop) instead. No other code changes: the base URL is the only thing
// that decides where the request goes.

// Station credentials. Wokwi-WiFi / WokwiPass are the credentials of the
// virtual access point Wokwi provides; for physical hardware set these to the
// real network's SSID and password.
const char *WIFI_SSID = "Wokwi-WiFi";
const char *WIFI_PASSWORD = "WokwiPass";

// How often a reading is handed to the sender. Deliberately slower than the
// 1 Hz control cycle: the irrigation picture does not need 1 Hz, and a slow
// cadence keeps the volume of Backend writes low.
const unsigned long POST_INTERVAL_MS = 15000;

// Timeouts for the sender task. The connect timeout is short so a wrong or
// unreachable LAN IP is reported quickly. The overall timeout is deliberately
// longer than the Backend's own 30 s AI call, because this endpoint calls the
// AI service inline and a healthy but slow Backend therefore answers late.
// The sender runs on its own task, so a long timeout costs nothing here.
const unsigned long HTTP_CONNECT_TIMEOUT_MS = 5000;
const unsigned long HTTP_TIMEOUT_MS = 35000;

// How long the sender waits for the association to come up before giving up
// and retrying on a later tick.
const unsigned long WIFI_CONNECT_WAIT_MS = 15000;

// Payload buffer. The body is ~180 bytes; 256 leaves room for a longer
// device_id without silently truncating the JSON.
const size_t POST_PAYLOAD_CAPACITY = 256;

// ============================================================ FLOW MEASUREMENT
volatile unsigned long pulseCount = 0;
unsigned long lastPulseAt = 0;

void IRAM_ATTR pulseCounter() {
  pulseCount++;
  lastPulseAt = micros();
}

// Snapshot-and-clear inside a short critical section. The previous firmware
// called detachInterrupt() around the whole one-second block, LCD writes
// included, so every pulse arriving during a display update was discarded and
// the running total always under-read. pulseCount is a 32-bit volatile, so the
// load is atomic and only the reset needs protecting - the interrupt stays
// attached permanently.
unsigned long takePulseCount() {
  noInterrupts();
  unsigned long captured = pulseCount;
  pulseCount = 0;
  interrupts();
  return captured;
}

// ============================================================ SENSOR STATE
struct Readings {
  float moisture = 0.0;      // Wokwi simulation percent, 0 dry .. 100 wet
  float temperature = NAN;   // DHT22 degC
  float humidity = NAN;      // DHT22 %RH
  float light = -1.0;        // BH1750 native lux, negative means unavailable
  float flowRate = 0.0;      // L/min
  float totalLitres = 0.0;   // running total
  bool pumpOn = false;
};

Readings r;
bool dhtPresent = false;
bool lightPresent = false;
bool lightReadFailed = false;
bool leakAlarm = false;
const char *sensorError = "";
int dryRunCounter = 0;
int leakCounter = 0;
int lcdPage = 0;

const unsigned long BOOT_BANNER_MS = 2000;
unsigned long bootTime = 0;
unsigned long lastMeasure = 0;
unsigned long lastClimate = 0;

// ============================================================ LCD
// Four pages, one page per second, so each cycle rewrites only 32 characters
// rather than the whole screen. That keeps loop() fast and the flow interrupt
// responsive.
//
// Every format below is sized so the widest possible real sensor value still
// fits 16 columns; a value wider than its field would silently push the rest
// of the row off the display.
//   moisture    0..100          -> %3d   (3)
//   temperature -40.0..80.0     -> %5.1f (5)
//   humidity    0.0..100.0      -> %5.1f (5)
//   light       0..65535 lux    -> %5u   (5)
//   flow        0.00..30.00     -> %5.2f (5)
void lcdLine(uint8_t row, const char *text) {
  lcd.setCursor(0, row);
  lcd.print(text);
  for (uint8_t i = (uint8_t)strlen(text); i < 16; i++) {
    lcd.print(' ');
  }
}

void showPage() {
  char top[17];
  char bottom[17];

  switch (lcdPage) {
    case 0:  // soil + temperature
      snprintf(top, sizeof(top), "Soil:%3d%%", (int)r.moisture);
      if (isnan(r.temperature)) {
        snprintf(bottom, sizeof(bottom), "Temp:ERR");
      } else {
        snprintf(bottom, sizeof(bottom), "Temp:%5.1fC", r.temperature);
      }
      break;

    case 1:  // humidity + light
      if (isnan(r.humidity)) {
        snprintf(top, sizeof(top), "RH:ERR");
      } else {
        snprintf(top, sizeof(top), "RH:%5.1f%%", r.humidity);
      }
      if (r.light < 0) {
        snprintf(bottom, sizeof(bottom), "Light:ERR");
      } else {
        snprintf(bottom, sizeof(bottom), "Light:%5u lx", (unsigned)r.light);
      }
      break;

    case 2:  // flow + running total
      snprintf(top, sizeof(top), "Flow:%5.2fL/m", r.flowRate);
      snprintf(bottom, sizeof(bottom), "Total:%6.1fL", r.totalLitres);
      break;

    default:  // status
      snprintf(top, sizeof(top), "Pump:%-3s [TMP]", r.pumpOn ? "ON" : "OFF");
      if (leakAlarm) {
        snprintf(bottom, sizeof(bottom), "ERR:LEAK");
      } else if (*sensorError == '\0') {
        snprintf(bottom, sizeof(bottom), "AI:OFF  Phase 2");
      } else {
        snprintf(bottom, sizeof(bottom), "ERR:%s", sensorError);
      }
      break;
  }

  lcdLine(0, top);
  lcdLine(1, bottom);
  lcdPage = (lcdPage + 1) % 4;
}

// ============================================================ SENSORS
void readSoilMoisture() {
  // Wokwi SIMULATION value, NOT a calibrated real-world soil reading. The
  // capacitive sensor's analogue output rises with wetness, so mapping
  // 0..4095 straight onto 0..100 already yields the project convention
  // 0 = dry, 100 = saturated, which is the scale the dataset and model use.
  // No dry/wet calibration constants are invented here because the project has
  // none; on real hardware this needs a genuine two-point calibration.
  int raw = analogRead(SOIL_PIN);
  long percent = map(raw, 0, 4095, 0, 100);
  r.moisture = (float)constrain(percent, 0L, 100L);
}

void readClimate() {
  // readTemperature()/readHumidity() return NAN on any bus error, checksum
  // failure or absent sensor. NAN is carried through to diagnostics and the
  // LCD shows ERR; it is never rendered as a plausible number.
  //
  // force=true on both calls is essential and not optional. The DHT library
  // refuses to sample more often than every 2000 ms and enforces that with a
  // blocking delay(), and it refreshes that timer after EVERY read. Reading
  // temperature and then humidity in the ordinary way would make the second
  // call stall loop() for ~2 s, wrecking the flow window and the display. The
  // DHT22 returns both values in a single frame, so forcing the reads is safe.
  //
  // The two signatures are asymmetric: readHumidity(bool force) takes force
  // first, while readTemperature(bool S, bool force) has S first.
  r.humidity = dht.readHumidity(true);
  r.temperature = dht.readTemperature(false, true);

  // Presence cannot be checked in setup() because begin() returns void, so it
  // is derived here from whether a valid frame was ever decoded. The flag is
  // deliberately sticky: once a sensor has answered, a later NAN means a READ
  // failure rather than a missing sensor, and clearing the flag would
  // misreport every transient glitch as DHT22-MISS.
  if (!isnan(r.temperature) && !isnan(r.humidity)) {
    dhtPresent = true;
  }
}

void readLight() {
  if (!lightPresent) {
    r.light = -1.0;
    return;
  }
  // readLightLevel() returns native lux as a float, and signals failure with a
  // NEGATIVE value: -1 when the two-byte read did not complete and -2 when the
  // device was never configured. Any negative reading is therefore a failure.
  // The last good value is kept rather than replacing it with a bogus number,
  // but the failure is remembered so it can still be reported: relying on
  // r.light < 0 alone would never fire, because the value is sticky.
  float lux = lightSensor.readLightLevel();
  if (lux < 0) {
    lightReadFailed = true;
    return;
  }
  r.light = lux;  // native lux
  lightReadFailed = false;
}

// ============================================================ PUMP (TEMPORARY)
// TEMPORARY / PROTOTYPE hardware test only.
//
// The two protections below are genuine hardware safety and are kept
// deliberately; only the crop-blind threshold is temporary.
void applyTemporaryRule() {
  bool wantOn = r.moisture < TEMP_LOCAL_DRY_THRESHOLD;

  if (wantOn) {
    if (r.flowRate == 0.0f) {
      // Dry run: commanded on but no water after 3 seconds.
      if (dryRunCounter < DRY_RUN_LIMIT) {
        dryRunCounter++;
      }
      r.pumpOn = (dryRunCounter < DRY_RUN_LIMIT);
    } else {
      dryRunCounter = 0;
      r.pumpOn = true;
    }
  } else {
    dryRunCounter = 0;
    r.pumpOn = false;
  }

  // Leak: water still moving with the pump off for 3 seconds.
  if (!r.pumpOn && r.flowRate > LEAK_FLOW_THRESHOLD) {
    if (leakCounter < LEAK_LIMIT) {
      leakCounter++;
    }
  } else {
    leakCounter = 0;
  }
  leakAlarm = (leakCounter >= LEAK_LIMIT);

  digitalWrite(RELAY_PIN, r.pumpOn ? HIGH : LOW);
  digitalWrite(PUMP_LED, r.pumpOn ? HIGH : LOW);
}

// ============================================================ DIAGNOSTICS
// Serial access is shared with the Phase 2A sender task and is guarded by a
// mutex defined further down; forward-declared here so this section can lock it.
void serialBegin();
void serialEnd();

// One line per 1 Hz measurement cycle. Never per loop() iteration.
//
// Held under the Serial lock so the Phase 2A sender task's [net] lines cannot
// interleave into the middle of this one. Content and cadence are unchanged.
void reportCycle(unsigned long now, unsigned long windowMs) {
  serialBegin();
  Serial.print(F("[ag] t="));
  Serial.print(now);
  Serial.print(F("ms win="));
  Serial.print(windowMs);
  Serial.print(F("ms"));

  Serial.print(F("  soil="));
  Serial.print(r.moisture, 1);
  Serial.print(F("% adc="));
  Serial.print(analogRead(SOIL_PIN));

  Serial.print(F("  temp="));
  if (isnan(r.temperature)) {
    Serial.print(F("NAN"));
  } else {
    Serial.print(r.temperature, 1);
  }
  Serial.print(F("C  hum="));
  if (isnan(r.humidity)) {
    Serial.print(F("NAN"));
  } else {
    Serial.print(r.humidity, 1);
  }
  Serial.print(F("%"));

  Serial.print(F("  light="));
  if (r.light < 0) {
    Serial.print(F("NAN"));
  } else {
    Serial.print(r.light, 0);
  }
  Serial.print(F("lux"));

  Serial.print(F("  flow="));
  Serial.print(r.flowRate, 2);
  Serial.print(F("L/min  total="));
  Serial.print(r.totalLitres, 2);
  Serial.print(F("L  pulses="));
  Serial.print(lastPulseAt > 0 ? F("seen") : F("none"));

  Serial.print(F("  relay="));
  Serial.print(digitalRead(RELAY_PIN) ? F("ON") : F("OFF"));
  Serial.print(F("  led="));
  Serial.print(digitalRead(PUMP_LED) ? F("ON") : F("OFF"));
  Serial.print(F("  dryrun="));
  Serial.print(dryRunCounter);
  Serial.print(F("  leak="));
  Serial.print(leakAlarm ? F("YES") : F("no"));

  Serial.print(F("  err="));
  // Both branches must be the same type: F() yields const __FlashStringHelper*
  // while sensorError is a const char*, and mixing them in one ternary does not
  // compile. A plain literal keeps the ternary homogeneous.
  Serial.println(*sensorError == '\0' ? "none" : sensorError);
  serialEnd();
}

void checkSensorHealth() {
  sensorError = "";
  if (!dhtPresent) {
    sensorError = "DHT22-MISS";
  } else if (isnan(r.temperature) || isnan(r.humidity)) {
    sensorError = "DHT22-READ";
  } else if (!lightPresent) {
    sensorError = "BH1750-MISS";
  } else if (lightReadFailed) {
    sensorError = "BH1750-READ";
  }
}

// ============================================================ NETWORK (PHASE 2A)
// Phase 2A transport only. Nothing in this section reads a sensor, writes a
// pin, or influences the relay. It moves one already-measured reading to the
// Backend and reports the outcome.

SemaphoreHandle_t serialLock = NULL;

// Serial is written from two contexts: the 1 Hz control loop and the sender
// task. Without a lock their output interleaves mid-line and corrupts
// diagnostics, so every print block takes this. The wait is bounded by a print
// and is negligible next to the 1 s cycle, which is why a mutex is used rather
// than a spinlock - Serial.print() can block on a full UART buffer and must
// never be held inside a critical section.
void serialBegin(void) {
  if (serialLock) {
    xSemaphoreTake(serialLock, portMAX_DELAY);
  }
}

void serialEnd(void) {
  if (serialLock) {
    xSemaphoreGive(serialLock);
  }
}

// ------------------------------------------------------- handoff slot
// One "latest reading wins" slot rather than a queue. Only the newest reading
// is worth sending, so a backlog would be pure waste, and a single slot means
// the sender can never fall behind the control loop. A snapshot is copied by
// value under a short critical section, so the sender never reads partially
// written sensor data.
typedef struct {
  bool pending;
  char payload[POST_PAYLOAD_CAPACITY];
} PostSlot;

portMUX_TYPE postMux = portMUX_INITIALIZER_UNLOCKED;
PostSlot postSlot;

bool takePostForSend(char *out, size_t capacity) {
  bool got = false;
  portENTER_CRITICAL(&postMux);
  if (postSlot.pending) {
    size_t i = 0;
    while (i + 1 < capacity && postSlot.payload[i] != '\0') {
      out[i] = postSlot.payload[i];
      i++;
    }
    out[i] = '\0';
    postSlot.pending = false;
    got = true;
  }
  portEXIT_CRITICAL(&postMux);
  return got;
}

void queuePost(const char *payload) {
  portENTER_CRITICAL(&postMux);
  size_t i = 0;
  while (i + 1 < POST_PAYLOAD_CAPACITY && payload[i] != '\0') {
    postSlot.payload[i] = payload[i];
    i++;
  }
  postSlot.payload[i] = '\0';
  postSlot.pending = true;
  portEXIT_CRITICAL(&postMux);
}

// ------------------------------------------------------- endpoint
// BACKEND_BASE_URL + SENSOR_PATH. Built here and nowhere else, so the Hackathon
// change is a one-line edit. A trailing slash on the base is tolerated so
// "http://host:8000/" does not produce a doubled separator.
char gEndpoint[128];

void buildEndpoint() {
  const char *base = BACKEND_BASE_URL;
  size_t len = strlen(base);
  while (len > 0 && base[len - 1] == '/') {
    len--;
  }
  if (len >= sizeof(gEndpoint) - 2) {
    gEndpoint[0] = '\0';
    return;
  }
  memcpy(gEndpoint, base, len);
  strcpy(gEndpoint + len, SENSOR_PATH);
}

// ------------------------------------------------------- validation
// The Backend rejects a non-finite temperature or humidity with HTTP 500
// (SQLite cannot store them), and it silently accepts a negative light value as
// a real reading. Neither is recoverable server-side, so a reading carrying any
// of them is never sent. No placeholder is substituted: a skipped upload is
// honest, a fabricated zero would poison the stored row and the AI input.
// soil_moisture and water_flow are always finite here because the firmware
// derives them from map()/arithmetic and never introduces a failure sentinel.
bool readingsUploadable() {
  return isfinite(r.temperature) && isfinite(r.humidity) && isfinite(r.light) &&
         r.light >= 0.0f && isfinite(r.moisture) && isfinite(r.flowRate);
}

const char *readingsRejectReason() {
  if (!isfinite(r.temperature) || !isfinite(r.humidity)) {
    return "climate invalid (NAN/Inf from DHT22)";
  }
  if (!isfinite(r.light) || r.light < 0.0f) {
    return "light invalid (BH1750 failure sentinel)";
  }
  if (!isfinite(r.moisture) || !isfinite(r.flowRate)) {
    return "soil/flow non-finite";
  }
  return "";
}

// ------------------------------------------------------- payload
// Built with snprintf, not ArduinoJson: ArduinoJson is not part of the ESP32
// Arduino core, and this body is a fixed shape with no nesting, so a library
// would add a dependency for nothing.
//
// device_id is interpolated raw. DEVICE_ID is a compile-time constant
// containing no quote or backslash, so the result is always valid JSON. If it
// is ever made configurable it MUST be escaped first.
//
// The field list is exactly what the Backend requires. crop_type is
// deliberately absent: the Backend resolves the crop from the Plot row, and
// extra keys are silently discarded.
bool buildPostPayload(char *out, size_t capacity) {
  int n = snprintf(
      out, capacity,
      "{\"device_id\":\"%s\",\"soil_moisture\":%.1f,\"temperature\":%.1f,"
      "\"humidity\":%.1f,\"light\":%.0f,\"water_flow\":%.2f,\"pump_status\":%s}",
      DEVICE_ID, r.moisture, r.temperature, r.humidity, r.light, r.flowRate,
      r.pumpOn ? "true" : "false");
  // n < 0 is an encoding failure; n >= capacity is truncation, which would
  // produce invalid JSON. Either way the reading is dropped rather than sent.
  if (n <= 0 || (size_t)n >= capacity) {
    return false;
  }
  return true;
}

// ------------------------------------------------------- Wi-Fi
// Reported once per state change rather than once per attempt, so a Backend or
// router outage cannot flood the log.
bool wifiLoggedDown = false;

bool ensureWifiConnected() {
  if (WiFi.status() == WL_CONNECTED) {
    if (wifiLoggedDown) {
      serialBegin();
      Serial.println(F("[net] wifi reconnected"));
      serialEnd();
      wifiLoggedDown = false;
    }
    return true;
  }

  if (!wifiLoggedDown) {
    serialBegin();
    Serial.println(F("[net] wifi down, associating"));
    serialEnd();
    wifiLoggedDown = true;
  }

  WiFi.reconnect();
  unsigned long start = millis();
  while (WiFi.status() != WL_CONNECTED &&
         (millis() - start) < WIFI_CONNECT_WAIT_MS) {
    delay(100);
  }
  return WiFi.status() == WL_CONNECTED;
}

// ------------------------------------------------------- one POST
// Returns true only on HTTP 200. Deliberately performs no retry: the Backend
// has no idempotency key, so a request that timed out may well have been
// stored. Re-sending it would create a second SensorReading row for one event,
// so a failed attempt is dropped and the next scheduled tick sends a NEW
// reading instead.
void performPost(const char *payload) {
  ensureWifiConnected();

  if (WiFi.status() != WL_CONNECTED) {
    serialBegin();
    Serial.println(F("[net] POST skipped: no wifi; reading not sent"));
    serialEnd();
    return;
  }

  if (gEndpoint[0] == '\0') {
    serialBegin();
    Serial.println(F("[net] POST skipped: BACKEND_BASE_URL too long"));
    serialEnd();
    return;
  }

  WiFiClient client;
  HTTPClient http;
  http.setConnectTimeout(HTTP_CONNECT_TIMEOUT_MS);
  http.setTimeout(HTTP_TIMEOUT_MS);

  serialBegin();
  Serial.print(F("[net] POST "));
  Serial.print(gEndpoint);
  Serial.print(F(" -> "));
  Serial.println(payload);
  serialEnd();

  bool begun = http.begin(client, gEndpoint);
  if (!begun) {
    serialBegin();
    Serial.println(F("[net] FAIL: http.begin() could not start"));
    serialEnd();
    http.end();
    return;
  }

  http.addHeader(F("Content-Type"), F("application/json"));

  int status = http.POST(payload);

  serialBegin();
  Serial.print(F("[net] HTTP "));
  Serial.print(status);
  serialEnd();

  if (status == 200) {
    String body = http.getString();
    serialBegin();
    Serial.print(F("[net] OK: reading accepted, response "));
    Serial.println(body.substring(0, 200));
    serialEnd();
  } else if (status > 0) {
    String body = http.getString();
    serialBegin();
    Serial.print(F("[net] REJECTED: "));
    Serial.println(body.substring(0, 200));
    serialEnd();
  } else {
    // Negative codes are transport failures (no route, timeout, refused).
    // Whether or not the Backend stored the reading is unknown, so this is
    // logged and NOT retried.
    serialBegin();
    Serial.print(F("[net] FAIL: transport error "));
    Serial.print(status);
    Serial.println(F(" (not retried; the Backend may or may not have stored it)"));
    serialEnd();
  }

  http.end();
}

// ------------------------------------------------------- sender task
// This is the decoupling. The POST is a blocking call and the Backend can take
// up to its 30 s AI timeout to answer, so running it inside loop() would stall
// the 1 Hz cycle for that whole period. That matters here because dryRunCounter
// and leakCounter advance once per cycle: a stalled loop would stretch the
// "3 seconds" protections into many seconds and make the flow window wrong.
//
// Instead the loop only copies a finished reading into postSlot and returns.
// This task runs on the other core, does the blocking work there, and the
// safety logic never waits for the network.
void backendSenderTask(void *unused) {
  (void)unused;
  char payload[POST_PAYLOAD_CAPACITY];
  for (;;) {
    if (!takePostForSend(payload, sizeof(payload))) {
      vTaskDelay(100 / portTICK_PERIOD_MS);
      continue;
    }
    performPost(payload);
    // Small gap so two sends cannot sit back to back on the wire.
    vTaskDelay(250 / portTICK_PERIOD_MS);
  }
}

unsigned long lastPost = 0;

// ============================================================ SETUP
void setup() {
  Serial.begin(115200);
  unsigned long serialStart = millis();
  while (!Serial && millis() - serialStart < 2000) {
    ;  // never block forever on a host that never opens the monitor
  }

  serialLock = xSemaphoreCreateMutex();

  pinMode(RELAY_PIN, OUTPUT);
  pinMode(PUMP_LED, OUTPUT);
  pinMode(FLOW_SENSOR_PIN, INPUT_PULLUP);

  // Fail safe: the pump stays off until a reading says otherwise.
  digitalWrite(RELAY_PIN, LOW);
  digitalWrite(PUMP_LED, LOW);

  Wire.begin(I2C_SDA, I2C_SCL);
  lcd.init();
  lcd.backlight();
  lcd.clear();
  lcdLine(0, "AgriGuard");
  lcdLine(1, "Phase 1 sensors");
  bootTime = millis();

  // The DHT library's begin() returns void, so it cannot report whether a
  // sensor is actually attached. Presence is derived from the first real read
  // in readClimate(), which is the only reliable signal the library offers.
  dht.begin();

  lightPresent = lightSensor.begin();
  if (!lightPresent) {
    Serial.println(F("[ag] WARN: BH1750 not found on I2C 21/22"));
  }

  attachInterrupt(digitalPinToInterrupt(FLOW_SENSOR_PIN), pulseCounter, RISING);

  // ---- Phase 2A: network start-up -------------------------------------
  buildEndpoint();

  // WiFi.begin() is asynchronous: it requests the association and returns, so
  // this does not delay the first control cycle. Waiting for the association is
  // the sender task's job.
  WiFi.persistent(false);
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);  // modem sleep stalls short HTTP exchanges
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  // Core 1 is where the Arduino loopTask (this file's loop()) runs, so the
  // sender is pinned to core 0 to get genuine parallelism rather than
  // time-slicing on the core that owns the safety logic.
  xTaskCreatePinnedToCore(backendSenderTask, "backendSender", 8192, NULL, 1,
                          NULL, 0);

  Serial.println(F("[ag] ------------------------------------------------"));
  Serial.println(F("[ag] AgriGuard ESP32 Phase 2A - sensors + Backend POST"));
  Serial.print(F("[ag] backend="));
  // Two separate statements rather than a ternary: char* and
  // const __FlashStringHelper* have no common type, so a ternary here would not
  // compile.
  if (gEndpoint[0] != '\0') {
    Serial.println(gEndpoint);
  } else {
    Serial.println(F("<invalid BACKEND_BASE_URL>"));
  }
  Serial.print(F("[ag] device_id="));
  Serial.println(DEVICE_ID);
  Serial.println(F("[ag] local soil<40 rule is a TEMPORARY test only"));
  Serial.print(F("[ag] pins: soil="));
  Serial.print(SOIL_PIN);
  Serial.print(F(" flow="));
  Serial.print(FLOW_SENSOR_PIN);
  Serial.print(F(" dht="));
  Serial.print(DHT_PIN);
  Serial.print(F(" relay="));
  Serial.print(RELAY_PIN);
  Serial.print(F(" led="));
  Serial.print(PUMP_LED);
  Serial.print(F(" sda="));
  Serial.print(I2C_SDA);
  Serial.print(F(" scl="));
  Serial.println(I2C_SCL);
  Serial.println(F("[ag] local soil<40 rule is a TEMPORARY test only"));
  Serial.println(F("[ag] ------------------------------------------------"));

  lastMeasure = millis();
  lastClimate = millis();
}

// ============================================================ LOOP
void loop() {
  unsigned long now = millis();

  if (now - lastMeasure < MEASURE_INTERVAL_MS) {
    return;  // keep the loop tight and the flow interrupt responsive
  }
  unsigned long elapsed = now - lastMeasure;
  lastMeasure = now;

  if (now - bootTime < BOOT_BANNER_MS) {
    return;
  }

  // --- climate on its own slower, non-blocking cadence ----------------
  if (now - lastClimate >= CLIMATE_INTERVAL_MS) {
    lastClimate = now;
    readClimate();
  }

  readSoilMoisture();
  readLight();

  // --- flow, measured with the interrupt permanently attached ---------
  unsigned long pulses = takePulseCount();
  // The chip toggles its output with a half-period of 1/freq, so it produces
  // freq = 7.5 * L/min rising edges per second. Inverting that:
  //     L/min = pulses_in_window * 1000 / (window_ms * 7.5)
  // Scaling by the measured window length keeps the result correct even if a
  // cycle runs long, so a slow cycle does not read as a high flow.
  float litres_in_window = (float)pulses / YF_S201_PULSES_PER_LITRE;
  r.flowRate = (elapsed > 0)
                   ? litres_in_window * (1000.0f / (float)elapsed)
                   : 0.0f;
  r.totalLitres += r.flowRate / 60.0f;

  applyTemporaryRule();
  checkSensorHealth();
  reportCycle(now, elapsed);
  showPage();

  // --- Phase 2A: hand one finished reading to the sender -----------------
  // This is the entire network contribution to the control loop: build a
  // snapshot and copy it into a slot. No socket, no blocking call, no waiting
  // on the Backend. If the reading is not uploadable the slot is simply not
  // filled, so nothing is sent and nothing is invented. Either way the relay,
  // the dry-run counter and the leak counter above have already run, exactly
  // as before.
  if (now - lastPost >= POST_INTERVAL_MS) {
    lastPost = now;
    const char *reject = readingsRejectReason();
    if (reject[0] != '\0') {
      serialBegin();
      Serial.print(F("[net] POST skipped: "));
      Serial.println(reject);
      serialEnd();
    } else {
      char payload[POST_PAYLOAD_CAPACITY];
      if (buildPostPayload(payload, sizeof(payload))) {
        queuePost(payload);
      } else {
        serialBegin();
        Serial.println(F("[net] POST skipped: payload would be truncated"));
        serialEnd();
      }
    }
  }
}
