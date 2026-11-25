// ==========================================
// API BASE URL
//
// Phase 2B. The base URL used to be a hardcoded literal, so pointing the
// dashboard at a different Backend meant editing source at the venue. It is now
// resolved in this order:
//
//   1. ?api=http://192.168.1.4:8000   in the page URL   (one-off, highest priority)
//   2. localStorage "agriguard_api_base"                  (remembered choice)
//   3. http://127.0.0.1:8000                             (same-laptop default)
//
// A ?api= value is remembered, so it only has to be typed once per machine.
//
// Loopback is the correct default here and NOT a bug: the Backend runs on the
// same laptop as the dashboard, so the browser may address it directly. The
// ESP32 cannot - its own BACKEND_BASE_URL in sketch.ino must be the laptop's
// LAN IP. The two are configured separately and are not expected to match.
// ==========================================

const API_DEFAULT_BASE_URL = "http://127.0.0.1:8000";
const API_STORAGE_KEY = "agriguard_api_base";

// Every request is abandoned after this long. Without it a hung Backend left
// the promise unsettled forever and the 15 s interval kept stacking up new
// requests on top of it.
const API_TIMEOUT_MS = 8000;

// The firmware posts one reading every 15 s, so ~100 readings is about 25
// minutes of history. Asking for the whole table instead meant downloading
// every reading ever stored, on every refresh.
const API_HISTORY_LIMIT = 100;

function stripTrailingSlash(value) {
    return String(value).replace(/\/+$/, "");
}

function readQueryApiOverride() {
    try {
        const params = new URLSearchParams(window.location.search);
        const value = params.get("api");
        return value ? stripTrailingSlash(value) : null;
    } catch (error) {
        return null;
    }
}

function readStoredBaseUrl() {
    try {
        const value = window.localStorage.getItem(API_STORAGE_KEY);
        return value ? stripTrailingSlash(value) : null;
    } catch (error) {
        // Private browsing / disabled storage: fall through to the default.
        return null;
    }
}

function storeBaseUrl(value) {
    try {
        window.localStorage.setItem(API_STORAGE_KEY, value);
    } catch (error) {
        // Not fatal - the URL still works for this page load.
    }
}

const queryOverride = readQueryApiOverride();

const API_BASE_URL =
    queryOverride || readStoredBaseUrl() || API_DEFAULT_BASE_URL;

if (queryOverride) {
    storeBaseUrl(queryOverride);
}

// Shown in the connection badge, so a wrong URL is obvious instead of showing
// up as a mysterious "offline".
function getApiBaseUrl() {
    return API_BASE_URL;
}

// The crop list is not hardcoded: it is read from the Backend's own contract at
// /api/health, which reports exactly the crops the AI service recognises. The
// fallback mirrors AgriGuard_Backend/crops.py and is only used if the Backend
// cannot be reached at all; the UI says so when that happens.
const API_FALLBACK_CROPS = [
    "chili",
    "maize",
    "onion",
    "pepper",
    "potato",
    "rice",
    "soybean",
    "tomato",
    "wheat"
];


// ==========================================
// HELPERS
// ==========================================

function buildUrl(endpoint) {
    return `${API_BASE_URL}${endpoint}`;
}

// One place that owns the timeout, so every call gets it and none can forget.
async function fetchWithTimeout(url, options) {
    const controller = new AbortController();
    const timer = window.setTimeout(
        () => controller.abort(),
        API_TIMEOUT_MS
    );

    try {
        return await fetch(url, {
            ...options,
            signal: controller.signal
        });
    } finally {
        window.clearTimeout(timer);
    }
}

// Distinguishes a timeout from a rejected connection, because "the request took
// longer than 8 s" and "nothing is listening on this address" have different
// causes and different fixes.
function describeFailure(error, endpoint) {
    if (error && error.name === "AbortError") {
        return `${endpoint} timed out after ${API_TIMEOUT_MS / 1000}s`;
    }
    return `API Error: cannot reach ${API_BASE_URL} (${endpoint})`;
}

async function apiRequest(endpoint) {

    let response;

    try {
        response = await fetchWithTimeout(buildUrl(endpoint));
    } catch (error) {
        throw new Error(describeFailure(error, endpoint));
    }

    if (!response.ok) {
        throw new Error(
            `API Error ${response.status}: ${endpoint}`
        );
    }

    return await response.json();
}


// Plot configuration is the only write the dashboard performs. The crop is
// registered here, against the device that reports for it, so the firmware
// never has to name a crop.
async function apiWriteRequest(
    endpoint,
    method,
    body
) {

    let response;

    try {
        response = await fetchWithTimeout(buildUrl(endpoint), {
            method,
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(body)
        });
    } catch (error) {
        throw new Error(describeFailure(error, endpoint));
    }

    // The Backend explains why a plot was rejected in `detail`. Surfacing it
    // matters here: an unknown crop or a duplicate device_id is a 4xx the
    // operator has to fix, and a bare status code does not say which.
    if (!response.ok) {
        let detail = "";
        try {
            const body = await response.json();
            detail = body && body.detail ? ` - ${body.detail}` : "";
        } catch (parseError) {
            // Non-JSON error body; the status code is all we have.
        }
        throw new Error(
            `API Error ${response.status}: ${endpoint}${detail}`
        );
    }

    return await response.json();
}


// ==========================================
// HEALTH
// Also the source of the crop vocabulary and the resolved AI URL.
// ==========================================

async function getHealth() {
    return await apiRequest(
        "/api/health"
    );
}


// ==========================================
// PLOT CONFIGURATION
// CROP IDENTITY LIVES HERE, NOT ON THE ESP32
// ==========================================

async function getPlots() {
    return await apiRequest(
        "/api/plots"
    );
}

async function createPlot(
    name,
    cropType,
    deviceId
) {

    return await apiWriteRequest(
        "/api/plots",
        "POST",
        {
            name,
            crop_type: cropType,
            device_id: deviceId
        }
    );
}


// ==========================================
// SENSOR DATA
// ==========================================

async function getLatestSensorData() {
    return await apiRequest(
        "/api/latest-sensor-data"
    );
}

// Bounded by default. The response shape is unchanged - still an array of
// readings, newest first - so nothing downstream has to adapt.
async function getSensorHistory(limit = API_HISTORY_LIMIT) {
    return await apiRequest(
        `/api/sensor-data?limit=${encodeURIComponent(limit)}`
    );
}


// ==========================================
// AI PREDICTION
//
// Phase 2B note: the real prediction is NOT read from this endpoint.
//
// GET /api/ai-prediction serves the AIPrediction table, and nothing in the
// project ever writes to it, so it returns "No prediction available" even when
// the AI is working. The prediction the Backend actually stores is on the
// reading itself, under latest-sensor-data -> irrigation. This wrapper is kept
// because POST /api/ai-prediction exists and is part of the API contract.
// ==========================================

async function getAIPrediction() {
    return await apiRequest(
        "/api/ai-prediction"
    );
}


// ==========================================
// CROP HEALTH
// ==========================================

async function getCropHealth() {
    return await apiRequest(
        "/api/crop-health"
    );
}


// ==========================================
// IRRIGATION
// Populated in Phase 2B by pump-transition derivation in the Backend, from the
// pump_status / water_flow the firmware already reports.
// ==========================================

async function getIrrigationHistory() {
    return await apiRequest(
        "/api/irrigation-history"
    );
}


// ==========================================
// WATER USAGE
// ==========================================

async function getWaterUsage() {
    return await apiRequest(
        "/api/water-usage"
    );
}