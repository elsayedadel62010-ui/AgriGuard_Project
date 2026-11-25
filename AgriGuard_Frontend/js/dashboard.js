// ==========================================
// AgriGuard dashboard
//
// Phase 2B. Three things changed materially here:
//
//   1. The AI Prediction card no longer reads GET /api/ai-prediction. That
//      endpoint serves the AIPrediction table, which nothing in the project
//      writes to, so it reported "No prediction available" even while the AI
//      was working. The real prediction is stored on the reading and arrives
//      on latest-sensor-data under `irrigation`.
//   2. "No data" is no longer rendered as "Alert" (crop health) or as a
//      permanently visible NPK card.
//   3. Every endpoint is settled independently, so one failure no longer blanks
//      the whole dashboard.
// ==========================================

const userName =
    localStorage.getItem("agriguard_user_name") || "User";

const el = id => document.getElementById(id);

// Matches POST_INTERVAL_MS in AgriGuard_Hardware/sketch.ino. The dashboard used
// to poll every 5 s against a device that posts every 15 s, so three of every
// five refreshes redrew identical data.
const REFRESH_MS = 15000;

// How the AI service reports where a decision came from. These are the literal
// `decision_source` values in AgriGuard_Ai/agriguard/irrigation.py.
const DECISION_SOURCES = {
    model: {
        label: "Trained model",
        className: "source-model",
        note: "Scored by the trained irrigation model."
    },
    rule_fallback: {
        label: "Rule fallback",
        className: "source-rule",
        note: "This crop has no model column, so a threshold rule answered instead of the model."
    },
    pump_guard: {
        label: "Pump guard",
        className: "source-guard",
        note: "The pump was already running, so irrigation was not recommended again."
    },
    water_running_guard: {
        label: "Water-flow guard",
        className: "source-guard",
        note: "Water was already flowing, so irrigation was not recommended again."
    }
};

let loadInFlight = false;
let cropsPopulated = false;


// ==========================================
// ESCAPING
// Backend strings (device_id, plot names, crop names, AI reasons) are
// interpolated into innerHTML below, so they are escaped. Previously they were
// interpolated raw.
// ==========================================

function escapeHtml(value) {

    if (value === null || value === undefined) {

        return "";

    }

    return String(value)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");

}


// ==========================================
// FORMAT TIME
// ==========================================

function formatTime(timestamp) {

    if (!timestamp) return "--";

    const date = new Date(timestamp);

    if (Number.isNaN(date.getTime())) {
        return timestamp;
    }

    return date.toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit"
    });

}


// ==========================================
// CONNECTION STATUS
// ==========================================

function setConnectionStatus(connected) {

    const badgeDot = el("connectionDot");
    const sidebarDot = el("sidebarDot");

    const connectionText = el("connectionText");
    const sidebarStatus = el("sidebarStatus");

    if (connected) {

        if (badgeDot) {
            badgeDot.className = "status-dot";
        }

        if (sidebarDot) {
            sidebarDot.className = "status-dot";
        }

        if (connectionText) {
            connectionText.textContent =
                "System Connected";
        }

        if (sidebarStatus) {
            sidebarStatus.textContent =
                "System Online";
        }

    } else {

        if (badgeDot) {
            badgeDot.className =
                "status-dot danger";
        }

        if (sidebarDot) {
            sidebarDot.className =
                "status-dot danger";
        }

        if (connectionText) {
            connectionText.textContent =
                "Backend Offline";
        }

        if (sidebarStatus) {
            sidebarStatus.textContent =
                "System Offline";
        }
    }

    // The resolved URL is in the tooltip, because "Offline" with no address is
    // the least useful message when the real problem is a wrong API_BASE_URL.
    if (connectionText) {
        connectionText.title =
            `Backend: ${getApiBaseUrl()}`;
    }
}


// ==========================================
// SENSOR DATA
// ==========================================

function updateSensorUI(data) {

    if (!data) return;

    // --------------------------------------
    // Main Sensor Values
    // --------------------------------------

    el("moistureValue").textContent =
        data.soil_moisture ?? "--";

    el("temperatureValue").textContent =
        data.temperature ?? "--";

    el("humidityValue").textContent =
        data.humidity ?? "--";

    el("lightValue").textContent =
        data.light_intensity ??
        data.light ??
        "--";

    el("flowValue").textContent =
        data.water_flow ?? "--";


    // --------------------------------------
    // Soil Moisture
    // --------------------------------------

    if (data.soil_moisture !== undefined) {

        const moisture =
            Math.max(
                0,
                Math.min(
                    100,
                    Number(data.soil_moisture)
                )
            );

        el("moistureProgress").style.width =
            `${moisture}%`;

        el("moistureStatus").textContent =

            moisture < 35
                ? "Low moisture level"

                : moisture < 55
                    ? "Moderate moisture level"

                    : "Moisture level available";
    }


    // --------------------------------------
    // Temperature
    // --------------------------------------

    if (data.temperature !== undefined) {

        el("temperatureStatus").textContent =
            "Current sensor reading";
    }


    // --------------------------------------
    // Humidity
    // --------------------------------------

    if (data.humidity !== undefined) {

        el("humidityStatus").textContent =
            "Current sensor reading";
    }


    // --------------------------------------
    // Pump Status
    // The relay is driven ONLY by the firmware's local rule. Nothing the
    // Backend or the AI says can switch the pump, and the card says so, so the
    // displayed state is not read as an AI decision.
    // --------------------------------------

    const pumpOn =
        Boolean(data.pump_status);

    el("pumpState").textContent =
        pumpOn ? "ON" : "OFF";

    el("pumpState").className =
        `pump-state ${pumpOn ? "on" : "off"}`;

    el("pumpState").title =
        "Set by the ESP32's local soil rule, not by the AI.";


    // --------------------------------------
    // Irrigation Status
    // --------------------------------------

    el("irrigationStatusSmall").textContent =

        `Irrigation: ${
            data.irrigation_status ??
            (pumpOn
                ? "Irrigating"
                : "Not Irrigating")
        }`;


    // --------------------------------------
    // Current Irrigation Section
    // --------------------------------------

    const irrigationStatus =
        el("irrigationStatus");

    const irrigationAction =
        el("irrigationAction");

    if (irrigationStatus) {

        irrigationStatus.textContent =
            pumpOn
                ? "Irrigating"
                : "Not Irrigating";
    }

    if (irrigationAction) {

        irrigationAction.textContent =
            pumpOn
                ? "Water is currently being supplied."
                : "Irrigation system is currently inactive.";
    }


    // --------------------------------------
    // Last Update
    // --------------------------------------

    el("lastUpdate").textContent =
        formatTime(data.timestamp);


    // --------------------------------------
    // NPK
    // Phase 2B: the Backend always returns an `npk` object, even when all three
    // values are null because no NPK sensor is fitted - which is the case for
    // the current firmware. Testing the object alone therefore always showed
    // the card, permanently displaying "--". It is shown only when at least
    // one value was actually reported.
    // --------------------------------------

    const npk = data.npk;
    const nitrogen = npk ? npk.nitrogen : null;
    const phosphorus = npk ? npk.phosphorus : null;
    const potassium = npk ? npk.potassium : null;

    const npkPresent =
        nitrogen !== null && nitrogen !== undefined ||
        phosphorus !== null && phosphorus !== undefined ||
        potassium !== null && potassium !== undefined;

    const npkCard = el("npkCard");

    if (npkCard) {

        if (npkPresent) {

            npkCard.style.display = "flex";

            el("npkN").textContent =
                nitrogen ?? "--";

            el("npkP").textContent =
                phosphorus ?? "--";

            el("npkK").textContent =
                potassium ?? "--";

        } else {

            npkCard.style.display = "none";

        }
    }
}


// ==========================================
// AI PREDICTION
//
// Phase 2B source change, and the reason it matters: GET /api/ai-prediction
// reads a table nothing writes, so the card showed "No prediction available"
// while the AI was in fact answering. The stored prediction is on the reading,
// so this now takes the latest-sensor-data payload.
//
// The single signal for "did the AI answer" is `irrigation_needed`. The AI
// either returned a boolean decision or it did not. When it did not, the card
// says the prediction is unavailable - it must NEVER fall back to "No
// irrigation needed", because that reads as a considered decision when in fact
// nobody was asked.
// ==========================================

function updatePrediction(reading) {

    const irrigation =
        reading && reading.irrigation
            ? reading.irrigation
            : null;

    // /api/sensor-data POST returns this key; /api/latest-sensor-data does not
    // currently carry it, so this is honoured if present rather than assumed.
    const serviceError =
        (reading && reading.irrigation_error) || null;

    const answered =
        irrigation &&
        irrigation.irrigation_needed !== null &&
        irrigation.irrigation_needed !== undefined;

    const sourceBadge = el("predictionSource");

    // --------------------------------------
    // No prediction
    // --------------------------------------

    if (!answered) {

        el("predictionText").textContent =
            "Prediction unavailable";

        el("recommendationText").textContent =
            serviceError
                ? `The AI service could not be reached: ${serviceError}`
                : "No prediction has been recorded for this reading. The AI service may be offline.";

        el("predictionReason").textContent =
            "";

        el("predictionConfidence").textContent =
            "--";

        if (sourceBadge) {

            sourceBadge.textContent =
                "Unavailable";

            sourceBadge.className =
                "ai-source source-unavailable";

        }

        return;
    }


    // --------------------------------------
    // A real prediction
    // --------------------------------------

    const needed =
        Boolean(irrigation.irrigation_needed);

    el("predictionText").textContent =
        needed
            ? "Irrigation Recommended"
            : "Irrigation Not Needed";

    const source = irrigation.decision_source;
    const meta =
        DECISION_SOURCES[source] || null;

    el("recommendationText").textContent =
        meta
            ? meta.note
            : "Advisory only. The pump is not driven by this prediction.";

    el("predictionReason").textContent =
        irrigation.reason || "";

    // prediction_score is a 0..1 model score, explicitly NOT a calibrated
    // probability (AgriGuard_Ai/agriguard/contracts.py), so it is shown as a
    // score and never labelled "confidence".
    const score = irrigation.prediction_score;

    el("predictionConfidence").textContent =
        score === null || score === undefined
            ? "--"
            : Number(score).toFixed(2);

    if (sourceBadge) {

        sourceBadge.textContent =
            meta ? meta.label : (source || "Unknown source");

        sourceBadge.className =
            `ai-source ${meta ? meta.className : "source-unknown"}`;

    }
}


// ==========================================
// CROP HEALTH
//
// Phase 2B: the Backend returns placeholder values when it has no crop-health
// result, and the old code compared them against "healthy", got false, and
// rendered a red ALERT badge - presenting the absence of data as a detected
// crop problem. Absence is now its own state.
// ==========================================

const CROP_HEALTH_UNKNOWN_VALUES = [
    "unavailable",
    "unknown",
    "no data",
    "none",
    ""
];

function updateCropHealth(data) {

    if (!data) return;

    const result =
        data.result ??
        data.crop_condition ??
        "";

    const normalised =
        String(result).trim().toLowerCase();

    // No row in the table yet. This is the common case: no ESP32-CAM producer
    // exists in the current firmware, so crop health is always unknown and must
    // never look like a failure.
    const unknown =
        CROP_HEALTH_UNKNOWN_VALUES.includes(
            normalised
        );

    const healthy =
        normalised === "healthy";

    if (unknown) {

        el("healthResult").textContent =
            "No data yet";

        el("healthIcon").textContent = "❔";

        el("healthConfidence").textContent =
            "--";

        const badge = el("healthBadge");

        badge.textContent = "No data";
        badge.className = "health-badge neutral";

        el("healthAlert").classList.add("hidden");

        return;
    }


    // --------------------------------------
    // Real result
    // --------------------------------------

    el("healthResult").textContent = result;

    el("healthIcon").textContent = healthy ? "🌿" : "⚠️";

    el("healthConfidence").textContent =
        data.confidence == null
            ? "--"
            : `${Math.round(
                Number(data.confidence) * 100
            )}%`;


    // --------------------------------------
    // Badge
    // --------------------------------------

    const badge = el("healthBadge");

    badge.textContent = healthy ? "Healthy" : "Alert";

    badge.className =
        `health-badge ${healthy ? "" : "warning"}`;


    // --------------------------------------
    // Alert
    // --------------------------------------

    const alert = el("healthAlert");

    if (data.alert || !healthy) {

        alert.textContent =
            data.message ||
            "Possible crop-health issue detected.";

        alert.classList.remove("hidden");

    } else {

        alert.classList.add("hidden");
    }
}


// ==========================================
// WATER USAGE
// ==========================================

function updateWaterUsage(data) {

    if (!data) return;

    const value =
        data.current_consumption ??
        data.total_consumption ??
        data.water_used;


    el("waterConsumption").textContent =

        value == null
            ? "--"
            : Number(value).toFixed(1);
}


// ==========================================
// IRRIGATION HISTORY
// ==========================================

function updateIrrigationHistory(history) {

    const body =
        el("irrigationHistoryBody");


    if (
        !Array.isArray(history) ||
        history.length === 0
    ) {

        body.innerHTML = `
            <tr>
                <td colspan="3" class="empty-row">
                    No completed irrigation events yet. One is recorded when the
                    relay turns off after the pump has run.
                </td>
            </tr>
        `;

        return;
    }


    body.innerHTML = history.map(event => `

        <tr>

            <td>
                ${escapeHtml(
                    event.time ??
                    event.timestamp ??
                    "--"
                )}
            </td>

            <td>
                ${escapeHtml(event.duration ?? "--")}
            </td>

            <td>
                ${
                    escapeHtml(
                        event.water_used ??
                        event.water_consumption ??
                        "--"
                    )
                } L
            </td>

        </tr>

    `).join("");
}


// ==========================================
// PLOTS
// The crop vocabulary comes from the Backend's own contract at /api/health,
// not from a list typed into the dashboard, so the two cannot drift.
// ==========================================

async function loadCrops() {

    const select = el("plotCrop");

    if (!select || cropsPopulated) return;

    let crops = [];
    let fromBackend = true;

    try {

        const health = await getHealth();

        if (Array.isArray(health.crops) && health.crops.length > 0) {
            crops = health.crops;
        }

    } catch (error) {

        fromBackend = false;

    }

    if (crops.length === 0) {
        crops = API_FALLBACK_CROPS;
        fromBackend = false;
    }

    select.innerHTML = crops
        .map(crop =>
            `<option value="${escapeHtml(crop)}">${escapeHtml(crop)}</option>`
        )
        .join("");

    const note = el("plotCropsNote");

    if (note) {
        note.textContent = fromBackend
            ? "Crop list read from the Backend."
            : "Backend unreachable - showing the documented fallback list.";
    }

    cropsPopulated = true;

}

function updatePlots(plots) {

    const body = el("plotListBody");

    const count = el("plotCount");

    if (count) {
        count.textContent = Array.isArray(plots) && plots.length
            ? `${plots.length} registered`
            : "none registered";
    }

    if (!Array.isArray(plots) || plots.length === 0) {

        body.innerHTML = `
            <tr>
                <td colspan="3" class="empty-row">
                    No plots registered. Until one exists for
                    <code>ESP32_01</code>, the Backend rejects that device with
                    HTTP 422 and stores nothing.
                </td>
            </tr>
        `;

        return;
    }

    body.innerHTML = plots.map(plot => `

        <tr>
            <td>${escapeHtml(plot.name ?? "--")}</td>
            <td>${escapeHtml(plot.crop_type ?? "--")}</td>
            <td><code>${escapeHtml(plot.device_id ?? "--")}</code></td>
        </tr>

    `).join("");

}

function setupPlotForm() {

    const form = el("plotForm");

    if (!form) return;

    form.addEventListener("submit", async event => {

        event.preventDefault();

        const message = el("plotMessage");
        const submit = el("plotSubmit");

        const name = el("plotName").value.trim();
        const deviceId = el("plotDevice").value.trim();
        const crop = el("plotCrop").value;

        if (submit) submit.disabled = true;

        if (message) {
            message.textContent = "Registering...";
            message.className = "plot-message";
        }

        try {

            await createPlot(name, crop, deviceId);

            if (message) {
                message.textContent =
                    `Registered "${name}" (${crop}) for ${deviceId}.`;
                message.className = "plot-message success";
            }

            form.reset();
            el("plotDevice").value = "ESP32_01";

            // Populate the crop select again if reset cleared it.
            cropsPopulated = false;
            await loadCrops();

            const plots = await getPlots();
            updatePlots(plots);

        } catch (error) {

            if (message) {
                message.textContent = error.message;
                message.className = "plot-message error";
            }

        } finally {

            if (submit) submit.disabled = false;

        }

    });

}


// ==========================================
// LOAD DASHBOARD
//
// Phase 2B: allSettled instead of all. Previously one failing endpoint rejected
// the whole Promise.all, the catch block ran, and the entire dashboard - cards
// included - was marked "Backend Offline" even when five of six responses had
// arrived.
// ==========================================

const ENDPOINTS = [
    { key: "health", label: "health", call: getHealth },
    { key: "sensor", label: "latest sensor data", call: getLatestSensorData },
    { key: "history", label: "sensor history", call: getSensorHistory },
    { key: "irrigation", label: "irrigation history", call: getIrrigationHistory },
    { key: "water", label: "water usage", call: getWaterUsage },
    { key: "cropHealth", label: "crop health", call: getCropHealth },
    { key: "plots", label: "plots", call: getPlots }
];

async function loadDashboard() {

    if (loadInFlight) return;
    loadInFlight = true;

    try {

        console.log("Loading AgriGuard API data...");

        const settled = await Promise.allSettled(
            ENDPOINTS.map(endpoint => endpoint.call())
        );

        const data = {};
        const failures = [];

        settled.forEach((result, index) => {

            if (result.status === "fulfilled") {
                data[ENDPOINTS[index].key] = result.value;
            } else {
                failures.push(
                    `${ENDPOINTS[index].label}: ${result.reason.message}`
                );
            }

        });

        const okCount = settled.length - failures.length;

        // ----------------------------------
        // Update UI, each panel on its own data
        // ----------------------------------

        if (data.sensor !== undefined) {
            updateSensorUI(data.sensor);
            // The prediction travels on the reading, not on its own endpoint.
            updatePrediction(data.sensor);
        } else {
            setPredictionUnavailable("No sensor reading available.");
        }

        if (data.cropHealth !== undefined) {
            updateCropHealth(data.cropHealth);
        }

        if (data.water !== undefined) {
            updateWaterUsage(data.water);
        }

        if (data.irrigation !== undefined) {
            updateIrrigationHistory(data.irrigation);
        }

        if (data.plots !== undefined) {
            updatePlots(data.plots);
        }

        if (data.health !== undefined) {
            await loadCrops();
        }

        // ----------------------------------
        // Charts
        // ----------------------------------

        if (data.history !== undefined) {

            if (
                typeof createOrUpdateCharts ===
                "function"
            ) {

                createOrUpdateCharts(data.history);
            }

        }


        // ----------------------------------
        // Connection
        //
        // "Offline" now means the Backend could not be reached at all, which is
        // what health failing plus the latest reading failing means. A single
        // failing endpoint is reported as degraded instead.
        // ----------------------------------

        const reachable =
            data.health !== undefined ||
            data.sensor !== undefined;

        setConnectionStatus(reachable);

        const mode = el("dataMode");

        if (mode) {

            if (!reachable) {
                mode.textContent = "API ERROR";
                mode.className = "data-mode";
            } else if (failures.length > 0) {
                mode.textContent =
                    `PARTIAL (${okCount}/${settled.length})`;
                mode.className = "data-mode";
                console.warn(
                    "Some AgriGuard endpoints failed:", failures
                );
            } else {
                mode.textContent = "REAL API DATA";
                mode.className = "data-mode real";
            }

        }

        if (!reachable) {
            console.error(
                `AgriGuard dashboard cannot reach ${getApiBaseUrl()}:`,
                failures
            );
        }

    } finally {

        loadInFlight = false;

    }

}


function setPredictionUnavailable(reason) {

    el("predictionText").textContent =
        "Prediction unavailable";

    el("recommendationText").textContent = reason;

    el("predictionReason").textContent = "";

    el("predictionConfidence").textContent = "--";

    const sourceBadge = el("predictionSource");

    if (sourceBadge) {
        sourceBadge.textContent = "Unavailable";
        sourceBadge.className =
            "ai-source source-unavailable";
    }

}


// ==========================================
// MOBILE MENU
// ==========================================

el("menuBtn")?.addEventListener(
    "click",
    () => {

        el("sidebar")
            ?.classList
            .toggle("open");
    }
);


// ==========================================
// NAVIGATION
// ==========================================

document
    .querySelectorAll(".nav-link")
    .forEach(link => {

        link.addEventListener(
            "click",
            () => {

                document
                    .querySelectorAll(".nav-link")
                    .forEach(item =>
                        item.classList.remove("active")
                    );

                link.classList.add("active");

                el("sidebar")
                    ?.classList
                    .remove("open");
            }
        );
    });


setupPlotForm();


// ==========================================
// START DASHBOARD
// ==========================================

loadDashboard();


// ==========================================
// AUTO REFRESH
// ==========================================

setInterval(
    loadDashboard,
    REFRESH_MS
);