let moistureChart;
let temperatureChart;
let waterChart;


// ==========================================
// CHART OPTIONS
// ==========================================

function opts() {

    return {
        responsive: true,
        maintainAspectRatio: false,

        plugins: {
            legend: {
                display: false
            }
        },

        scales: {
            x: {
                grid: {
                    display: false
                },

                ticks: {
                    color: "#6B7C87",
                    font: {
                        size: 10
                    }
                }
            },

            y: {
                beginAtZero: true,

                grid: {
                    color: "#E8EFED"
                },

                ticks: {
                    color: "#6B7C87",
                    font: {
                        size: 10
                    }
                }
            }
        }
    };
}


// ==========================================
// FORMAT CHART TIME
// ==========================================

function formatChartTime(timestamp) {

    if (!timestamp) {
        return "--";
    }

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
// CREATE / UPDATE CHARTS
// ==========================================

// Phase 2B: if Chart.js failed to load from both the vendored copy and the
// CDN fallback, every chart stays absent. The rest of the dashboard is built
// from plain DOM text and still works, so this returns quietly instead of
// throwing and taking the sensor cards down with it.
function chartsAvailable() {

    return typeof Chart !== "undefined";

}

function createOrUpdateCharts(history) {

    if (!chartsAvailable()) {

        console.warn(
            "Chart.js is not loaded; charts are disabled."
        );

        return;

    }

    if (!Array.isArray(history)) {

        return;

    }


    // --------------------------------------
    // Sort by time
    // --------------------------------------

    const sortedHistory = [...history].sort(
        (a, b) =>
            new Date(a.timestamp) -
            new Date(b.timestamp)
    );


    // --------------------------------------
    // Labels
    // --------------------------------------

    const labels = sortedHistory.map(
        item => formatChartTime(item.timestamp)
    );


    // --------------------------------------
    // Soil Moisture
    // --------------------------------------

    const moistureData = sortedHistory.map(
        item =>
            Number(item.soil_moisture ?? 0)
    );


    // --------------------------------------
    // Temperature
    // --------------------------------------

    const temperatureData = sortedHistory.map(
        item =>
            Number(item.temperature ?? 0)
    );


    // --------------------------------------
    // Water Flow
    // Phase 2B: `water_flow` is an INSTANTANEOUS RATE in L/min as computed by
    // the firmware, not a volume in litres. It is therefore plotted here with
    // the card labelled "Water Flow" / L/min. Cumulative litres are shown on
    // the Total Water Used card instead, and come from the Backend's completed
    // irrigation events - never invented from this series.
    // --------------------------------------

    const waterData = sortedHistory.map(
        item =>
            Number(item.water_flow ?? 0)
    );


    // --------------------------------------
    // Destroy old charts
    // --------------------------------------

    if (moistureChart) {
        moistureChart.destroy();
    }

    if (temperatureChart) {
        temperatureChart.destroy();
    }

    if (waterChart) {
        waterChart.destroy();
    }


    // ======================================
    // SOIL MOISTURE CHART
    // ======================================

    const moistureCanvas =
        document.getElementById("moistureChart");

    if (moistureCanvas) {

        moistureChart = new Chart(
            moistureCanvas,
            {
                type: "line",

                data: {
                    labels: labels,

                    datasets: [
                        {
                            label: "Soil Moisture",

                            data: moistureData,

                            borderColor: "#0F766E",

                            backgroundColor:
                                "rgba(15,118,110,.1)",

                            fill: true,

                            tension: 0.35,

                            pointRadius: 3
                        }
                    ]
                },

                options: opts()
            }
        );
    }


    // ======================================
    // TEMPERATURE CHART
    // ======================================

    const temperatureCanvas =
        document.getElementById(
            "temperatureChart"
        );

    if (temperatureCanvas) {

        temperatureChart = new Chart(
            temperatureCanvas,
            {
                type: "line",

                data: {
                    labels: labels,

                    datasets: [
                        {
                            label: "Temperature",

                            data: temperatureData,

                            borderColor: "#2563EB",

                            backgroundColor:
                                "rgba(37,99,235,.08)",

                            fill: true,

                            tension: 0.35,

                            pointRadius: 3
                        }
                    ]
                },

                options: opts()
            }
        );
    }


    // ======================================
    // WATER CONSUMPTION CHART
    // ======================================

    const waterCanvas =
        document.getElementById("waterChart");

    if (waterCanvas) {

        waterChart = new Chart(
            waterCanvas,
            {
                type: "bar",

                data: {
                    labels: labels,

                    datasets: [
                        {
                            label: "Water Flow",

                            data: waterData,

                            backgroundColor: "#22C55E",

                            borderRadius: 6
                        }
                    ]
                },

                options: opts()
            }
        );
    }
}