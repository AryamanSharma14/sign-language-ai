// web/static/js/dashboard.js
const HISTORY_MAX = 50;

// ── Chart setup ────────────────────────────────────────────────────────────────
const ctx = document.getElementById("metrics-chart").getContext("2d");
const metricsChart = new Chart(ctx, {
  type: "line",
  data: {
    labels: Array(60).fill(""),
    datasets: [
      {
        label: "FPS",
        data: Array(60).fill(null),
        borderColor: "#58a6ff",
        borderWidth: 1.5,
        pointRadius: 0,
        tension: 0.3,
        yAxisID: "yFps",
      },
      {
        label: "Inference ms",
        data: Array(60).fill(null),
        borderColor: "#3fb950",
        borderWidth: 1.5,
        pointRadius: 0,
        tension: 0.3,
        yAxisID: "yMs",
      },
    ],
  },
  options: {
    animation: false,
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { labels: { color: "#8b949e", font: { size: 11 } } } },
    scales: {
      x: { display: false },
      yFps: {
        position: "left",
        min: 0,
        ticks: { color: "#58a6ff", font: { size: 10 } },
        grid: { color: "#21262d" },
      },
      yMs: {
        position: "right",
        min: 0,
        ticks: { color: "#3fb950", font: { size: 10 } },
        grid: { drawOnChartArea: false },
      },
    },
  },
});

function pushMetric(fps, inferenceMs) {
  metricsChart.data.datasets[0].data.shift();
  metricsChart.data.datasets[0].data.push(fps);
  metricsChart.data.datasets[1].data.shift();
  metricsChart.data.datasets[1].data.push(inferenceMs);
  metricsChart.update("none");
}

// ── Gauge ──────────────────────────────────────────────────────────────────────
const gaugeFill = document.getElementById("gauge-fill");
const confValue = document.getElementById("confidence-value");

function updateGauge(conf) {
  const pct = Math.round(conf * 100);
  gaugeFill.style.width = pct + "%";
  confValue.textContent = conf.toFixed(2);
  gaugeFill.className = "gauge-fill";
  if (conf < 0.6) gaugeFill.classList.add("low");
  else if (conf < 0.8) gaugeFill.classList.add("med");
}

// ── History ────────────────────────────────────────────────────────────────────
const historyBody = document.getElementById("history-body");

function addHistoryRow(ts, gesture, command, conf) {
  const tr = document.createElement("tr");
  tr.innerHTML = `<td>${ts}</td><td><strong>${gesture}</strong></td><td>${command}</td><td>${conf.toFixed(2)}</td>`;
  historyBody.insertBefore(tr, historyBody.firstChild);
  while (historyBody.rows.length > HISTORY_MAX) {
    historyBody.deleteRow(historyBody.rows.length - 1);
  }
}

// ── WebSocket ──────────────────────────────────────────────────────────────────
const gestureLabel = document.getElementById("gesture-label");
const gestureCommand = document.getElementById("gesture-command");

function connect() {
  const ws = new WebSocket(`ws://${location.host}/ws`);

  ws.onmessage = (evt) => {
    const data = JSON.parse(evt.data);
    gestureLabel.textContent = data.gesture || "--";
    gestureCommand.textContent = data.command || "--";
    updateGauge(data.confidence || 0);
    pushMetric(data.fps || 0, data.inference_ms || 0);
    addHistoryRow(data.timestamp, data.gesture, data.command, data.confidence);
  };

  ws.onclose = () => setTimeout(connect, 2000);  // auto-reconnect
  ws.onerror = () => ws.close();
}

connect();
