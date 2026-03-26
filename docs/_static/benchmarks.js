"use strict";

(function () {
  function loadData(callback) {
    if (window.BENCHMARK_DATA) {
      callback(window.BENCHMARK_DATA);
    } else {
      document.getElementById("benchmark-charts").innerHTML =
        "<p>Benchmark data not yet available. Charts will appear after the first benchmark run on <code>main</code> completes.</p>";
    }
  }

  function isDarkMode() {
    // Furo uses data-theme on <body>; also check prefers-color-scheme
    var body = document.body;
    if (body.dataset.theme === "dark") return true;
    if (body.dataset.theme === "light") return false;
    return window.matchMedia("(prefers-color-scheme: dark)").matches;
  }

  function getThemeColors() {
    var dark = isDarkMode();
    return {
      text: dark ? "#cfd0d0" : "#4a4a4a",
      grid: dark ? "#3a3a3a" : "#e0e0e0",
      bg: dark ? "#1a1c1e" : "#ffffff",
    };
  }

  // Chart.js v2 compatible palette
  var COLORS = [
    "#3572a5",
    "#e4573d",
    "#2ea043",
    "#d29922",
    "#8957e5",
    "#1f6feb",
  ];

  // Parse mean duration from the extra field (e.g. "mean: 339.88 msec\nrounds: 5")
  function parseMeanSeconds(bench) {
    if (!bench.extra) return null;
    var match = bench.extra.match(/mean:\s*([\d.]+)\s*([\w]+)/);
    if (!match) return null;
    var value = parseFloat(match[1]);
    var unit = match[2];
    if (unit === "msec") return value / 1000;
    if (unit === "usec") return value / 1000000;
    if (unit === "nsec") return value / 1000000000;
    if (unit === "sec") return value;
    return null;
  }

  function renderCharts(data) {
    var container = document.getElementById("benchmark-charts");
    if (!container) return;
    container.innerHTML = "";

    // Collect benchmarks per test case
    var entries = data.entries;
    var allBenches = new Map();

    Object.keys(entries).forEach(function (name) {
      entries[name].forEach(function (entry) {
        entry.benches.forEach(function (bench) {
          var arr = allBenches.get(bench.name);
          if (!arr) {
            arr = [];
            allBenches.set(bench.name, arr);
          }
          arr.push({
            commit: entry.commit,
            date: entry.date,
            bench: bench,
          });
        });
      });
    });

    var colorIdx = 0;
    allBenches.forEach(function (dataset, benchName) {
      var wrapper = document.createElement("div");
      wrapper.style.marginBottom = "1rem";
      container.appendChild(wrapper);

      var title = document.createElement("p");
      title.textContent = benchName;
      title.style.textAlign = "center";
      title.style.fontWeight = "normal";
      title.style.margin = "0.5rem 0 0 0";
      wrapper.appendChild(title);

      var canvas = document.createElement("canvas");
      canvas.style.maxWidth = "100%";
      wrapper.appendChild(canvas);

      var color = COLORS[colorIdx % COLORS.length];
      colorIdx++;

      var theme = getThemeColors();

      new Chart(canvas, {
        type: "line",
        data: {
          labels: dataset.map(function (d) {
            return d.commit.id.slice(0, 7);
          }),
          datasets: [
            {
              label: benchName,
              data: dataset.map(function (d) {
                var secs = parseMeanSeconds(d.bench);
                return secs !== null ? secs : 1.0 / d.bench.value;
              }),
              borderColor: color,
              backgroundColor: color + "30",
              borderWidth: 2,
              pointRadius: 3,
              pointHoverRadius: 6,
              fill: true,
            },
          ],
        },
        options: {
          responsive: true,
          aspectRatio: window.innerWidth >= 768 ? 3 : 2,
          legend: { display: false },
          scales: {
            xAxes: [
              {
                scaleLabel: {
                  display: true,
                  labelString: "commit",
                  fontColor: theme.text,
                },
                ticks: { fontColor: theme.text },
                gridLines: { color: theme.grid },
              },
            ],
            yAxes: [
              {
                scaleLabel: {
                  display: true,
                  labelString: "seconds",
                  fontColor: theme.text,
                },
                ticks: {
                  beginAtZero: true,
                  fontColor: theme.text,
                },
                gridLines: { color: theme.grid },
              },
            ],
          },
          tooltips: {
            callbacks: {
              afterTitle: function (items) {
                var idx = items[0].index;
                var d = dataset[idx];
                return (
                  "\n" +
                  d.commit.message +
                  "\n\n" +
                  d.commit.timestamp +
                  " by @" +
                  d.commit.committer.username
                );
              },
              label: function (item) {
                var label = item.value;
                var b = dataset[item.index].bench;
                label += " " + b.unit;
                if (b.range) label += " (" + b.range + ")";
                return label;
              },
            },
          },
          onClick: function (_event, activeElems) {
            if (activeElems.length === 0) return;
            var idx = activeElems[0]._index;
            window.open(dataset[idx].commit.url, "_blank");
          },
        },
      });
    });

    // Last update info
    var info = document.createElement("p");
    info.className = "small";
    info.style.textAlign = "center";
    info.style.opacity = "0.7";
    info.textContent = "Last updated: " + new Date(data.lastUpdate).toLocaleString();
    container.appendChild(info);
  }

  // Re-render on Furo theme toggle
  var observer = new MutationObserver(function (mutations) {
    mutations.forEach(function (m) {
      if (m.attributeName === "data-theme" && window.BENCHMARK_DATA) {
        renderCharts(window.BENCHMARK_DATA);
      }
    });
  });

  document.addEventListener("DOMContentLoaded", function () {
    observer.observe(document.body, { attributes: true });
    loadData(renderCharts);
  });
})();
