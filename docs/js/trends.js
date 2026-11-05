/* Complaint trends: aggregated counts only. Vanilla SVG line chart and bar chart. */
(function () {
  "use strict";
  var app = document.getElementById("tr-app");
  if (!app) return;
  var $ = function (id) { return document.getElementById(id); };
  function esc(s) { var d = document.createElement("div"); d.textContent = s; return d.innerHTML; }
  var NS = "http://www.w3.org/2000/svg";
  // one colour per system, fixed by system (never by rank); the second half is dashed so colour is not the only cue
  var COLORS = ["#1c1f24", "#7d8896", "#0072B2", "#009E73", "#CC79A7", "#8a5a00", "#56B4E9", "#332288", "#999933", "#6b3a1e"];
  var M, rec, on, make = -1, year = -1, tipEl = $("tr-tip"), series, maxY;

  function nf(x) { return x.toLocaleString("en-US"); }
  function monthLabel(m) { return ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][+m.slice(4) - 1] + " " + m.slice(0, 4); }

  // counts[system][month] for the current make / year filter
  function compute() {
    var S = M.classes.length, T = M.months.length, out = [], s, t, k;
    for (s = 0; s < S; s++) out.push(new Array(T).fill(0));
    for (t = 0; t < T; t++) for (s = 0; s < S; s++) {
      var g = t * S + s;
      for (k = rec.offsets[g]; k < rec.offsets[g + 1]; k++) {
        if (make >= 0 && rec.make[k] !== make) continue;
        if (year >= 0 && rec.year[k] !== year) continue;
        out[s][t] += rec.count[k];
      }
    }
    return out;
  }

  function topMakes(sys) {
    var tot = new Array(M.makes.length).fill(0), T = M.months.length, S = M.classes.length, t, k;
    for (t = 0; t < T; t++) {
      var g = t * S + sys;
      for (k = rec.offsets[g]; k < rec.offsets[g + 1]; k++) {
        if (year >= 0 && rec.year[k] !== year) continue;
        tot[rec.make[k]] += rec.count[k];
      }
    }
    return tot.map(function (v, i) { return [i, v]; }).filter(function (x) { return x[0] < M.makes.length - 1; }).sort(function (a, b) { return b[1] - a[1]; }).slice(0, 10);
  }

  function el(tag, attrs, parent) { var e = document.createElementNS(NS, tag); for (var k in attrs) e.setAttribute(k, attrs[k]); if (parent) parent.appendChild(e); return e; }

  function niceMax(v) { if (v <= 0) return 10; var p = Math.pow(10, Math.floor(Math.log10(v))), f = v / p; return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10) * p; }

  var W = 1000, H = 420, PL = 54, PR = 16, PT = 14, PB = 34;
  function size() { var w = $("tr-line").parentNode.clientWidth || 1000; W = Math.max(300, w); H = W < 600 ? 300 : 420; PL = W < 600 ? 42 : 54; }
  function x(t) { return PL + (W - PL - PR) * t / (M.months.length - 1); }
  function y(v) { return PT + (H - PT - PB) * (1 - v / maxY); }

  function drawLine() {
    size(); var svg = $("tr-line"); svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    while (svg.childNodes.length > 1) svg.removeChild(svg.lastChild);
    series = compute();
    var vis = M.classes.map(function (c, i) { return i; }).filter(function (i) { return on[i]; });
    var m = 0; vis.forEach(function (i) { m = Math.max(m, Math.max.apply(null, series[i])); });
    maxY = niceMax(m * 1.05);
    var g = el("g", {}, svg), i, v;
    for (i = 0; i <= 4; i++) {
      v = maxY * i / 4;
      el("line", { x1: PL, x2: W - PR, y1: y(v), y2: y(v), stroke: "#e4e1d8", "stroke-width": 1 }, g);
      var tx = el("text", { x: PL - 8, y: y(v) + 4, "text-anchor": "end" }, g); tx.textContent = nf(Math.round(v));
    }
    M.months.forEach(function (mm, t) {
      if (mm.slice(4) === "01") {
        el("line", { x1: x(t), x2: x(t), y1: H - PB, y2: H - PB + 5, stroke: "#7d8896" }, g);
        var lb = el("text", { x: x(t), y: H - 10, "text-anchor": "middle" }, g); lb.textContent = mm.slice(0, 4);
      }
    });
    el("line", { x1: PL, x2: W - PR, y1: H - PB, y2: H - PB, stroke: "#7d8896" }, g);
    vis.forEach(function (s) {
      var d = series[s].map(function (c, t) { return (t ? "L" : "M") + x(t).toFixed(1) + " " + y(c).toFixed(1); }).join(" ");
      el("path", { d: d, fill: "none", stroke: COLORS[s], "stroke-width": 2.4, "stroke-linejoin": "round", "stroke-dasharray": s >= 5 ? "7 4" : "none" }, g);
    });
    el("line", { id: "tr-cross", x1: 0, x2: 0, y1: PT, y2: H - PB, stroke: "#1c1f24", "stroke-width": 1, "stroke-dasharray": "3 3", visibility: "hidden" }, g);
    var tot = 0; vis.forEach(function (s) { tot += series[s].reduce(function (a, b) { return a + b; }, 0); });
    $("tr-summary").textContent = vis.length ? "Selected systems, " + (make >= 0 ? M.makes[make] : "all makes") + ", " + (year >= 0 ? (M.years[year] === 0 ? "model year before 2010 or not given" : "model year " + M.years[year]) : "all model years") + ": " + nf(tot) + " complaints." : "Switch on at least one system.";
    drawTable(vis);
  }

  function drawTable(vis) {
    var yrs = {}, y0;
    M.months.forEach(function (m, t) { y0 = m.slice(0, 4); yrs[y0] = yrs[y0] || vis.map(function () { return 0; }); vis.forEach(function (s, k) { yrs[y0][k] += series[s][t]; }); });
    var h = "<table class=\"mx-static\"><caption>Complaints per year for the selected systems and filters.</caption><thead><tr><th scope=\"col\">System</th>" + Object.keys(yrs).map(function (a) { return "<th scope=\"col\">" + a + "</th>"; }).join("") + "</tr></thead><tbody>";
    vis.forEach(function (s, k) { h += "<tr><th scope=\"row\">" + esc(M.plain_names[s]) + "</th>" + Object.keys(yrs).map(function (a) { return "<td>" + nf(yrs[a][k]) + "</td>"; }).join("") + "</tr>"; });
    $("tr-table").innerHTML = h + "</tbody></table>";
  }

  function chips() {
    $("tr-chips").innerHTML = M.classes.map(function (c, i) {
      return '<button type="button" class="chip" data-i="' + i + '" aria-pressed="' + (on[i] ? "true" : "false") + '" style="--c:' + COLORS[i] + '"><span class="sw' + (i >= 5 ? " dash" : "") + '"></span>' + esc(M.plain_names[i]) + "</button>";
    }).join("");
    $("tr-chips").querySelectorAll(".chip").forEach(function (b) {
      b.addEventListener("click", function () { var i = +b.getAttribute("data-i"); on[i] = !on[i]; b.setAttribute("aria-pressed", on[i] ? "true" : "false"); drawLine(); });
    });
  }

  function drawMakes() {
    var s = +$("tr-sys").value, top = topMakes(s), mx = top.length ? top[0][1] : 1;
    $("tr-makes").innerHTML = top.map(function (r) {
      return '<div class="bar-row"><span class="name">' + esc(M.makes[r[0]]) + '</span><span class="track"><span class="fill" style="display:block;width:' + (mx ? r[1] / mx * 100 : 0).toFixed(1) + '%"></span></span><span class="pct">' + nf(r[1]) + "</span></div>";
    }).join("") || '<p class="small">No complaints for this filter.</p>';
  }

  function hover(ev) {
    var svg = $("tr-line"), r = svg.getBoundingClientRect(), px = (ev.clientX - r.left) / r.width * W;
    var t = Math.max(0, Math.min(M.months.length - 1, Math.round((px - PL) / (W - PL - PR) * (M.months.length - 1))));
    var cross = $("tr-cross"); cross.setAttribute("x1", x(t)); cross.setAttribute("x2", x(t)); cross.setAttribute("visibility", "visible");
    var rows = M.classes.map(function (c, i) { return i; }).filter(function (i) { return on[i]; }).sort(function (a, b) { return series[b][t] - series[a][t]; });
    tipEl.innerHTML = "<b>" + monthLabel(M.months[t]) + "</b>" + rows.map(function (i) { return '<div class="row"><span><span class="dot" style="background:' + COLORS[i] + '"></span>' + esc(M.plain_names[i]) + "</span><b>" + nf(series[i][t]) + "</b></div>"; }).join("");
    tipEl.hidden = false;
    var wrap = svg.parentNode.getBoundingClientRect(), left = (x(t) / W) * r.width + 12;
    if (left + tipEl.offsetWidth > wrap.width) left = (x(t) / W) * r.width - tipEl.offsetWidth - 12;
    tipEl.style.left = Math.max(0, left) + "px"; tipEl.style.top = "8px";
  }
  function leave() { tipEl.hidden = true; var c = $("tr-cross"); if (c) c.setAttribute("visibility", "hidden"); }

  function fill(sel, items) { sel.innerHTML = items.map(function (a) { return '<option value="' + a[0] + '">' + esc(a[1]) + "</option>"; }).join(""); }

  Promise.all([fetch("data/trends.json").then(function (r) { return r.json(); }), fetch("data/trends.bin").then(function (r) { return r.arrayBuffer(); })]).then(function (a) {
    M = a[0]; var buf = a[1], L = M.layout, S = M.classes.length;
    rec = { offsets: new Uint32Array(buf, L.offsets, M.months.length * S + 1), make: new Uint8Array(buf, L.make, M.records), year: new Uint8Array(buf, L.year, M.records), count: new Uint16Array(buf, L.count, M.records) };
    on = M.classes.map(function (c, i) { return i < 4; });
    fill($("tr-make"), [[-1, "All makes"]].concat(M.makes.map(function (m, i) { return [i, m === "OTHER MAKES" ? "All other makes" : m]; })));
    fill($("tr-year"), [[-1, "All model years"]].concat(M.years.map(function (yv, i) { return [i, yv === 0 ? "Before 2010 or not given" : String(yv)]; })));
    fill($("tr-sys"), M.classes.map(function (c, i) { return [i, M.plain_names[i]]; }));
    $("tr-make").addEventListener("change", function () { make = +this.value; drawLine(); });
    $("tr-year").addEventListener("change", function () { year = +this.value; drawLine(); drawMakes(); });
    $("tr-sys").addEventListener("change", drawMakes);
    var svg = $("tr-line"); svg.addEventListener("pointermove", hover); svg.addEventListener("pointerleave", leave);
    var rt; window.addEventListener("resize", function () { clearTimeout(rt); rt = setTimeout(function () { if (!app.hidden) drawLine(); }, 150); });
    chips(); app.hidden = false; $("tr-fallback").hidden = true; drawLine(); drawMakes();
  }).catch(function () { $("tr-fallback").insertAdjacentHTML("afterbegin", '<p class="verdict miss">The data files could not be loaded.</p>'); });
})();
