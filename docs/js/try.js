/* Try it: runs the MLP from mlp.js in the browser. Files are fetched only when this page opens. */
(function () {
  "use strict";
  var app = document.getElementById("try-app");
  if (!app || !window.MLP) return;
  app.hidden = false;
  var $ = function (id) { return document.getElementById(id); };
  var text = $("try-text"), go = $("try-go"), real = $("try-real"), clear = $("try-clear"), out = $("try-out"), src = $("try-source");
  var model = null, sample = null, loaded = null;

  function mb(x) { return (x / 1e6).toFixed(1); }

  function fetchWithProgress(url, onBytes) {
    return fetch(url).then(function (r) {
      if (!r.ok) throw new Error(url + ": " + r.status);
      var reader = r.body && r.body.getReader ? r.body.getReader() : null;
      if (!reader) return r.arrayBuffer().then(function (b) { onBytes(b.byteLength); return b; });
      var chunks = [], got = 0;
      return (function pump() {
        return reader.read().then(function (d) {
          if (d.done) {
            var all = new Uint8Array(got), o = 0;
            chunks.forEach(function (c) { all.set(c, o); o += c.length; });
            return all.buffer;
          }
          chunks.push(d.value); got += d.value.length; onBytes(d.value.length);
          return pump();
        });
      })();
    });
  }

  function start() {
    var total = 0, done = 0, bar = $("try-progress"), label = $("try-load-text");
    function upd(n) { done += n; bar.value = total ? Math.min(100, done / total * 100) : 0; label.textContent = "Loading the model: " + mb(done) + " of " + mb(total) + " MB"; }
    fetchWithProgress("data/mlp_vocab.json", function () {}).then(function (vb) {
      var meta = JSON.parse(new TextDecoder().decode(vb));
      total = vb.byteLength + meta.weights_bytes; done = vb.byteLength; upd(0);
      return fetchWithProgress("data/mlp_weights.bin", upd).then(function (wb) { model = MLP.load(meta, wb); });
    }).then(function () {
      $("try-load-text").textContent = "Model ready (" + mb(done) + " MB downloaded)."; $("try-progress").hidden = true;
      [text, go, real, clear].forEach(function (e) { e.disabled = false; });
    }).catch(function (e) {
      $("try-load-text").textContent = "The model files could not be loaded (" + e.message + "). Check your connection and reload the page.";
    });
  }

  function esc(s) { var d = document.createElement("div"); d.textContent = s; return d.innerHTML; }

  function render(t) {
    var p = MLP.predict(model, t), order = Array.prototype.map.call(p, function (v, i) { return i; }).sort(function (a, b) { return p[b] - p[a]; });
    var names = model.meta.plain_names, top = order[0];
    var html = '<div class="bars" role="list" aria-label="Top three systems">';
    order.slice(0, 3).forEach(function (c, k) {
      html += '<div class="bar-row' + (k === 0 ? " top" : "") + '" role="listitem"><span class="name">' + esc(names[c]) + '</span><span class="track"><span class="fill" style="display:block;width:' + (p[c] * 100).toFixed(1) + '%"></span></span><span class="pct">' + (p[c] * 100).toFixed(0) + '%</span></div>';
    });
    html += "</div>";
    if (loaded) {
      var ok = loaded.label === top;
      html += '<p class="verdict' + (ok ? "" : " miss") + '">NHTSA filed this complaint under <strong>' + esc(names[loaded.label]) + "</strong>. The model " + (ok ? "got it right." : "got it wrong.") + "</p>";
    }
    var words = t.split(/\s+/).filter(Boolean);
    if (words.length > 300) {
      html += '<p class="small">Word highlights are shown for texts up to 300 words. This one has ' + words.length + ".</p>";
      out.innerHTML = html; return;
    }
    var occ = MLP.occlusion(model, t, top), mx = 1e-9;
    occ.importance.forEach(function (v) { mx = Math.max(mx, Math.abs(v)); });
    html += '<div class="legend"><span><span class="swatch" style="background:#f2a900"></span>pushes toward the top answer</span><span><span class="swatch" style="background:#8a929c"></span>pushes away from it</span></div><p class="wi-text">';
    occ.words.forEach(function (w, i) {
      var v = occ.importance[i], r = Math.abs(v) / mx;
      html += r < 0.08 ? "<span title=\"" + v.toFixed(2) + "\">" + esc(w) + "</span> "
        : "<span class=\"" + (v > 0 ? "sup" : "arg") + "\" style=\"--a:" + (0.2 + 0.7 * r).toFixed(2) + "\" title=\"" + (v > 0 ? "+" : "") + v.toFixed(2) + "\">" + esc(w) + "</span> ";
    });
    html += '</p><p class="small">Hover or tap a word to see its value: the change in the log-odds of the top answer when that word is removed.</p>';
    out.innerHTML = html;
  }

  function run() {
    var t = text.value.trim();
    if (t.split(/\s+/).filter(Boolean).length < 3) { out.innerHTML = '<p class="verdict miss">Write at least a few words so there is something to read.</p>'; return; }
    out.innerHTML = '<p class="small">Working...</p>';
    setTimeout(function () { render(t); }, 20);
  }

  go.addEventListener("click", function () { loaded = loaded && loaded.text === text.value ? loaded : null; if (!loaded) src.hidden = true; run(); });
  clear.addEventListener("click", function () { text.value = ""; out.innerHTML = ""; loaded = null; src.hidden = true; text.focus(); });
  real.addEventListener("click", function () {
    function pick() {
      var it = sample.items[Math.floor(Math.random() * sample.items.length)];
      loaded = { text: it.text, label: it.label };
      text.value = it.text;
      src.hidden = false;
      src.textContent = "Real complaint, report #" + it.id + ", received " + it.received.slice(0, 4) + "-" + it.received.slice(4, 6) + " (" + it.year + " " + it.make + "). Masked text from data/clean/complaints_temporal.csv, test split.";
      run();
    }
    if (sample) return pick();
    real.disabled = true;
    fetch("data/sample_complaints.json").then(function (r) { return r.json(); }).then(function (j) { sample = j; real.disabled = false; pick(); })
      .catch(function () { real.disabled = false; out.innerHTML = '<p class="verdict miss">The complaint sample could not be loaded.</p>'; });
  });
  start();
})();
