/* Mix-up explorer: DistilBERT temporal confusion matrix as an HTML grid; click a cell to read real complaints. */
(function () {
  "use strict";
  var app = document.getElementById("mx-app");
  if (!app) return;
  var $ = function (id) { return document.getElementById(id); };
  function esc(s) { var d = document.createElement("div"); d.textContent = s; return d.innerHTML; }
  var D, selected = null, hideDiag = false;

  function shade(i, j) {
    var rowSum = D.matrix[i].reduce(function (a, b) { return a + b; }, 0), v = D.matrix[i][j];
    if (!v) return "";
    var share = v / rowSum;
    if (i === j) return "rgba(242,169,0," + (0.18 + 0.72 * share).toFixed(2) + ")";
    return "rgba(214,69,69," + Math.min(0.78, 0.14 + 3.2 * share).toFixed(2) + ")";
  }

  function grid() {
    var k = D.classes.length, h = '<table class="mx"><thead><tr><th class="axis" scope="col"><span class="nm">True ↓ / chosen →</span></th>';
    for (var j = 0; j < k; j++) h += '<th scope="col" title="' + esc(D.plain_names[j]) + '">' + (j + 1) + "</th>";
    h += "</tr></thead><tbody>";
    for (var i = 0; i < k; i++) {
      h += '<tr><th scope="row">' + (i + 1) + ' <span class="nm">' + esc(D.plain_names[i]) + "</span></th>";
      for (j = 0; j < k; j++) {
        var v = D.matrix[i][j], hide = hideDiag && i === j;
        h += '<td><button type="button" class="mx-cell ' + (i === j ? "diag" : "off") + (hide ? " hidden-diag" : "") + '" data-i="' + i + '" data-j="' + j + '"' + (v ? "" : " disabled") + ' aria-pressed="' + (selected && selected[0] === i && selected[1] === j ? "true" : "false") +
          '" aria-label="NHTSA filed under ' + esc(D.plain_names[i]) + ", model chose " + esc(D.plain_names[j]) + ": " + v + ' complaints"' + (v && !hide ? ' style="background:' + shade(i, j) + '"' : "") + ">" + v + "</button></td>";
      }
      h += "</tr>";
    }
    $("mx-grid").innerHTML = h + "</tbody></table><ol class=\"small mx-key\">" + D.plain_names.map(function (x) { return "<li>" + esc(x) + "</li>"; }).join("") + "</ol>";
    $("mx-grid").querySelectorAll(".mx-cell").forEach(function (b) {
      b.addEventListener("click", function () { select(+b.getAttribute("data-i"), +b.getAttribute("data-j")); });
      b.addEventListener("keydown", function (e) {
        var di = { ArrowUp: -1, ArrowDown: 1 }[e.key] || 0, dj = { ArrowLeft: -1, ArrowRight: 1 }[e.key] || 0;
        if (!di && !dj) return;
        var t = $("mx-grid").querySelector('.mx-cell[data-i="' + (+b.getAttribute("data-i") + di) + '"][data-j="' + (+b.getAttribute("data-j") + dj) + '"]');
        if (t) { e.preventDefault(); t.focus(); }
      });
    });
  }

  function select(i, j) {
    selected = [i, j];
    $("mx-grid").querySelectorAll(".mx-cell").forEach(function (b) { b.setAttribute("aria-pressed", +b.getAttribute("data-i") === i && +b.getAttribute("data-j") === j ? "true" : "false"); });
    var rowSum = D.matrix[i].reduce(function (a, b) { return a + b; }, 0), v = D.matrix[i][j], items = D.examples[i + "," + j] || [], ok = i === j;
    var h = "<h3>" + (ok ? "Correct: " + esc(D.plain_names[i]) : esc(D.plain_names[i]) + " put under " + esc(D.plain_names[j])) + "</h3><p>" + v + " of " + rowSum + " complaints NHTSA filed under " + esc(D.plain_names[i].toLowerCase()) + " (" + (v / rowSum * 100).toFixed(1) + "%). Showing " + items.length + (items.length === 1 ? " complaint" : " random complaints") + ".</p>";
    items.forEach(function (it) {
      h += '<div class="mx-card' + (ok ? " ok" : "") + '"><p>' + esc(it.text) + (it.cut ? " ..." : "") + '</p><div class="meta"><span>Model confidence ' + Math.round(it.conf * 100) + "%</span><span>Report #" + esc(it.id) + "</span></div></div>";
    });
    $("mx-detail").innerHTML = h;
  }

  $("mx-hide").addEventListener("click", function () {
    hideDiag = !hideDiag; this.setAttribute("aria-pressed", hideDiag ? "true" : "false"); this.textContent = hideDiag ? "Show the correct answers" : "Hide the correct answers";
    var sel = selected; grid(); if (sel) { selected = sel; $("mx-grid").querySelector('.mx-cell[data-i="' + sel[0] + '"][data-j="' + sel[1] + '"]').setAttribute("aria-pressed", "true"); }
  });

  fetch("data/mixups.json").then(function (r) { return r.json(); }).then(function (j) { D = j; grid(); app.hidden = false; $("mx-fallback").hidden = true; })
    .catch(function () { $("mx-fallback").insertAdjacentHTML("afterbegin", '<p class="verdict miss">The data file could not be loaded.</p>'); });
})();
