/* Beat the machine: 10 rounds, one complaint per system, precomputed predictions from the three models. */
(function () {
  "use strict";
  var app = document.getElementById("game-app");
  if (!app) return;
  var $ = function (id) { return document.getElementById(id); };
  var stage = $("g-stage"), roundEl = $("g-round"), live = $("g-live");
  var MODELS = [["mlp", "MLP"], ["bilstm", "BiLSTM"], ["distilbert", "DistilBERT"]];
  var data = null, rounds, idx, score;

  function esc(s) { var d = document.createElement("div"); d.textContent = s; return d.innerHTML; }
  function shuffle(a) { for (var i = a.length - 1; i > 0; i--) { var j = Math.floor(Math.random() * (i + 1)), t = a[i]; a[i] = a[j]; a[j] = t; } return a; }

  var ROUNDS = 10;
  function deal() {
    // each round: pick a system at random, then an unused complaint of that system (systems may repeat; complaints never do within a game)
    var byClass = data.classes.map(function () { return []; });
    data.items.forEach(function (it) { byClass[it.label].push(it); });
    byClass.forEach(shuffle);
    rounds = [];
    while (rounds.length < ROUNDS) {
      var open = byClass.map(function (l, c) { return l.length ? c : -1; }).filter(function (c) { return c >= 0; });
      var c = open[Math.floor(Math.random() * open.length)];
      rounds.push(byClass[c].pop());
    }
    idx = 0; score = { you: 0, mlp: 0, bilstm: 0, distilbert: 0 };
    show();
  }

  function board(done) {
    live.innerHTML = '<span class="you">You ' + score.you + "</span>" + MODELS.map(function (m) { return "<span>" + m[1] + " " + score[m[0]] + "</span>"; }).join("");
    live.setAttribute("aria-label", "Score after " + done + " rounds");
  }

  function show() {
    var it = rounds[idx];
    roundEl.textContent = "Round " + (idx + 1) + " of " + rounds.length;
    board(idx);
    var html = '<p class="field-label" id="g-q">Which system failed?</p><blockquote class="complaint" style="margin-left:0">' + esc(it.text) + '</blockquote><div class="choices" role="group" aria-labelledby="g-q">';
    data.plain_names.forEach(function (nm, c) { html += '<button class="choice" type="button" data-c="' + c + '">' + esc(nm) + "</button>"; });
    stage.innerHTML = html + "</div>";
    stage.querySelectorAll(".choice").forEach(function (b) { b.addEventListener("click", function () { pick(+b.getAttribute("data-c")); }); });
  }

  function pick(c) {
    var it = rounds[idx];
    var ok = c === it.label; if (ok) score.you++;
    MODELS.forEach(function (m) { if (it[m[0]].pred === it.label) score[m[0]]++; });
    board(idx + 1);
    stage.querySelectorAll(".choice").forEach(function (b) {
      var k = +b.getAttribute("data-c"); b.disabled = true;
      if (k === it.label) { b.classList.add("is-true"); b.insertAdjacentHTML("beforeend", '<span class="tag">NHTSA</span>'); }
      else if (k === c) { b.classList.add("is-yours-wrong"); b.insertAdjacentHTML("beforeend", '<span class="tag">you</span>'); }
    });
    var rows = '<tr><td>You</td><td>' + esc(data.plain_names[c]) + '</td><td class="conf"></td><td class="' + (ok ? "ok" : "no") + '">' + (ok ? "right" : "wrong") + "</td></tr>";
    MODELS.forEach(function (m) {
      var g = it[m[0]], good = g.pred === it.label;
      rows += "<tr><td>" + m[1] + "</td><td>" + esc(data.plain_names[g.pred]) + "</td><td class=\"conf\">" + Math.round(g.conf * 100) + '%</td><td class="' + (good ? "ok" : "no") + '">' + (good ? "right" : "wrong") + "</td></tr>";
    });
    var last = idx === rounds.length - 1;
    stage.insertAdjacentHTML("beforeend", '<div class="reveal"><p class="verdict' + (ok ? "" : " miss") + '">NHTSA filed this complaint under <strong>' + esc(data.plain_names[it.label]) + "</strong>. " + (ok ? "You got it." : "You missed it.") +
      '</p><table><thead><tr><th>Who</th><th>Answer</th><th class="conf">Confidence</th><th>Result</th></tr></thead><tbody>' + rows + '</tbody></table></div><div class="btns"><button class="btn primary" id="g-next" type="button">' + (last ? "See your score" : "Next complaint") + "</button></div>");
    var nb = $("g-next"); nb.addEventListener("click", function () { idx++; if (idx < rounds.length) show(); else finish(); }); nb.focus();
  }

  function finish() {
    var n = rounds.length, rows = [["you", "You"]].concat(MODELS);
    var beaten = MODELS.filter(function (m) { return score.you > score[m[0]]; }).map(function (m) { return m[1]; });
    var tied = MODELS.filter(function (m) { return score.you === score[m[0]]; }).map(function (m) { return m[1]; });
    var higher = MODELS.length - beaten.length - tied.length;
    var msg = beaten.length === 3 ? "You beat all three models." : beaten.length ? "You beat " + beaten.join(" and ") + "." : "";
    if (tied.length) msg += (msg ? " " : "") + "You tied with " + tied.join(" and ") + ".";
    if (higher === 3) msg = "All three models scored higher than you this time.";
    else if (higher) msg += " " + higher + (higher === 1 ? " model" : " models") + " scored higher.";
    roundEl.textContent = "Game over"; board(n);
    var h = '<div class="final"><h2>You got ' + score.you + " of " + n + " right</h2><p>" + msg + '</p><div class="final-board">';
    rows.forEach(function (r) { h += '<div class="final-row' + (r[0] === "you" ? " you" : "") + '"><span class="nm">' + r[1] + '</span><span class="tr"><span class="fl" style="width:' + (score[r[0]] / n * 100) + '%"></span></span><span class="sc">' + score[r[0]] + "/" + n + "</span></div>"; });
    stage.innerHTML = h + '</div><p class="small">Ten complaints are too few to say who is better. The full test results are on the <a href="results.html">Results</a> page.</p><div class="btns"><button class="btn primary" id="g-again" type="button">Play again</button></div></div>';
    $("g-again").addEventListener("click", deal); $("g-again").focus();
  }

  fetch("data/sample_complaints.json").then(function (r) { return r.json(); }).then(function (j) { data = j; app.hidden = false; deal(); })
    .catch(function () { app.hidden = false; stage.innerHTML = '<p class="verdict miss">The complaints could not be loaded. Check your connection and reload the page.</p>'; });
})();
