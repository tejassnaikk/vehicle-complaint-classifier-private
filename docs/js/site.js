/* Small vanilla JS for What Failed on This Car? Everything works without it:
   figures are plain links to the full image, the matrices are all shown, details are native <details>. */
(function () {
  "use strict";
  document.documentElement.classList.add("js");

  /* ---- keep the current page visible in the horizontally scrolling nav (phones) ---- */
  var cur = document.querySelector('nav.main a[aria-current="page"]');
  if (cur && cur.parentNode && cur.parentNode.parentNode) {
    var list = cur.parentNode.parentNode;
    list.scrollLeft = Math.max(0, cur.offsetLeft - (list.clientWidth - cur.offsetWidth) / 2);
  }

  /* ---- click-to-enlarge lightbox ---- */
  var box = null, lastFocus = null;
  function closeBox() {
    if (!box) return;
    document.body.removeChild(box);
    box = null;
    document.removeEventListener("keydown", onKey);
    if (lastFocus && lastFocus.focus) lastFocus.focus();
  }
  function onKey(e) {
    if (e.key === "Escape") closeBox();
    if (e.key === "Tab" && box) { e.preventDefault(); box.querySelector(".lb-close").focus(); }
  }
  function openBox(link) {
    var img = link.querySelector("img");
    lastFocus = link;
    box = document.createElement("div");
    box.className = "lightbox";
    box.setAttribute("role", "dialog");
    box.setAttribute("aria-modal", "true");
    box.setAttribute("aria-label", "Enlarged figure");
    var big = document.createElement("img");
    big.src = link.getAttribute("href");
    big.alt = img ? img.alt : "";
    var cap = document.createElement("div");
    cap.className = "lb-cap";
    var fc = link.closest("figure");
    var c = fc ? fc.querySelector("figcaption") : null;
    cap.textContent = c ? c.textContent : (img ? img.alt : "");
    var btn = document.createElement("button");
    btn.className = "lb-close";
    btn.type = "button";
    btn.textContent = "Close (Esc)";
    btn.addEventListener("click", closeBox);
    box.appendChild(btn); box.appendChild(big); box.appendChild(cap);
    box.addEventListener("click", function (e) { if (e.target === box) closeBox(); });
    document.body.appendChild(box);
    document.addEventListener("keydown", onKey);
    btn.focus();
  }
  document.addEventListener("click", function (e) {
    var a = e.target.closest ? e.target.closest("a.zoom") : null;
    if (!a || e.metaKey || e.ctrlKey || e.shiftKey || e.button !== 0) return;
    e.preventDefault();
    openBox(a);
  });

  /* ---- confusion matrix viewer: model tabs, split tabs, counts/share toggle ---- */
  function segment(label, options, current, onPick, role) {
    var group = document.createElement("div");
    group.className = "cm-group";
    var span = document.createElement("span");
    span.textContent = label;
    var seg = document.createElement("div");
    seg.className = "seg";
    seg.setAttribute("role", role === "tab" ? "tablist" : "group");
    seg.setAttribute("aria-label", label);
    options.forEach(function (o) {
      var b = document.createElement("button");
      b.type = "button";
      b.textContent = o.text;
      b.dataset.value = o.value;
      b.setAttribute(role === "tab" ? "aria-selected" : "aria-pressed", String(o.value === current));
      if (role === "tab") b.setAttribute("role", "tab");
      b.addEventListener("click", function () { onPick(o.value); });
      seg.appendChild(b);
    });
    group.appendChild(span); group.appendChild(seg);
    return { el: group, seg: seg, role: role };
  }
  document.querySelectorAll(".cm-viewer").forEach(function (v) {
    var figs = Array.prototype.slice.call(v.querySelectorAll("figure[data-model]"));
    var state = { model: "distilbert", split: "temporal", kind: "normalized" };
    var controls = document.createElement("div");
    controls.className = "cm-controls";
    var status = document.createElement("div");
    status.className = "small";
    status.setAttribute("aria-live", "polite");
    function render() {
      figs.forEach(function (f) {
        f.hidden = !(f.dataset.model === state.model && f.dataset.split === state.split && f.dataset.kind === state.kind);
      });
      [mTabs, sTabs, kTabs].forEach(function (s) {
        Array.prototype.forEach.call(s.seg.children, function (b) {
          var key = s === mTabs ? "model" : s === sTabs ? "split" : "kind";
          b.setAttribute(s.role === "tab" ? "aria-selected" : "aria-pressed", String(b.dataset.value === state[key]));
        });
      });
      status.textContent = "Showing " + state.model + ", " + state.split + " split, " + (state.kind === "normalized" ? "share of each true class" : "counts") + ".";
    }
    var mTabs = segment("Model", [{ text: "MLP", value: "mlp" }, { text: "BiLSTM", value: "bilstm" }, { text: "DistilBERT", value: "distilbert" }], state.model, function (x) { state.model = x; render(); }, "tab");
    var sTabs = segment("Split", [{ text: "Temporal (main)", value: "temporal" }, { text: "Random", value: "random" }], state.split, function (x) { state.split = x; render(); }, "tab");
    var kTabs = segment("Show", [{ text: "Share of true class", value: "normalized" }, { text: "Counts", value: "counts" }], state.kind, function (x) { state.kind = x; render(); }, "toggle");
    controls.appendChild(mTabs.el); controls.appendChild(sTabs.el); controls.appendChild(kTabs.el);
    v.insertBefore(controls, v.firstChild);
    v.appendChild(status);
    render();
  });
})();
