/* Browser version of the trained MLP: sklearn-style TF-IDF (1-2 grams, l2 norm) and the three dense layers.
   Weights are int8 with one float32 scale per input row. Works in a page (window.MLP) and in Node (module.exports). */
(function (root) {
  "use strict";
  var TOKEN = /[\p{L}\p{N}_]{2,}/gu; // same tokens as sklearn's (?u)\b\w\w+\b on lowercase text

  function load(meta, buffer) {
    var index = new Map(), i;
    for (i = 0; i < meta.terms.length; i++) index.set(meta.terms[i], i);
    var layers = meta.layers.map(function (L) {
      return {
        nIn: L.in, nOut: L.out,
        w: new Int8Array(buffer, L.w_offset, L.in * L.out),
        scale: new Float32Array(buffer, L.scale_offset, L.in),
        bias: new Float32Array(buffer, L.bias_offset, L.out)
      };
    });
    return { meta: meta, index: index, idf: Float32Array.from(meta.idf), layers: layers };
  }

  function tokens(text) { return (text.toLowerCase().match(TOKEN) || []); }

  // sparse TF-IDF vector: {idx: Int32Array, val: Float32Array}
  function vectorize(model, text) {
    var tk = tokens(text), counts = new Map(), k, j, t;
    for (k = 0; k < tk.length; k++) {
      t = tk[k]; j = model.index.get(t);
      if (j !== undefined) counts.set(j, (counts.get(j) || 0) + 1);
      if (k + 1 < tk.length) { j = model.index.get(t + " " + tk[k + 1]); if (j !== undefined) counts.set(j, (counts.get(j) || 0) + 1); }
    }
    var idx = Int32Array.from(counts.keys()).sort(), val = new Float32Array(idx.length), norm = 0;
    for (k = 0; k < idx.length; k++) { val[k] = counts.get(idx[k]) * model.idf[idx[k]]; norm += val[k] * val[k]; }
    norm = Math.sqrt(norm) || 1;
    for (k = 0; k < val.length; k++) val[k] /= norm;
    return { idx: idx, val: val };
  }

  function forward(model, x) {
    var L0 = model.layers[0], h = new Float32Array(L0.bias), k, f, j, base, s;
    for (k = 0; k < x.idx.length; k++) {
      f = x.idx[k]; s = x.val[k] * L0.scale[f]; base = f * L0.nOut;
      for (j = 0; j < L0.nOut; j++) h[j] += s * L0.w[base + j];
    }
    for (j = 0; j < h.length; j++) if (h[j] < 0) h[j] = 0;
    for (var li = 1; li < model.layers.length; li++) {
      var L = model.layers[li], o = new Float32Array(L.bias), i;
      for (i = 0; i < L.nIn; i++) {
        if (h[i] === 0) continue;
        s = h[i] * L.scale[i]; base = i * L.nOut;
        for (j = 0; j < L.nOut; j++) o[j] += s * L.w[base + j];
      }
      if (li < model.layers.length - 1) for (j = 0; j < o.length; j++) if (o[j] < 0) o[j] = 0;
      h = o;
    }
    return h; // logits
  }

  function softmax(z) {
    var m = -Infinity, s = 0, p = new Float64Array(z.length), i;
    for (i = 0; i < z.length; i++) if (z[i] > m) m = z[i];
    for (i = 0; i < z.length; i++) { p[i] = Math.exp(z[i] - m); s += p[i]; }
    for (i = 0; i < z.length; i++) p[i] /= s;
    return p;
  }

  function logit(p) { return Math.log(Math.max(p, 1e-12)) - Math.log(Math.max(1 - p, 1e-12)); }

  function predict(model, text) { return softmax(forward(model, vectorize(model, text))); }

  // leave-one-word-out: drop in the log-odds of the predicted class when a whitespace-separated word is removed
  function occlusion(model, text, cls) {
    var words = text.split(/\s+/).filter(Boolean), base = logit(predict(model, words.join(" "))[cls]), out = [], i;
    for (i = 0; i < words.length; i++) {
      var rest = words.slice(0, i).concat(words.slice(i + 1)).join(" ");
      out.push(base - logit(predict(model, rest)[cls]));
    }
    return { words: words, importance: out };
  }

  var api = { load: load, tokens: tokens, vectorize: vectorize, forward: forward, softmax: softmax, predict: predict, occlusion: occlusion };
  if (typeof module !== "undefined" && module.exports) module.exports = api; else root.MLP = api;
})(typeof window !== "undefined" ? window : globalThis);
