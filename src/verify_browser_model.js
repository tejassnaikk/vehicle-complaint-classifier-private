// Runs docs/js/mlp.js in Node on the same 1,000 temporal test texts that Python scored with the original float model.
// Called by src/11_export_web.py. Writes results/browser_model_check.json.
const fs = require("fs"), path = require("path");
const root = path.resolve(__dirname, "..");
const MLP = require(path.join(root, "docs/js/mlp.js"));
const meta = JSON.parse(fs.readFileSync(path.join(root, "docs/data/mlp_vocab.json"), "utf8"));
const buf = fs.readFileSync(path.join(root, "docs/data/mlp_weights.bin"));
const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
const model = MLP.load(meta, ab);
const ref = JSON.parse(fs.readFileSync(path.join(root, "results/browser_model_check_reference.json"), "utf8"));
let agree = 0, maxDiff = 0, sumDiff = 0, n = ref.texts.length, disagreements = [];
for (let i = 0; i < n; i++) {
  const p = MLP.predict(model, ref.texts[i]), q = ref.probs[i];
  let a = 0, b = 0, d = 0;
  for (let c = 0; c < p.length; c++) { if (p[c] > p[a]) a = c; if (q[c] > q[b]) b = c; d = Math.max(d, Math.abs(p[c] - q[c])); }
  if (a === b) agree++; else disagreements.push({ odino: ref.odino[i], python_top2_gap: Math.abs(q[b] - q[a]) });
  maxDiff = Math.max(maxDiff, d); sumDiff += d;
}
const report = {
  what: "Python (scikit-learn TF-IDF + float32 PyTorch MLP) versus the browser code (docs/js/mlp.js, int8 weights with per-row float32 scales) on the same texts",
  n_texts: n, texts: "random sample of the temporal test set, seed 42",
  agreement_pct: +(100 * agree / n).toFixed(2), n_disagree: n - agree,
  max_abs_probability_difference: +maxDiff.toExponential(3), mean_max_abs_probability_difference: +(sumDiff / n).toExponential(3),
  weights_bytes: meta.weights_bytes, node: process.version, disagreements
};
fs.writeFileSync(path.join(root, "results/browser_model_check.json"), JSON.stringify(report, null, 2));
console.log(JSON.stringify(report));
