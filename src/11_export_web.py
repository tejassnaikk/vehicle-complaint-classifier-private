"""Export data for the interactive "Explore" pages into docs/data/. Everything comes from files written by earlier scripts.

  python src/11_export_web.py            all exports
  python src/11_export_web.py try        only the browser MLP (+ the 1,000-text agreement check) and the complaint sample

Complaint text shown on the site comes only from the masked CSVs in data/clean/.
Outputs (docs/data/): mlp_vocab.json, mlp_weights.bin, sample_complaints.json, mixups.json, trends.json
Check report: results/browser_model_check.json
"""
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CLASSES, CLEAN_DIR, MODELS_DIR, PLAIN_NAMES, RESULTS_DIR as R, ROOT, SEED

OUT = ROOT / "docs" / "data"
OUT.mkdir(parents=True, exist_ok=True)
MODELS = ["mlp", "bilstm", "distilbert"]
N_CHECK = 1000


def temporal_test():
    d = pd.read_csv(CLEAN_DIR / "complaints_temporal.csv", dtype=str, keep_default_na=False)
    return d[d.split == "test"].reset_index(drop=True)


# ---------------------------------------------------------------------------------------------- 1. browser MLP
def quantize_rows(w_in_out):
    """int8 with one float32 scale per input row (row = one input feature). w_in_out has shape (in, out)."""
    scale = np.abs(w_in_out).max(axis=1) / 127.0
    scale = np.where(scale == 0, 1.0, scale).astype(np.float32)
    q = np.clip(np.rint(w_in_out / scale[:, None]), -127, 127).astype(np.int8)
    return q, scale


def export_mlp():
    with open(MODELS_DIR / "mlp_temporal_tfidf.pkl", "rb") as f:
        vec = pickle.load(f)
    sd = torch.load(MODELS_DIR / "mlp_temporal.pt", map_location="cpu")
    assert vec.lowercase and vec.ngram_range == (1, 2) and vec.norm == "l2" and vec.use_idf and vec.smooth_idf and not vec.sublinear_tf
    assert vec.token_pattern == r"(?u)\b\w\w+\b" and vec.strip_accents is None and vec.stop_words is None and vec.binary is False
    n_feat = len(vec.vocabulary_)
    terms = [None] * n_feat
    for t, i in vec.vocabulary_.items():
        terms[i] = t
    idf = vec.idf_.astype(np.float32)
    layers, blob, off = [], bytearray(), 0
    for k in ["net.0", "net.3", "net.6"]:
        w = sd[f"{k}.weight"].numpy().T.copy()   # (in, out)
        b = sd[f"{k}.bias"].numpy().astype(np.float32)
        q, s = quantize_rows(w)
        entry = {"in": int(w.shape[0]), "out": int(w.shape[1])}
        for name, arr in [("w", q), ("scale", s), ("bias", b)]:
            raw = arr.tobytes()
            entry[f"{name}_offset"] = off
            blob += raw
            pad = (-len(raw)) % 4
            blob += b"\0" * pad
            off += len(raw) + pad
        layers.append(entry)
    (OUT / "mlp_weights.bin").write_bytes(bytes(blob))
    meta = {"classes": CLASSES, "plain_names": [PLAIN_NAMES[c] for c in CLASSES], "token_pattern": r"(?u)\b\w\w+\b", "ngram_range": [1, 2],
            "n_features": n_feat, "layers": layers, "weights_bytes": len(blob),
            "terms": terms, "idf": [float(f"{x:.7g}") for x in idf]}
    (OUT / "mlp_vocab.json").write_text(json.dumps(meta, separators=(",", ":")))
    return vec, sd


def reference_probs(vec, sd, texts):
    X = torch.from_numpy(vec.transform(texts).toarray().astype(np.float32))
    h = torch.relu(X @ sd["net.0.weight"].T + sd["net.0.bias"])
    h = torch.relu(h @ sd["net.3.weight"].T + sd["net.3.bias"])
    return torch.softmax(h @ sd["net.6.weight"].T + sd["net.6.bias"], 1).numpy()


def check_browser_model(vec, sd):
    t = temporal_test().sample(N_CHECK, random_state=SEED).reset_index(drop=True)
    probs = reference_probs(vec, sd, t.text.tolist())
    ref = ROOT / "results" / "browser_model_check_reference.json"
    ref.write_text(json.dumps({"odino": t.odino.tolist(), "texts": t.text.tolist(), "probs": np.round(probs, 7).tolist()}))
    subprocess.run(["node", str(ROOT / "src" / "verify_browser_model.js")], check=True)
    rep = json.loads((R / "browser_model_check.json").read_text())
    ref.unlink()  # the 1,000 texts are already in data/clean; the report keeps only the numbers
    assert rep["agreement_pct"] >= 99.5, rep
    return rep


def export_sample():
    """200 test complaints, 20 per class, 25 to 120 words, with the predictions of all three models (temporal split)."""
    t = temporal_test()
    nw = t.text.str.split().str.len()
    pred = {}
    for m in MODELS:
        p = pd.read_csv(R / m / "temporal" / "test_predictions.csv", dtype={"odino": str}).set_index("odino")
        assert (p.index == t.odino).all() or set(p.index) == set(t.odino)
        pred[m] = p
    rows = []
    for c in CLASSES:
        pool = t[(t.label == c) & (nw >= 25) & (nw <= 120)]
        assert len(pool) >= 20, c
        rows.append(pool.sample(20, random_state=SEED))
    s = pd.concat(rows).sample(frac=1, random_state=SEED).reset_index(drop=True)
    items = []
    for r in s.itertuples():
        it = {"id": r.odino, "text": r.text, "label": CLASSES.index(r.label), "year": r.year, "make": r.make, "received": r.ldate}
        for m in MODELS:
            q = pred[m].loc[r.odino]
            it[m] = {"pred": CLASSES.index(q.pred_label), "conf": round(float(q.confidence), 4)}
        items.append(it)
    out = {"classes": CLASSES, "plain_names": [PLAIN_NAMES[c] for c in CLASSES], "source": "data/clean/complaints_temporal.csv, split = test",
           "selection": "20 per class, 25 to 120 words, random with seed 42", "items": items}
    (OUT / "sample_complaints.json").write_text(json.dumps(out, separators=(",", ":")))
    return out


# ---------------------------------------------------------------------------------------------- 3. mix-up explorer
def export_mixups():
    """DistilBERT temporal test confusion matrix and up to 5 real complaints per (true, predicted) cell, with the model's confidence."""
    t = temporal_test()
    p = pd.read_csv(R / "distilbert" / "temporal" / "test_predictions.csv", dtype={"odino": str})
    d = p.merge(t[["odino", "text"]], on="odino", how="left")
    assert len(d) == len(t) and d.text.notna().all()
    k = len(CLASSES)
    mat = np.zeros((k, k), dtype=int)
    for yt, yp in zip(d.true_label, d.pred_label):
        mat[CLASSES.index(yt), CLASSES.index(yp)] += 1
    m = json.loads((R / "eval" / "distilbert" / "temporal" / "metrics.json").read_text())
    assert abs(np.trace(mat) / mat.sum() - m["accuracy"]) < 1e-9, "confusion matrix does not match results/eval/distilbert/temporal/metrics.json"
    ex = {}
    for (yt, yp), g in d.groupby(["true_label", "pred_label"]):
        g = g.sample(min(5, len(g)), random_state=SEED)
        items = []
        for r in g.itertuples():
            txt, cut = r.text, False
            if len(txt) > 600:
                txt, cut = txt[:600].rsplit(" ", 1)[0], True
            items.append({"id": r.odino, "conf": round(float(r.confidence), 4), "text": txt, "cut": cut})
        ex[f"{CLASSES.index(yt)},{CLASSES.index(yp)}"] = items
    out = {"model": "DistilBERT", "split": "temporal test set", "n": int(mat.sum()), "classes": CLASSES, "plain_names": [PLAIN_NAMES[c] for c in CLASSES],
           "matrix": mat.tolist(), "examples": ex, "source": "results/distilbert/temporal/test_predictions.csv and data/clean/complaints_temporal.csv"}
    (OUT / "mixups.json").write_text(json.dumps(out, separators=(",", ":")))
    return out


def write_file_sizes():
    """results/explore_files.json: size of every file the Explore pages download (the site cites these numbers)."""
    files = {f.name: f.stat().st_size for f in sorted(OUT.glob("*")) if f.is_file()}
    rep = {"files_bytes": files, "files_mb": {k: round(v / 1e6, 1) for k, v in files.items()},
           "mlp_page_download_mb": round(sum(v for k, v in files.items() if k in ("mlp_vocab.json", "mlp_weights.bin")) / 1e6, 1)}
    (R / "explore_files.json").write_text(json.dumps(rep, indent=2))


def main(which):
    if which in ("all", "try"):
        vec, sd = export_mlp()
        rep = check_browser_model(vec, sd)
        print("browser model check:", rep)
        s = export_sample()
        print("sample complaints:", len(s["items"]))
    if which in ("all", "mixups"):
        mx = export_mixups()
        print("mixups: n =", mx["n"], "cells with examples:", len(mx["examples"]))
    write_file_sizes()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
