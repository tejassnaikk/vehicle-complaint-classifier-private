"""Phase 6: evaluate all models from SAVED predictions and checkpoints. Nothing is retrained.

Outputs (all under results/):
  eval/<model>/<split>/            metrics.json, per_class.csv, confusion_matrix.csv, confusion_counts.png,
                                   confusion_normalized.png, error_examples.csv, correct_examples.csv
  eval/errors_all_models.csv       temporal test complaints that all three models got wrong (15 sampled)
  eval/per_class_f1_all.csv        per-class test F1 for every model x split
  eval/word_importance/            occlusion highlights (HTML snippets + JSON) for BiLSTM and DistilBERT
  significance.json                bootstrap 95% CIs (macro-F1), paired bootstrap differences, McNemar tests
  transcript_vs_owner.json / .csv  macro-F1 by group on the temporal test set
  summary_table.csv / .md / .json  model x split x accuracy x macro-F1 (+ CI, training time)
Seed 42 everywhere. Test labels are used only to score saved predictions.
"""
import html
import importlib.util
import json
import textwrap
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy.stats import binomtest

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CLASSES, ID2LABEL, LABEL2ID, MODELS_DIR, RESULTS_DIR, SEED, compute_metrics, load_splits, per_class_report
from config import PLAIN_NAMES

MODELS = ["mlp", "bilstm", "distilbert"]
NICE = {"mlp": "MLP (TF-IDF)", "bilstm": "BiLSTM", "distilbert": "DistilBERT"}
SPLITS = ["temporal", "random"]
K = len(CLASSES)
B = 1000
EVAL = RESULTS_DIR / "eval"


# ----------------------------------------------------------------------------- loading
def load_all():
    """preds[split][model] = DataFrame aligned to the test split rows, with text and metadata."""
    out = {}
    for sc in SPLITS:
        test = load_splits(sc)["test"][["odino", "make", "model", "year", "text", "label", "is_transcript"]]
        out[sc] = {}
        for m in MODELS:
            p = pd.read_csv(RESULTS_DIR / m / sc / "test_predictions.csv", dtype={"odino": str}).drop(columns=["is_transcript"])
            d = test.merge(p, on="odino", how="left", validate="one_to_one")
            assert d.y_pred.notna().all() and (d.label == d.true_label).all(), "prediction/test mismatch"
            d["y_pred"] = d.y_pred.astype(int)
            d["y_true"] = d.y_true.astype(int)
            out[sc][m] = d
        for m in MODELS[1:]:
            assert (out[sc][m].odino == out[sc][MODELS[0]].odino).all()
    return out


# ----------------------------------------------------------------------------- metrics & plots
def macro_f1_fast(yt, yp):
    cm = np.bincount(yt * K + yp, minlength=K * K).reshape(K, K)
    tp = np.diag(cm).astype(float)
    denom = cm.sum(0) + cm.sum(1)
    return float(np.mean(np.divide(2 * tp, denom, out=np.zeros(K), where=denom > 0)))


def macro_f1_min_support(yt, yp, min_support=30):
    """Macro-F1 over only the classes that have at least min_support true texts in this group."""
    cm = np.bincount(yt * K + yp, minlength=K * K).reshape(K, K)
    tp = np.diag(cm).astype(float)
    denom = cm.sum(0) + cm.sum(1)
    f1 = np.divide(2 * tp, denom, out=np.zeros(K), where=denom > 0)
    keep = cm.sum(1) >= min_support
    return float(f1[keep].mean()), int(keep.sum())


def bootstrap_idx(n, rng):
    return rng.integers(0, n, size=(B, n))


def boot_f1(yt, yp, idx):
    return np.array([macro_f1_fast(yt[i], yp[i]) for i in idx])


def top_pairs(d, k=5):
    w = d[d.y_true != d.y_pred]
    c = w.groupby(["true_label", "pred_label"]).size().sort_values(ascending=False)
    directed = [{"true": a, "pred": b, "count": int(n)} for (a, b), n in c.head(k).items()]
    u = {}
    for (a, b), n in c.items():
        key = " / ".join(sorted([a, b]))
        u[key] = u.get(key, 0) + int(n)
    unordered = [{"pair": p, "count": n} for p, n in sorted(u.items(), key=lambda x: -x[1])[:k]]
    return directed, unordered


def wrap(s):
    return "\n".join(textwrap.wrap(s, 12, break_long_words=False))


def plot_cm(cm, path, title, normalized):
    """Correct cells (diagonal) in an amber ramp, mix-ups (off-diagonal) in a red ramp: red marks errors only."""
    from matplotlib.colors import LinearSegmentedColormap
    import style as S
    labels = [wrap(PLAIN_NAMES[c]) for c in CLASSES]
    data = cm / cm.sum(1, keepdims=True) if normalized else cm.astype(float)
    diag = np.eye(K, dtype=bool)
    off_max = float(data[~diag].max())
    amber = LinearSegmentedColormap.from_list("amb", ["#fff4d6", S.AMBER])
    red = LinearSegmentedColormap.from_list("red", ["#fdf3f3", S.RED])
    fmt_ = ".2f" if normalized else ".0f"
    fig, ax = plt.subplots(figsize=(10.5, 9))
    common = dict(annot=True, fmt=fmt_, cbar=False, linewidths=1.4, linecolor="white", ax=ax, xticklabels=labels, yticklabels=labels,
                  annot_kws={"size": 11.5})
    sns.heatmap(data, mask=~diag, cmap=amber, vmin=(0.4 if normalized else 0), vmax=(1.0 if normalized else float(data[diag].max())), **common)
    sns.heatmap(data, mask=diag, cmap=red, vmin=0, vmax=off_max, **common)
    ax.set_xlabel("Predicted class", fontsize=13); ax.set_ylabel("True class", fontsize=13)
    S.title(ax, title + (" (share of each true class)" if normalized else " (number of complaints)"), size=19)
    plt.setp(ax.get_xticklabels(), rotation=0, fontsize=9.5); plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9.5)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)


def eval_one(d, m, sc):
    out = EVAL / m / sc
    out.mkdir(parents=True, exist_ok=True)
    yt, yp = d.y_true.values, d.y_pred.values
    met = compute_metrics(yt, yp)
    rep = per_class_report(yt, yp)
    cm = np.bincount(yt * K + yp, minlength=K * K).reshape(K, K)
    directed, unordered = top_pairs(d)
    (out / "metrics.json").write_text(json.dumps({
        "model": m, "split_scheme": sc, **met,
        "per_class": rep.to_dict(orient="records"),
        "top_confused_directed": directed, "top_confused_pairs": unordered}, indent=2))
    rep.to_csv(out / "per_class.csv", index=False)
    pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(out / "confusion_matrix.csv")
    title = f"{NICE[m]}, {sc} split, test set (n={len(d):,})"
    plot_cm(cm, out / "confusion_counts.png", title, False)
    plot_cm(cm, out / "confusion_normalized.png", title, True)

    cols = ["odino", "vehicle_make", "vehicle_model", "vehicle_year", "is_transcript", "true_label", "pred_label",
            "confidence", "text_first_300_chars"]
    dd = d.rename(columns={"make": "vehicle_make", "model": "vehicle_model", "year": "vehicle_year"})
    dd["text_first_300_chars"] = dd.text.str.slice(0, 300)
    wrong = dd[dd.y_true != dd.y_pred].sample(15, random_state=SEED)
    right = dd[(dd.y_true == dd.y_pred) & (dd.confidence >= 0.95)]
    right = right.sample(5, random_state=SEED) if len(right) >= 5 else dd[dd.y_true == dd.y_pred].nlargest(5, "confidence")
    wrong[cols].to_csv(out / "error_examples.csv", index=False)
    right[cols].to_csv(out / "correct_examples.csv", index=False)
    return met, rep


# ----------------------------------------------------------------------------- significance
def significance(preds):
    rng = np.random.default_rng(SEED)
    res = {"n_bootstrap": B, "seed": SEED, "macro_f1_95ci": {}, "paired_difference_macro_f1_95ci": {}, "mcnemar_exact_temporal": []}
    for sc in SPLITS:
        d0 = preds[sc][MODELS[0]]
        idx = bootstrap_idx(len(d0), rng)  # same resamples for every model: paired
        yt = d0.y_true.values
        boots = {m: boot_f1(yt, preds[sc][m].y_pred.values, idx) for m in MODELS}
        for m in MODELS:
            point = macro_f1_fast(yt, preds[sc][m].y_pred.values)
            lo, hi = np.percentile(boots[m], [2.5, 97.5])
            res["macro_f1_95ci"].setdefault(m, {})[sc] = {"macro_f1": point, "ci_low": float(lo), "ci_high": float(hi)}
        for a, b in combinations(MODELS, 2):
            diff = boots[b] - boots[a]
            lo, hi = np.percentile(diff, [2.5, 97.5])
            pt = macro_f1_fast(yt, preds[sc][b].y_pred.values) - macro_f1_fast(yt, preds[sc][a].y_pred.values)
            res["paired_difference_macro_f1_95ci"].setdefault(sc, {})[f"{b} minus {a}"] = {
                "difference": float(pt), "ci_low": float(lo), "ci_high": float(hi), "ci_excludes_zero": bool(lo > 0 or hi < 0)}
    # McNemar (exact, two-sided) on temporal test predictions
    t = preds["temporal"]
    for a, b in [("mlp", "bilstm"), ("mlp", "distilbert"), ("bilstm", "distilbert")]:
        ca, cb = t[a].y_true == t[a].y_pred, t[b].y_true == t[b].y_pred
        both, neither = int((ca & cb).sum()), int((~ca & ~cb).sum())
        only_a, only_b = int((ca & ~cb).sum()), int((~ca & cb).sum())
        n = only_a + only_b
        p = float(binomtest(min(only_a, only_b), n, 0.5).pvalue) if n else 1.0
        chi2 = (abs(only_a - only_b) - 1) ** 2 / n if n else 0.0
        res["mcnemar_exact_temporal"].append({
            "model_a": a, "model_b": b, "both_correct": both, "both_wrong": neither,
            f"only_{a}_correct": only_a, f"only_{b}_correct": only_b,
            "accuracy_a": float(ca.mean()), "accuracy_b": float(cb.mean()),
            "chi2_continuity_corrected": float(chi2), "p_value_exact": p,
            "significant_at_0.05": bool(p < 0.05), "significant_bonferroni_0.0167": bool(p < 0.05 / 3)})
    res["note"] = ("McNemar compares paired per-text correctness on the same temporal test set (exact binomial, two-sided). "
                   "Bootstrap resamples test texts with replacement (seed 42); the same resamples are used for all models within a split.")
    (RESULTS_DIR / "significance.json").write_text(json.dumps(res, indent=2))
    return res


def transcript_analysis(preds):
    rng = np.random.default_rng(SEED)
    rows = []
    t = preds["temporal"]
    for grp, flag in [("owner-written", False), ("transcript", True)]:
        mask = (t[MODELS[0]].is_transcript == flag).values
        idx_g = np.where(mask)[0]
        boot_idx = idx_g[rng.integers(0, len(idx_g), size=(B, len(idx_g)))]  # resample within group, paired across models
        support = np.bincount(t[MODELS[0]].y_true.values[idx_g], minlength=K)
        for m in MODELS:
            yt, yp = t[m].y_true.values, t[m].y_pred.values
            bs = boot_f1(yt, yp, boot_idx)
            lo, hi = np.percentile(bs, [2.5, 97.5])
            rows.append({"model": m, "group": grp, "n": int(mask.sum()), "accuracy": float((yt[idx_g] == yp[idx_g]).mean()),
                         "macro_f1": macro_f1_fast(yt[idx_g], yp[idx_g]), "macro_f1_ci_low": float(lo), "macro_f1_ci_high": float(hi),
                         "macro_f1_classes_with_30plus_texts": macro_f1_min_support(yt[idx_g], yp[idx_g])[0],
                         "n_classes_with_30plus_texts": macro_f1_min_support(yt[idx_g], yp[idx_g])[1],
                         "min_class_support": int(support.min()),
                         "smallest_class": ID2LABEL[int(support.argmin())]})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "transcript_vs_owner.csv", index=False)
    sup = {g: {ID2LABEL[i]: int(c) for i, c in enumerate(np.bincount(t[MODELS[0]].y_true.values[(t[MODELS[0]].is_transcript == f).values], minlength=K))}
           for g, f in [("owner-written", False), ("transcript", True)]}
    (RESULTS_DIR / "transcript_vs_owner.json").write_text(json.dumps(
        {"split": "temporal test", "n_bootstrap": B, "rows": rows, "class_support_by_group": sup,
         "note": "Macro-F1 over all 10 classes within each group; groups have very different class mixes, so compare with the CIs."}, indent=2))
    return df, sup


# ----------------------------------------------------------------------------- summary
def summary(preds, sig):
    rows = []
    for m in MODELS:
        for sc in SPLITS:
            j = json.loads((RESULTS_DIR / m / sc / "metrics.json").read_text())
            ci = sig["macro_f1_95ci"][m][sc]
            rows.append({"model": NICE[m], "split": sc, "val_accuracy": j["val"]["accuracy"], "val_macro_f1": j["val"]["macro_f1"],
                         "test_accuracy": j["test"]["accuracy"], "test_macro_f1": j["test"]["macro_f1"],
                         "test_macro_f1_ci_low": ci["ci_low"], "test_macro_f1_ci_high": ci["ci_high"],
                         "test_weighted_f1": j["test"]["weighted_f1"], "n_test": j["test"]["n"],
                         "best_epoch": j["best_epoch"], "epochs_run": j["epochs_run"], "train_time_s": j["wall_time_s"]})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "summary_table.csv", index=False)
    (RESULTS_DIR / "summary_table.json").write_text(json.dumps(rows, indent=2))
    lines = ["| Model | Split | Val acc | Val macro-F1 | Test acc | Test macro-F1 (95% CI) | Train time |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        tt = r["train_time_s"]
        time_s = f"{tt:.0f} s" if tt < 120 else f"{tt / 60:.0f} min"
        lines.append(f"| {r['model']} | {r['split']} | {r['val_accuracy']:.4f} | {r['val_macro_f1']:.4f} | {r['test_accuracy']:.4f} | "
                     f"{r['test_macro_f1']:.4f} ({r['test_macro_f1_ci_low']:.4f} to {r['test_macro_f1_ci_high']:.4f}) | {time_s} |")
    (RESULTS_DIR / "summary_table.md").write_text("\n".join(lines) + "\n")
    return df


# ----------------------------------------------------------------------------- word importance (occlusion)
def load_bilstm():
    spec = importlib.util.spec_from_file_location("bilstm_mod", Path(__file__).resolve().parent / "06_train_bilstm.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    vocab = json.loads((MODELS_DIR / "bilstm_temporal_vocab.json").read_text())
    model = mod.BiLSTM(len(vocab) + 2, K)
    model.load_state_dict(torch.load(MODELS_DIR / "bilstm_temporal.pt", map_location="cpu"))
    model.eval()

    @torch.no_grad()
    def logits(texts):
        enc = [[vocab.get(w, mod.UNK) for w in mod.tokenize(t)[:mod.CFG["max_len"]]] or [mod.PAD] for t in texts]
        out = []
        for i in range(0, len(enc), 64):
            ch = enc[i:i + 64]
            L = max(len(s) for s in ch)
            x = torch.zeros(len(ch), L, dtype=torch.long)
            for j, s in enumerate(ch):
                x[j, :len(s)] = torch.tensor(s)
            out.append(model(x, torch.tensor([len(s) for s in ch])))
        return torch.cat(out).numpy()
    return logits


def load_distilbert():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=K)
    model.load_state_dict(torch.load(MODELS_DIR / "distilbert_temporal.pt", map_location="cpu"))
    model.eval()

    @torch.no_grad()
    def logits(texts):
        out = []
        for i in range(0, len(texts), 32):
            enc = tok(texts[i:i + 32], truncation=True, max_length=256, padding=True, return_tensors="pt")
            out.append(model(**enc).logits)
        return torch.cat(out).numpy()
    return logits


def pick_examples(t):
    """6 temporal-test complaints, 25-90 words: 2 both-right, 2 both-wrong, 2 DistilBERT-right/BiLSTM-wrong. Seed 42."""
    b, d = t["bilstm"], t["distilbert"]
    nw = d.text.str.split().str.len()
    ok = (nw >= 25) & (nw <= 90)
    cats = {"both_correct": (b.y_true == b.y_pred) & (d.y_true == d.y_pred),
            "both_wrong": (b.y_true != b.y_pred) & (d.y_true != d.y_pred),
            "distilbert_right_bilstm_wrong": (b.y_true != b.y_pred) & (d.y_true == d.y_pred)}
    rng = np.random.default_rng(SEED)
    chosen, used = [], set()
    for cat, mask in cats.items():
        cand = np.where(ok & mask)[0]
        rng.shuffle(cand)
        n_cat = 0
        for i in cand:
            lab = d.true_label[i]
            if lab in used and len(used) < K:
                continue
            chosen.append((cat, int(i))); used.add(lab); n_cat += 1
            if n_cat == 2:
                break
    return chosen


def render_html(words, imp, header, legend=True):
    mx = max(float(np.abs(imp).max()), 1e-9)
    spans = []
    for w, v in zip(words, imp):
        r = abs(v) / mx
        w_ = html.escape(w)
        if r < 0.08:
            spans.append(f"<span title=\"{v:+.3f}\">{w_}</span>")
        else:
            rgb = "242,169,0" if v > 0 else "92,102,116"
            spans.append(f"<span title=\"{v:+.3f}\" style=\"background:rgba({rgb},{0.15 + 0.7 * r:.2f});color:#111;border-radius:3px;padding:0 1px\">{w_}</span>")
    leg = ("<p class=\"wi-legend\"><span style=\"background:rgba(242,169,0,.6);color:#111;padding:0 4px;border-radius:3px\">supports the predicted class</span> "
           "<span style=\"background:rgba(92,102,116,.5);color:#111;padding:0 4px;border-radius:3px\">argues against it</span> "
           "(importance = drop in the predicted class's log-odds when the word is removed)</p>") if legend else ""
    return f"<div class=\"wi\"><p class=\"wi-head\">{header}</p><p class=\"wi-text\">{' '.join(spans)}</p>{leg}</div>\n"


def word_importance(preds):
    t = preds["temporal"]
    chosen = pick_examples(t)
    wdir = EVAL / "word_importance"
    wdir.mkdir(parents=True, exist_ok=True)
    fns = {"bilstm": load_bilstm(), "distilbert": load_distilbert()}
    summary_rows, page = [], ["<!doctype html><meta charset=utf-8><title>Word importance preview</title>"
                              "<body style=\"font-family:system-ui;max-width:860px;margin:2rem auto;line-height:1.6\">"]
    for n, (cat, i) in enumerate(chosen, 1):
        row = t["distilbert"].iloc[i]
        words = row.text.split()
        page.append(f"<h3>Example {n} ({cat.replace('_', ' ')}): true class {row.true_label}</h3>")
        for m, fn in fns.items():
            variants = [row.text] + [" ".join(words[:j] + words[j + 1:]) for j in range(len(words))]
            Z = fn(variants)                       # logits for the original text and each one-word-removed variant
            base = torch.softmax(torch.from_numpy(Z[0]), 0).numpy()
            pred = int(base.argmax())
            saved = t[m].iloc[i]
            assert pred == int(saved.y_pred) and abs(float(base[pred]) - float(saved.confidence)) < 5e-3, \
                f"checkpoint mismatch for {m} example {n}: {pred} vs {saved.y_pred}, {base[pred]:.4f} vs {saved.confidence:.4f}"
            others = np.delete(Z, pred, axis=1)
            logodds = Z[:, pred] - (np.log(np.exp(others - others.max(1, keepdims=True)).sum(1)) + others.max(1))
            imp = logodds[0] - logodds[1:]          # drop in the predicted class's log-odds when the word is removed
            hdr = (f"<b>{NICE[m]}</b>: predicted <b>{ID2LABEL[pred]}</b> ({base[pred]:.0%} confidence); "
                   f"true class <b>{row.true_label}</b>{' (correct)' if pred == int(row.y_true) else ' (wrong)'}")
            frag = render_html(words, imp, hdr)
            (wdir / f"example{n}_{m}.html").write_text(frag)
            page.append(frag)
            top = np.argsort(-imp)[:5]
            summary_rows.append({"example": n, "category": cat, "model": m, "odino": row.odino, "true_label": row.true_label,
                                 "pred_label": ID2LABEL[pred], "confidence": float(base[pred]), "n_words": len(words),
                                 "top_supporting_words": [{"word": words[j], "importance": float(imp[j])} for j in top],
                                 "most_against_word": {"word": words[int(imp.argmin())], "importance": float(imp.min())},
                                 "importances": [float(x) for x in imp], "words": words})
        print(f"  word importance example {n}/6 done ({cat}, {row.true_label})", flush=True)
    (wdir / "preview.html").write_text("\n".join(page))
    (wdir / "word_importance.json").write_text(json.dumps(
        {"method": "occlusion: remove one whitespace-separated word; importance = log-odds of the predicted class before minus after (log-odds, not probability, so confident predictions do not saturate at 0)",
         "device": "cpu", "selection": "temporal test, 25-90 words; 2 both-models-right, 2 both-wrong, 2 DistilBERT-right/BiLSTM-wrong; seed 42",
         "examples": summary_rows}, indent=2))


# ----------------------------------------------------------------------------- main
def main():
    EVAL.mkdir(parents=True, exist_ok=True)
    preds = load_all()
    allf1 = {}
    for sc in SPLITS:
        for m in MODELS:
            met, rep = eval_one(preds[sc][m], m, sc)
            allf1[f"{m}_{sc}"] = rep.set_index("class").f1
            saved = json.loads((RESULTS_DIR / m / sc / "metrics.json").read_text())["test"]
            assert abs(met["accuracy"] - saved["accuracy"]) < 1e-9 and abs(met["macro_f1"] - saved["macro_f1"]) < 1e-9, "metrics differ from training run"
            print(f"{m:10s} {sc:9s} acc {met['accuracy']:.4f} macroF1 {met['macro_f1']:.4f} (matches training-run metrics)", flush=True)
    pd.DataFrame(allf1).round(4).to_csv(EVAL / "per_class_f1_all.csv")

    t = preds["temporal"]
    allwrong = (t["mlp"].y_true != t["mlp"].y_pred) & (t["bilstm"].y_true != t["bilstm"].y_pred) & (t["distilbert"].y_true != t["distilbert"].y_pred)
    aw = t["distilbert"][allwrong]
    print(f"temporal test: {int(allwrong.sum())} complaints misclassified by all three models")
    cols = ["odino", "make", "model", "year", "is_transcript", "true_label"]
    ex = aw.sample(min(15, len(aw)), random_state=SEED)[cols].copy()
    for m in MODELS:
        ex[f"{m}_pred"] = t[m].loc[ex.index, "pred_label"]
    ex["text_first_300_chars"] = aw.loc[ex.index, "text"].str.slice(0, 300)
    ex.to_csv(EVAL / "errors_all_models.csv", index=False)

    sig = significance(preds)
    print("\nBootstrap 95% CIs on test macro-F1:")
    for m in MODELS:
        for sc in SPLITS:
            c = sig["macro_f1_95ci"][m][sc]
            print(f"  {m:10s} {sc:9s} {c['macro_f1']:.4f} [{c['ci_low']:.4f}, {c['ci_high']:.4f}]")
    print("Paired bootstrap differences in macro-F1:")
    for sc, d in sig["paired_difference_macro_f1_95ci"].items():
        for k, v in d.items():
            print(f"  {sc:9s} {k:25s} {v['difference']:+.4f} [{v['ci_low']:+.4f}, {v['ci_high']:+.4f}] {'excludes 0' if v['ci_excludes_zero'] else 'includes 0'}")
    print("McNemar (exact), temporal test:")
    for r in sig["mcnemar_exact_temporal"]:
        a, b = r["model_a"], r["model_b"]
        print(f"  {a} vs {b}: only {a} right {r['only_'+a+'_correct']}, only {b} right {r['only_'+b+'_correct']}, p = {r['p_value_exact']:.3g}")
    tdf, sup = transcript_analysis(preds)
    print("\nTranscript vs owner-written (temporal test):")
    print(tdf.round(4).to_string(index=False))
    summary(preds, sig)
    print("\nSummary table written.")
    print((RESULTS_DIR / "summary_table.md").read_text())
    word_importance(preds)


if __name__ == "__main__":
    main()
