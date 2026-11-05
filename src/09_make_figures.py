"""Phase 7 (restyled in the Phase 8 redesign): make every website figure from files in results/ and data/.

Figures are written to results/figures/ (chart data in results/figures/data/) and copied to docs/img/.
Nothing is retrained. Look: src/style.py (asphalt charcoal, amber accent, red only for errors, Barlow Condensed headings).
Also writes results/eda/dataset_facts.json and results/data_funnel.json, and redraws training curves from history.csv.
"""
import csv
import difflib
import json
import re
import shutil
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.ticker import FuncFormatter

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import style as S
from common import plot_curves
from config import CLASSES, CLEAN_DIR, INTERIM_DIR, PLAIN_NAMES, RAW_DIR, RESULTS_DIR, ROOT

FIG = RESULTS_DIR / "figures"
DATA = FIG / "data"
DOCS_IMG = ROOT / "docs" / "img"
MODELS = ["mlp", "bilstm", "distilbert"]
NICE = {"mlp": "MLP (TF-IDF)", "bilstm": "BiLSTM", "distilbert": "DistilBERT"}
PLAIN = PLAIN_NAMES
SOURCE_NOTE = "Source: NHTSA vehicle owner complaints (US government data, public domain)"
MADE = []


def save(fig, name, svg=False, dpi=200):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=dpi)
    MADE.append(f"{name}.png")
    if svg:
        fig.savefig(FIG / f"{name}.svg")
        MADE.append(f"{name}.svg")
    plt.close(fig)


def jload(p):
    return json.loads(Path(p).read_text())


def fmt(n):
    return f"{int(round(n)):,}"


# ============================================================================ data tables behind the figures
def build_funnel():
    """results/data_funnel.json (raw counts from the raw files, later steps parsed from results/leakage_audit.txt)."""
    audit = (RESULTS_DIR / "leakage_audit.txt").read_text()

    def num(pattern):
        m = re.search(pattern, audit)
        assert m, pattern
        return int(m.group(1).replace(",", ""))

    raw = pd.concat([pd.read_csv(RAW_DIR / f, sep="\t", header=None, usecols=[1, 11, 16], names=["odino", "compdesc", "ldate"], quoting=csv.QUOTE_NONE,
                                 dtype=str, keep_default_na=False)
                     for f in ["COMPLAINTS_RECEIVED_2020-2024.txt", "COMPLAINTS_RECEIVED_2025-2026.txt"]])
    f = {
        "raw_rows": len(raw), "raw_complaints": int(raw.odino.nunique()),
        "single_label": num(r"single merged top_level: ([\d,]+)"), "in_class_list": num(r"Single-label and in class list: ([\d,]+)"),
        "dropped_no_failure": num(r"rule 11 drop[^:]*: ([\d,]+) texts"),
        "dropped_duplicates": num(r"Exact duplicate texts dropped \(after stripping\): ([\d,]+)"),
        "dropped_short": num(r"Texts under 20 words dropped \(after stripping\): ([\d,]+)"),
        "cleaned_complaints": num(r"Rows: [\d,]+ -> ([\d,]+)")}
    assert f["raw_rows"] == num(r"Raw rows: ([\d,]+)") and f["raw_complaints"] == num(r"ODINOs total: ([\d,]+)")
    f["complaints_per_year_file"] = "results/eda/complaints_per_year.csv"
    (RESULTS_DIR / "data_funnel.json").write_text(json.dumps(f, indent=2))
    r = raw.drop_duplicates("odino")
    r = r[r.ldate <= "20261031"]
    per_year = r.ldate.str[:4].value_counts().sort_index().rename_axis("year").rename("complaints").reset_index()
    per_year.to_csv(RESULTS_DIR / "eda" / "complaints_per_year.csv", index=False)
    rows_per = raw.groupby("odino").size()
    pool = pd.read_parquet(INTERIM_DIR / "complaints_clean.parquet")
    pool = pool[pool.ldate <= "20261031"]
    words = pool.text.str.split().str.len()
    d0, d1 = pd.Timestamp("2020-01-01"), pd.Timestamp("2026-10-31")
    facts = {"complaints_through_oct_2026": int(per_year.complaints.sum()), "days_covered": int((d1 - d0).days + 1),
             "complaints_per_day_average": float(per_year.complaints.sum() / ((d1 - d0).days + 1)),
             "complaints_with_multiple_component_rows": int((rows_per > 1).sum()),
             "pct_complaints_with_multiple_component_rows": float((rows_per > 1).mean() * 100),
             "multi_label_dropped_after_merge": f["raw_complaints"] - f["single_label"],
             "pct_multi_label_dropped_after_merge": float((f["raw_complaints"] - f["single_label"]) / f["raw_complaints"] * 100),
             "cleaned_pool_complaints": int(len(pool)), "median_words": float(words.median()), "p95_words": float(words.quantile(0.95)),
             "pct_transcripts_in_cleaned_pool": float(pool.is_transcript.mean() * 100),
             "first_complaint_date": str(raw.ldate.min()), "last_complaint_date": str(raw.ldate.max())}
    (RESULTS_DIR / "eda" / "dataset_facts.json").write_text(json.dumps(facts, indent=2))
    return f, per_year


def build_class_balance():
    pool = pd.read_parquet(INTERIM_DIR / "complaints_clean.parquet")
    pool = pool[pool.ldate <= "20261031"]
    t = pd.read_csv(CLEAN_DIR / "complaints_temporal.csv", dtype=str, keep_default_na=False)
    r = pd.read_csv(CLEAN_DIR / "complaints_random.csv", dtype=str, keep_default_na=False)
    df = pd.DataFrame({"class": CLASSES})
    df["cleaned_pool"] = df["class"].map(pool.label.value_counts())
    for name, d in [("temporal", t), ("random", r)]:
        ct = pd.crosstab(d.label, d.split)
        for s in ["train", "val", "test"]:
            df[f"{name}_{s}"] = df["class"].map(ct[s])
    df.to_csv(RESULTS_DIR / "class_balance.csv", index=False)
    return df, {"pool": len(pool), "temporal": len(t), "random": len(r)}


# ============================================================================ block diagrams
KIND = {"io": ("#efece4", S.STEEL), "layer": (S.CARD, S.ASPHALT), "out": (S.AMBER, S.ASPHALT)}


def vbox(ax, x, y, w, h, title, lines, kind):
    fill, edge = KIND[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.1", fc=fill, ec=edge, lw=1.7))
    ax.text(x + w / 2, y + h - 0.13, title, ha="center", va="top", fontproperties=S.HEAD, fontsize=15, color=S.INK)
    if lines:
        ax.text(x + w / 2, y + h - 0.47, "\n".join(lines), ha="center", va="top", fontsize=9.6, color=S.INK_2 if kind != "out" else S.INK, linespacing=1.4)


def vflow(name, heading, subtitle, steps, width=4.7):
    """Vertical block diagram (top to bottom) that fits in a narrow model card."""
    gap, pad = 0.3, 0.25
    heights = [0.62 + 0.2 * len(lines) for _, lines, _ in steps]
    sub = textwrap.fill(subtitle, 50)
    head_h = 0.55 + 0.2 * (sub.count("\n") + 1) + 0.15
    total = head_h + sum(heights) + gap * (len(steps) - 1) + pad
    fig, ax = plt.subplots(figsize=(width, total))
    ax.set_xlim(0, width); ax.set_ylim(0, total); ax.axis("off")
    ax.text(0.15, total - 0.08, heading, ha="left", va="top", fontproperties=S.HEAD, fontsize=23, color=S.INK)
    ax.text(0.15, total - 0.58, sub, ha="left", va="top", fontsize=9.8, color=S.INK_2, linespacing=1.4)
    y = total - head_h
    for i, ((t, lines, kind), h) in enumerate(zip(steps, heights)):
        vbox(ax, 0.15, y - h, width - 0.3, h, t, lines, kind)
        y -= h
        if i < len(steps) - 1:
            ax.add_patch(FancyArrowPatch((width / 2, y - 0.02), (width / 2, y - gap + 0.02), arrowstyle="-|>", mutation_scale=15, lw=1.7, color=S.ASPHALT))
            y -= gap
    fig.subplots_adjust(0, 0, 1, 1)
    save(fig, name, svg=True)


def architecture_diagrams():
    cfg = {m: jload(RESULTS_DIR / m / "temporal" / "config.json") for m in MODELS}
    met = {m: jload(RESULTS_DIR / m / "temporal" / "metrics.json") for m in MODELS}
    from transformers import AutoConfig
    hf = AutoConfig.from_pretrained(cfg["distilbert"]["pretrained"])
    arch = {"pretrained": cfg["distilbert"]["pretrained"], "n_layers": hf.n_layers, "n_heads": hf.n_heads, "dim": hf.dim,
            "hidden_dim_feed_forward": hf.hidden_dim, "vocab_size": hf.vocab_size, "classifier_dropout": hf.seq_classif_dropout,
            "n_params": met["distilbert"]["n_params"]}
    (RESULTS_DIR / "distilbert" / "architecture.json").write_text(json.dumps(arch, indent=2))
    c = cfg["mlp"]
    vflow("fig_arch_mlp", "MLP on TF-IDF", "Weighted words and two-word phrases go through two dense layers.",
          [("Complaint text", ["cleaned, lowercase"], "io"),
           ("TF-IDF", [f"{fmt(met['mlp']['n_features'])} features", "words and 2-word phrases", "fit on training texts only"], "layer"),
           (f"Dense {c['hidden'][0]}", ["ReLU", f"dropout {c['dropout']}"], "layer"),
           (f"Dense {c['hidden'][1]}", ["ReLU", f"dropout {c['dropout']}"], "layer"),
           (f"Dense {len(CLASSES)}", ["one score per class"], "layer"),
           ("Predicted class", [f"one of {len(CLASSES)} components"], "out")])
    c = cfg["bilstm"]
    vflow("fig_arch_bilstm", "BiLSTM", "Reads the words in order, forward and backward, then pools what it saw.",
          [("Complaint text", ["cleaned, lowercase"], "io"),
           ("Tokenizer", [f"vocabulary {fmt(met['bilstm']['vocab_size'])} words", f"first {c['max_len']} tokens kept"], "layer"),
           (f"Embedding {c['embed_dim']}", ["one trainable vector per word"], "layer"),
           (f"{c['num_layers']}-layer BiLSTM", [f"hidden {c['hidden']} per direction", f"dropout {c['dropout']} between layers", "padding packed away"], "layer"),
           ("Pooling", ["max over time", "mean over time", f"joined: {4 * c['hidden']} numbers"], "layer"),
           (f"Linear {len(CLASSES)}", [f"dropout {c['dropout']} first", "one score per class"], "layer"),
           ("Predicted class", [f"one of {len(CLASSES)} components"], "out")])
    c = cfg["distilbert"]
    vflow("fig_arch_distilbert", "DistilBERT", f"A pre-trained language model ({fmt(arch['n_params'])} parameters), trained further on our complaints.",
          [("Complaint text", ["cleaned, lowercase"], "io"),
           ("Tokenizer", ["WordPiece", f"vocabulary {fmt(arch['vocab_size'])}", f"first {c['max_len']} tokens kept"], "layer"),
           ("Embeddings", [f"{arch['dim']} numbers per token", "plus position"], "layer"),
           (f"{arch['n_layers']} Transformer layers", [f"{arch['n_heads']} attention heads each", f"feed-forward {fmt(arch['hidden_dim_feed_forward'])}", "each word sees all others"], "layer"),
           ("First-token vector", [f"{arch['dim']} numbers", "summarizes the complaint"], "layer"),
           ("Classifier head", [f"Dense {arch['dim']} + ReLU", f"dropout {arch['classifier_dropout']}", f"Linear {len(CLASSES)}"], "layer"),
           ("Predicted class", [f"one of {len(CLASSES)} components"], "out")])


def pipeline_diagram(f, sizes, bal):
    fig, ax = plt.subplots(figsize=(15, 5.4))
    ax.set_xlim(0, 15); ax.set_ylim(0, 5.4); ax.axis("off")
    ax.text(0.2, 5.25, "From the NHTSA file to the results", fontproperties=S.HEAD, fontsize=26, va="top", color=S.INK)
    t = {s: int(bal[f"temporal_{s}"].sum()) for s in ["train", "val", "test"]}
    r = {s: int(bal[f"random_{s}"].sum()) for s in ["train", "val", "test"]}
    stages = [("1  Raw NHTSA file", [f"{fmt(f['raw_rows'])} rows", f"{fmt(f['raw_complaints'])} complaints", "2020 to Oct 2026"], "io"),
              ("2  Cleaned dataset", [f"{fmt(f['cleaned_complaints'])} complaints", f"{len(CLASSES)} component classes", "one label per complaint", "leaks removed"], "layer"),
              ("3  Splits", ["Temporal (main):", f"train {fmt(t['train'])}", f"val {fmt(t['val'])}, test {fmt(t['test'])}", "Random 80/10/10:", f"train {fmt(r['train'])}", f"val {fmt(r['val'])}, test {fmt(r['test'])}"], "layer"),
              ("4  Three models", ["MLP on TF-IDF", "BiLSTM", "DistilBERT", "early stopping on", "validation macro-F1"], "layer"),
              ("5  Evaluation", ["test set scored once", "accuracy, macro-F1", "confusion matrices", "bootstrap intervals,", "McNemar tests"], "out")]
    notes = ["keep complaints with one\ncomponent; strip leaks;\ndrop duplicates and\ntexts under 20 words",
             "temporal: train before Nov 2025,\nval Nov 2025 to Apr 2026,\ntest May to Oct 2026;\ncaps 5,000 / 800 per class",
             "same text, same labels,\nsame splits", "best checkpoint\nby validation score"]
    n, gap = len(stages), 0.55
    bw = (15 - 0.4 - gap * (n - 1)) / n
    for i, (ttl, lines, kind) in enumerate(stages):
        x = 0.2 + i * (bw + gap)
        fill, edge = KIND[kind]
        ax.add_patch(FancyBboxPatch((x, 1.85), bw, 2.75, boxstyle="round,pad=0.0,rounding_size=0.12", fc=fill, ec=edge, lw=1.7))
        ax.text(x + bw / 2, 4.45, ttl, ha="center", va="top", fontproperties=S.HEAD, fontsize=18, color=S.INK)
        ax.text(x + bw / 2, 3.9, "\n".join(lines), ha="center", va="top", fontsize=10, color=S.INK_2 if kind != "out" else S.INK, linespacing=1.45)
        if i < n - 1:
            ax.add_patch(FancyArrowPatch((x + bw + 0.04, 3.25), (x + bw + gap - 0.04, 3.25), arrowstyle="-|>", mutation_scale=16, lw=1.8, color=S.ASPHALT))
            ax.text(x + bw + gap / 2, 1.7, notes[i], ha="center", va="top", fontsize=8.8, color=S.INK_2, linespacing=1.35)
    ax.text(0.2, 0.3, f"Cleaning funnel: {fmt(f['in_class_list'])} complaints in the {len(CLASSES)} classes, minus {fmt(f['dropped_no_failure'])} complaints reporting no failure, "
            f"{fmt(f['dropped_duplicates'])} duplicates and {fmt(f['dropped_short'])} short texts = {fmt(f['cleaned_complaints'])}.", fontsize=10, color=S.INK_2, va="bottom")
    fig.subplots_adjust(0, 0, 1, 1)
    save(fig, "fig_pipeline", svg=True)


# ============================================================================ cleaning example
def word_layout(fig, ax, words, marks, x0, y0, width, fs, lh):
    """Place words left to right with wrapping; ax is in pixel-like coordinates (y down)."""
    r = fig.canvas.get_renderer()
    x, y = x0, y0
    for w, hl in zip(words, marks):
        t = ax.text(x, y, w, fontsize=fs, va="top", ha="left", color=S.INK, zorder=4)
        wpx = t.get_window_extent(r).width / fig.dpi * 100
        if x + wpx > x0 + width and x > x0:
            x, y = x0, y + lh
            t.set_position((x, y))
        if hl:
            ax.add_patch(Rectangle((x - 2, y - 1), wpx + 4, lh - 3, fc=S.AMBER_SOFT, ec="none", zorder=3))
        x += wpx + fs * 100 / 72 * 0.34
    return y + lh


def cleaning_example():
    cols = {1: "odino", 3: "make", 4: "model", 5: "year", 11: "compdesc", 16: "ldate", 19: "text"}
    ODINO = "11308255"
    raw = None
    for fn in ["COMPLAINTS_RECEIVED_2020-2024.txt", "COMPLAINTS_RECEIVED_2025-2026.txt"]:
        d = pd.read_csv(RAW_DIR / fn, sep="\t", header=None, usecols=list(cols), quoting=csv.QUOTE_NONE, dtype=str, keep_default_na=False)
        d.columns = [cols[c] for c in sorted(cols)]
        if (d.odino == ODINO).any():
            raw = d[d.odino == ODINO].iloc[0]
            break
    cl = pd.read_csv(CLEAN_DIR / "complaints_temporal.csv", dtype=str, keep_default_na=False)
    cl = cl[cl.odino == ODINO].iloc[0]
    raw_text = re.sub(r"\s+", " ", raw.text).strip()
    rw, cw = raw_text.split(), cl.text.split()
    norm = lambda w: re.sub(r"[^a-z0-9]", "", w.lower())
    sm = difflib.SequenceMatcher(None, [norm(w) for w in rw], [norm(w) for w in cw], autojunk=False)
    removed = np.ones(len(rw), dtype=bool)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            removed[i1:i2] = False
    json.dump({"odino": ODINO, "raw_fields": {k: raw[k] for k in ["odino", "make", "model", "year", "compdesc", "ldate"]}, "raw_text": raw_text,
               "clean_row": {k: cl[k] for k in ["odino", "ldate", "make", "model", "year", "label", "is_transcript"]}, "clean_text": cl.text,
               "removed_raw_words": [w for w, r_ in zip(rw, removed) if r_]}, open(DATA / "cleaning_example.json", "w"), indent=2)
    W, H = 1300, 540
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
    ax.text(30, 18, "One complaint, before and after cleaning", fontproperties=S.HEAD, fontsize=26, va="top", color=S.INK)
    colw = 600
    for x, head, sub in [(30, "Raw NHTSA row", f"ODINO {raw.odino}  |  {raw.make} {raw.model} {raw.year}  |  component: {raw.compdesc}  |  received {raw.ldate}"),
                         (670, "Cleaned dataset row", f"odino {cl.odino}  |  label: {cl.label}  |  is_transcript: {cl.is_transcript}  |  ldate {cl.ldate}")]:
        ax.add_patch(FancyBboxPatch((x, 72), colw, 320, boxstyle="round,pad=0,rounding_size=10", fc="white", ec=S.STEEL_LIGHT, lw=1.6))
        ax.add_patch(Rectangle((x, 72), colw, 8, fc=S.ASPHALT if "Raw" in head else S.AMBER, ec="none"))
        ax.text(x + 16, 92, head, fontproperties=S.HEAD, fontsize=18, va="top", color=S.INK)
        ax.text(x + 16, 122, "\n".join(textwrap.wrap(sub, 78)), fontsize=8.2, color=S.INK_2, va="top")
    word_layout(fig, ax, rw, list(removed), 46, 180, colw - 32, 9.6, 22)
    word_layout(fig, ax, cw, [False] * len(cw), 686, 180, colw - 32, 9.6, 22)
    ax.add_patch(Rectangle((30, 418), 24, 14, fc=S.AMBER_SOFT, ec=S.AMBER, lw=0.8))
    ax.text(62, 416, "Removed by the cleaning rules. Everything else is kept; the text is also lowercased and extra spaces are collapsed.", fontsize=9.5, color=S.INK_2, va="top")
    ax.text(30, 446, "Removed here: the TL* marker, the campaign number with its component name (written by NHTSA, not the owner),\n"
            "the sentence \"parts distribution disconnect\", and the reviewer tags *BF*TP.   Side effect: a dangling \"notification of.\" remains.",
            fontsize=9.5, color=S.INK_2, va="top", linespacing=1.5)
    ax.text(30, 510, SOURCE_NOTE, fontsize=8.2, color=S.MUTED, va="top")
    save(fig, "fig_cleaning_example")


# ============================================================================ charts
def class_balance_chart(bal, tot):
    d = bal.sort_values("cleaned_pool")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 6.2), sharey=True, gridspec_kw={"width_ratios": [1, 1.05]})
    y = np.arange(len(d))
    a1.barh(y, d.cleaned_pool, color=S.ASPHALT_2, height=0.62)
    for yi, v in zip(y, d.cleaned_pool):
        a1.text(v + 600, yi, fmt(v), va="center", fontsize=11, color=S.INK_2)
    a1.set_yticks(y); a1.set_yticklabels([PLAIN[c] for c in d["class"]], fontsize=11.5)
    a1.set_xlim(0, d.cleaned_pool.max() * 1.17)
    S.title(a1, f"Before sampling: {fmt(tot['pool'])} cleaned complaints", size=19)
    shade = {"train": S.ASPHALT_2, "val": S.STEEL, "test": S.AMBER}
    left = np.zeros(len(d))
    for s in ["train", "val", "test"]:
        v = d[f"temporal_{s}"].values
        a2.barh(y, v, left=left, color=shade[s], height=0.62, edgecolor=S.CARD, linewidth=1.6,
                label={"train": "train (max 5,000)", "val": "validation (max 800)", "test": "test (max 800)"}[s])
        left += v
    for yi, v in zip(y, left):
        a2.text(v + 150, yi, fmt(v), va="center", fontsize=11, color=S.INK_2)
    a2.set_xlim(0, left.max() * 1.18)
    a2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3)
    S.title(a2, f"After sampling (temporal split): {fmt(tot['temporal'])} complaints", size=19)
    for a in (a1, a2):
        S.style_axes(a); a.set_xlabel("Complaints"); a.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
    S.footnote(fig, "Classes are capped so large ones (engine) do not dominate. Small classes keep every complaint they have.")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "fig_class_balance")


def per_class_f1_chart():
    f1 = pd.read_csv(RESULTS_DIR / "eval" / "per_class_f1_all.csv", index_col=0)
    order = f1["distilbert_temporal"].sort_values().index
    fig, ax = plt.subplots(figsize=(10.5, 6.4))
    y = np.arange(len(order))
    cols = [f"{m}_temporal" for m in MODELS]
    ax.hlines(y, f1.loc[order, cols].min(axis=1), f1.loc[order, cols].max(axis=1), color=S.GRID, lw=5, zorder=1)
    for m in MODELS:
        ax.scatter(f1.loc[order, f"{m}_temporal"], y, s=95, marker=S.MODEL_MARKER[m], color=S.MODEL_COLOR[m], edgecolor=S.MODEL_EDGE[m], linewidth=1.3, zorder=3, label=NICE[m])
    for yi, c in zip(y, order):
        S.bold_num(ax, f1.loc[c, "distilbert_temporal"] + 0.008, yi + 0.3, f"{f1.loc[c, 'distilbert_temporal']:.3f}", size=13, va="center")
    ax.set_yticks(y); ax.set_yticklabels([PLAIN[c] for c in order]); ax.set_xlim(0.72, 1.0)
    ax.set_xlabel("F1 score on the temporal test set (higher is better)")
    S.title(fig, "Electrical system is the hardest class for every model", size=25)
    ax.legend(loc="lower right", markerscale=1.1); S.style_axes(ax)
    S.footnote(fig, "Labels show DistilBERT. All values: results/eval/per_class_f1_all.csv")
    fig.tight_layout(rect=(0, 0.03, 1, 0.92))
    save(fig, "fig_per_class_f1")


def split_comparison_chart():
    s = pd.read_csv(RESULTS_DIR / "summary_table.csv")
    fig, ax = plt.subplots(figsize=(10, 5.6))
    off = 0.13
    for i, m in enumerate(MODELS):
        for sc, d, hollow in [("temporal", -off, False), ("random", off, True)]:
            r = s[(s.model == NICE[m]) & (s.split == sc)].iloc[0]
            x = i + d
            ax.vlines(x, r.test_macro_f1_ci_low, r.test_macro_f1_ci_high, color=S.MODEL_EDGE[m], lw=2.2, zorder=2)
            ax.scatter([x], [r.test_macro_f1], s=150, marker=S.MODEL_MARKER[m], zorder=3, color=S.CARD if hollow else S.MODEL_COLOR[m], edgecolor=S.MODEL_EDGE[m], linewidth=2.2)
            S.bold_num(ax, x + (0.045 if sc == "random" else -0.045), r.test_macro_f1, f"{r.test_macro_f1:.3f}", size=15, ha="left" if sc == "random" else "right", va="center")
    ax.set_xticks(range(3)); ax.set_xticklabels([NICE[m] for m in MODELS], fontsize=12.5)
    ax.set_xlim(-0.55, 2.55); ax.set_ylim(0.855, 0.92); ax.set_ylabel("Test macro-F1")
    S.title(ax, "A random split scores 0.9 to 1.5 points higher", size=21)
    ax.legend(handles=[Line2D([], [], marker="o", color=S.INK_2, mfc=S.INK_2, lw=0, ms=10, label="temporal split (main), filled"),
                       Line2D([], [], marker="o", color=S.INK_2, mfc=S.CARD, mew=2, lw=0, ms=10, label="random split, hollow")], loc="upper left")
    S.style_axes(ax, "y")
    S.footnote(fig, "Dots: test macro-F1. Lines: 95% bootstrap interval (results/significance.json). The axis does not start at zero.")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, "fig_split_comparison")


# ============================================================================ plain-language figures
def complaints_per_year_chart(per_year):
    fig, ax = plt.subplots(figsize=(10, 5.6))
    peak = per_year.complaints.idxmax()
    cols = [S.AMBER if i == peak else S.ASPHALT_2 for i in range(len(per_year))]
    cols[-1] = S.STEEL_LIGHT
    ax.bar(per_year.year, per_year.complaints, color=cols, width=0.66, edgecolor=[S.ASPHALT if i == peak else "none" for i in range(len(per_year))], linewidth=1.4)
    for x, v in zip(per_year.year, per_year.complaints):
        S.bold_num(ax, x, v + per_year.complaints.max() * 0.015, fmt(v), size=17, ha="center")
    labels = list(per_year.year[:-1]) + [f"{per_year.year.iloc[-1]}\n(Jan to Oct)"]
    ax.set_xticks(range(len(per_year))); ax.set_xticklabels(labels, fontsize=12.5)
    ax.set_ylim(0, per_year.complaints.max() * 1.13); ax.set_yticks([])
    S.title(ax, "Vehicle complaints sent to the US government, by year", size=22)
    ax.spines["left"].set_visible(False); ax.tick_params(length=0)
    S.footnote(fig, f"Each complaint is counted once, however many car parts the owner listed. {SOURCE_NOTE}")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "fig_complaints_per_year")


def quote_data():
    """Data for the HTML report-ticket cards on the Introduction (no image). Chosen by hand from a seed-42 sample for readability and
    because they contain no names, places or phone numbers."""
    def sentence_case(t):
        t = re.sub(r"\bi\b", "I", t)
        return re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), t)

    picks = [("11745554", "Air bags"), ("11753120", "Fuel system"), ("11771757", "Brakes")]
    t = pd.read_csv(CLEAN_DIR / "complaints_temporal.csv", dtype=str, keep_default_na=False)
    out = []
    for k, (odino, plain) in enumerate(picks, 1):
        r = t[t.odino == odino].iloc[0]
        assert r.is_transcript == "False" and PLAIN[r.label].startswith(plain.split()[0])
        out.append({"card": k, "odino": odino, "label": r.label, "plain_label": plain, "make": r.make, "year": r.year,
                    "received": pd.Timestamp(f"{r.ldate[:4]}-{r.ldate[4:6]}-01").strftime("%B %Y"), "excerpt": sentence_case(r.text),
                    "note": "Capital letters restored for reading; wording unchanged from the cleaned dataset."})
    json.dump(out, open(DATA / "quote_cards.json", "w"), indent=2)


def identified_correctly_chart():
    pc = pd.read_csv(RESULTS_DIR / "eval" / "distilbert" / "temporal" / "per_class.csv").set_index("class")
    n = jload(RESULTS_DIR / "eval" / "distilbert" / "temporal" / "metrics.json")["n"]
    d = (pc.recall * 100).round().astype(int).rename(index=PLAIN).sort_values()
    pd.DataFrame({"class": list(d.index), "percent_sorted_correctly": d.values}).to_csv(DATA / "identified_correctly.csv", index=False)
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    hard = set(d.index[:2])
    ax.barh(range(len(d)), d.values, color=[S.AMBER if c in hard else S.ASPHALT_2 for c in d.index], height=0.66,
            edgecolor=[S.ASPHALT if c in hard else "none" for c in d.index], linewidth=1.4)
    for i, v in enumerate(d.values):
        S.bold_num(ax, v + 0.9, i, f"{v}", size=18, va="center")
    ax.set_yticks(range(len(d))); ax.set_yticklabels(d.index, fontsize=12.5)
    ax.set_xlim(0, 108); ax.set_xticks([])
    S.title(fig, "Out of every 100 complaints of each kind,\nhow many did the computer put in the right group?", size=25)
    ax.spines["bottom"].set_visible(False); ax.tick_params(length=0)
    S.footnote(fig, f"Tested on {fmt(n)} complaints received May to October 2026 that the computer had not seen before. Amber marks the two hardest kinds.")
    fig.tight_layout(rect=(0, 0.04, 1, 0.84))
    save(fig, "fig_identified_correctly")


def mixups_chart():
    cm = pd.read_csv(RESULTS_DIR / "eval" / "distilbert" / "temporal" / "confusion_matrix.csv", index_col=0)
    tot = cm.sum(axis=1)
    panels = [("Engine and fuel system", [("FUEL/PROPULSION SYSTEM", "ENGINE"), ("ENGINE", "FUEL/PROPULSION SYSTEM")]),
              ("Engine and transmission", [("POWER TRAIN", "ENGINE"), ("ENGINE", "POWER TRAIN")]),
              ("Electrical complaints put elsewhere", None)]
    elec = cm.loc["ELECTRICAL SYSTEM"].drop("ELECTRICAL SYSTEM").sort_values(ascending=False).head(4)
    panels[2] = (panels[2][0], [("ELECTRICAL SYSTEM", c) for c in elec.index])
    mx = max(cm.loc[a, b] for _, ps in panels for a, b in ps)
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 7.8), gridspec_kw={"height_ratios": [2, 2, 4]})
    rows = []
    for ax, (ttl, ps) in zip(axes, panels):
        vals = [int(cm.loc[a, b]) for a, b in ps]
        labs = [f"{PLAIN[a]} complaints put under\n“{PLAIN[b]}”" for a, b in ps]
        ax.barh(range(len(ps)), vals, color=S.RED, height=0.6)
        for i, ((a, b), v) in enumerate(zip(ps, vals)):
            S.bold_num(ax, v + mx * 0.012, i, f"{v} of {int(tot[a])}", size=15, va="center")
            rows.append({"panel": ttl, "true_class": a, "put_under": b, "count": v, "of_total": int(tot[a])})
        ax.set_yticks(range(len(ps))); ax.set_yticklabels(labs, fontsize=10.5); ax.invert_yaxis()
        ax.set_xlim(0, mx * 1.25); ax.set_xticks([])
        ax.set_title(ttl, loc="left", fontproperties=S.HEAD, fontsize=18, color=S.INK, pad=8)
        ax.spines["bottom"].set_visible(False); ax.tick_params(length=0)
    pd.DataFrame(rows).to_csv(DATA / "mixups.csv", index=False)
    S.title(fig, "The most common mix-ups", size=24)
    S.footnote(fig, "Counts come from the same 7,320 test complaints. Bars share one scale. Red marks complaints put in the wrong group.", y=0.006)
    fig.tight_layout(rect=(0, 0.025, 1, 0.94))
    save(fig, "fig_mixups")


# ============================================================================ curves, copy to docs/img
def redraw_curves():
    """Redraw training curves from history.csv in the new style (no retraining)."""
    for m in MODELS:
        for sc in ["temporal", "random"]:
            h = pd.read_csv(RESULTS_DIR / m / sc / "history.csv").to_dict("records")
            plot_curves(h, RESULTS_DIR / m / sc / "curves.png", f"{NICE[m]}, {sc} split")


def copy_all():
    DOCS_IMG.mkdir(parents=True, exist_ok=True)
    for old in DOCS_IMG.glob("*"):
        old.unlink()
    n = 0
    for f in MADE:
        shutil.copy2(FIG / f, DOCS_IMG / f); n += 1
    for m in MODELS:
        for sc in ["temporal", "random"]:
            for src, dst in [(RESULTS_DIR / m / sc / "curves.png", f"curves_{m}_{sc}.png"),
                             (RESULTS_DIR / "eval" / m / sc / "confusion_counts.png", f"confusion_{m}_{sc}_counts.png"),
                             (RESULTS_DIR / "eval" / m / sc / "confusion_normalized.png", f"confusion_{m}_{sc}_normalized.png")]:
                shutil.copy2(src, DOCS_IMG / dst); n += 1
    print(f"copied {n} images to {DOCS_IMG}")


def main():
    FIG.mkdir(parents=True, exist_ok=True); DATA.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "eda").mkdir(exist_ok=True)
    for stale in FIG.glob("fig_quote_card_*.png"):
        stale.unlink()
    funnel, per_year = build_funnel()
    bal, tot = build_class_balance()
    print("funnel:", funnel)
    architecture_diagrams()
    pipeline_diagram(funnel, tot, bal)
    cleaning_example()
    class_balance_chart(bal, tot)
    per_class_f1_chart()
    split_comparison_chart()
    complaints_per_year_chart(per_year)
    quote_data()
    identified_correctly_chart()
    mixups_chart()
    redraw_curves()
    copy_all()
    print("figures:", ", ".join(MADE))


if __name__ == "__main__":
    main()
