"""Rewrite the results block of README.md from results/ so the README never shows stale numbers.

Replaces everything between <!-- RESULTS_START --> and <!-- RESULTS_END --> with the model table (results/summary_table.csv),
the dataset size line (results/data_funnel.json, results/class_balance.csv) and the weakest class per model (results/eval/per_class_f1_all.csv).
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import PLAIN_NAMES, RESULTS_DIR, ROOT

START, END = "<!-- RESULTS_START -->", "<!-- RESULTS_END -->"


def block():
    s = pd.read_csv(RESULTS_DIR / "summary_table.csv")
    lines = ["| Model | Split | Test accuracy | Test macro-F1 (95% bootstrap interval) | Training time |", "|---|---|---|---|---|"]
    for sp in ["temporal", "random"]:
        for _, r in s[s.split == sp].iterrows():
            t = f"{r.train_time_s:.0f} s" if r.train_time_s < 120 else f"{r.train_time_s / 60:.0f} min"
            lines.append(f"| {r.model} | {sp} | {r.test_accuracy:.4f} | {r.test_macro_f1:.4f} ({r.test_macro_f1_ci_low:.4f} to {r.test_macro_f1_ci_high:.4f}) | {t} |")
    f = json.loads((RESULTS_DIR / "data_funnel.json").read_text())
    cb = pd.read_csv(RESULTS_DIR / "class_balance.csv")
    n_t = int(cb[["temporal_train", "temporal_val", "temporal_test"]].sum().sum())
    f1 = pd.read_csv(RESULTS_DIR / "eval" / "per_class_f1_all.csv", index_col=0)
    weakest = {m: f1[f"{m}_temporal"].idxmin() for m in ["mlp", "bilstm", "distilbert"]}
    if len(set(weakest.values())) == 1:
        w = f"{PLAIN_NAMES[next(iter(weakest.values()))]} is the weakest class for every model on the temporal test set."
    else:
        w = "Weakest class on the temporal test set: " + ", ".join(f"{m}: {PLAIN_NAMES[c]}" for m, c in weakest.items()) + "."
    lines += ["", f"{len(cb)} classes, {f['cleaned_complaints']:,} cleaned complaints, {n_t:,} in the published temporal sample. {w}"]
    return "\n".join(lines)


def main():
    p = ROOT / "README.md"
    t = p.read_text()
    assert START in t and END in t, "README.md has no results markers"
    new = re.sub(re.escape(START) + r".*?" + re.escape(END), START + "\n" + block() + "\n" + END, t, flags=re.S)
    p.write_text(new)
    print("README results block updated")


if __name__ == "__main__":
    main()
