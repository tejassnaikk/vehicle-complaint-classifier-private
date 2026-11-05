"""Dry run of the contact-detail masking on the existing cleaned data. Changes no pipeline file.

Reads data/interim/complaints_clean.parquet (unmasked), applies src/masking.py, prints counts and 5 before/after examples,
and writes results/masking_preview.json. The real rule runs inside 03_clean_and_label.py.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import INTERIM_DIR, RESULTS_DIR, SEED
from masking import RULES, mask_contact_details, tag_counts

SHOW = {"phone": "[PHONE]", "email": "[EMAIL]", "address": "[ADDRESS]", "vin": "[VIN]", "url": "[URL]"}
PERSONAL = re.compile(r"gmail|yahoo|hotmail|outlook|aol|icloud|msn|comcast|live\.com")  # do not show personal-looking e-mails in reports


def context(before, after, tag, width=75):
    i = after.find(tag)
    # align the same stretch in the original text by trimming equal prefixes
    start = max(0, i - width)
    pre = after[start:i]
    j = before.find(pre[-30:], max(0, start - 20)) if len(pre) >= 30 else start
    j = j if j >= 0 else start
    k = after.find(" ", i + len(tag) + width)
    tail = after[i + len(tag): i + len(tag) + width]
    # original segment: from j up to the end of the masked item plus the same tail
    end_orig = before.find(tail[:25], j) if len(tail) >= 25 else -1
    orig = before[j: end_orig + len(tail)] if end_orig >= 0 else before[j: j + 2 * width]
    return "..." + orig + "...", "..." + after[start: i + len(tag) + width] + "..."


def main():
    d = pd.read_parquet(INTERIM_DIR / "complaints_clean.parquet")
    d = d[d.ldate <= "20261031"].reset_index(drop=True)
    masked = d.text.map(mask_contact_details)
    changed = masked != d.text
    counts = tag_counts(d.text)
    print(f"Texts in the cleaned pool: {len(d):,}")
    print("Texts containing each kind (before masking):", {k: f"{v:,}" for k, v in counts.items()})
    print(f"Texts changed by the masking rule: {int(changed.sum()):,} ({changed.mean():.1%})")
    placeholders = {k: int(masked.str.count(re.escape(t)).sum()) for k, t in SHOW.items()}
    print("Placeholders inserted:", {k: f"{v:,}" for k, v in placeholders.items()})
    from masking import STREET_RE as _S
    resid = {"phone": r"(?<!\d)(?:\+?1[ -.]?)?\(?\d{3}\)?[ -.]\d{3}[ -.]\d{4}(?!\d)", "email": r"[\w.+-]+@[\w-]+\.[\w.]+",
             "vin_like": r"\b(?=[a-z0-9]*\d)(?=[a-z0-9]*[a-z])[a-hj-npr-z0-9]{17}\b", "url": r"https?://|www\.", "state_and_zip": r"\b[a-z]{2}\s\d{5}\b"}
    residual = {k: int(masked.str.contains(v, regex=True, case=False).sum()) for k, v in resid.items()}
    print("Residual after masking (privacy_scan patterns):", residual)
    for k in ["phone", "email", "url", "state_and_zip"]:
        hit = masked[masked.str.contains(resid[k], regex=True, case=False)]
        print(f"  residual {k}: sample")
        for t in hit.sample(min(6, len(hit)), random_state=SEED) if len(hit) else []:
            m_ = re.search(resid[k], t, re.I)
            print("     ", t[max(0, m_.start() - 40): m_.end() + 25])
    # effect on the later rules: new duplicates and new short texts
    nw = masked.str.split().str.len()
    dup = masked.duplicated(keep="first") & ~d.text.duplicated(keep="first")
    print(f"Rows that would newly drop: {int(dup.sum())} exact duplicates, {int((nw < 20).sum() - (d.text.str.split().str.len() < 20).sum())} under 20 words")
    # 5 examples, one per kind, seed 42; skip personal-looking e-mails
    rng = np.random.default_rng(SEED)
    examples = []
    for kind, tag in SHOW.items():
        idx = np.where(masked.str.contains(re.escape(tag)))[0]
        rng.shuffle(idx)
        for i in idx:
            b, a = d.text[i], masked[i]
            if kind == "email" and PERSONAL.search(b):
                continue
            if kind == "phone" and re.search(r"my |i am |i called", b[max(0, a.find(tag) - 60): a.find(tag)]):
                continue
            ctx_b, ctx_a = context(b, a, tag)
            examples.append({"kind": kind, "odino": d.odino[i], "before": ctx_b, "after": ctx_a})
            break
    for e in examples:
        print(f"\n[{e['kind']}] odino {e['odino']}\n  BEFORE: {e['before']}\n  AFTER : {e['after']}")
    out = {"texts": int(len(d)), "texts_with_each_kind": counts, "texts_changed": int(changed.sum()), "placeholders_inserted": placeholders, "residual_after_masking": residual,
           "newly_duplicate_rows": int(dup.sum()), "examples": examples}
    out["examples"] = [{"kind": e["kind"], "odino": e["odino"], "after": e["after"]} for e in examples]  # "before" would show real contact details
    (RESULTS_DIR / "masking_preview.json").write_text(json.dumps(out, indent=2))
    # review aid: a random sample of address matches, to check for false positives
    from masking import STREET_RE, CITY_ST_ZIP_RE
    nz = d.text[d.text.str.contains(STREET_RE) & ~d.text.str.contains(r"\b[a-z]{2},?\s+\d{5}\b")]
    print(f"\nStreet matches WITHOUT a ZIP after them: {len(nz):,} texts. Random sample of matched text:")
    for t in nz.sample(25, random_state=SEED):
        print("   ", STREET_RE.search(t).group(0))


if __name__ == "__main__":
    main()
