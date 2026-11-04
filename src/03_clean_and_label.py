"""Phase 3: label, clean, and audit for leakage.

Steps (in order):
 1. top_level from COMPDESC (text before first ':', then before first ',', strip, upper).
 2. Apply MERGE_MAP to top_level.
 3. Multi-component rule: keep ODINOs whose rows map to exactly ONE merged top_level (strict:
    rows with out-of-list labels count too), and that label must be in CLASSES. One row per ODINO.
 4. Text cleaning: strip trailing reviewer tags (*TR etc.), lowercase, collapse whitespace,
    drop exact duplicate texts, drop texts under 20 words.
 5. Leakage audit (printed + saved to results/leakage_audit.txt). Nothing is removed for leakage here.
Output: data/interim/complaints_clean.parquet
"""
import csv
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CLASSES, INTERIM_DIR, MERGE_MAP, RAW_DIR, RESULTS_DIR, TEST_START, VAL_START

FILES = ["COMPLAINTS_RECEIVED_2020-2024.txt", "COMPLAINTS_RECEIVED_2025-2026.txt"]
# Published columns only (privacy rule); field positions from the NHTSA dictionary (0-based).
USECOLS = {1: "odino", 3: "make", 4: "model", 5: "year", 11: "compdesc", 16: "ldate", 19: "text"}
MIN_WORDS = 20
TAG_RE = re.compile(r"(?:\s*\*[A-Za-z]{2,3})+\s*$")  # one or more trailing reviewer tags
ANY_TAG_RE = re.compile(r"\*[A-Za-z]{2,3}\b")

OUT = []


def say(*a):
    line = " ".join(str(x) for x in a)
    print(line)
    OUT.append(line)


def top_level(s):
    return s.split(":")[0].split(",")[0].strip().upper()


def load():
    parts = [pd.read_csv(RAW_DIR / f, sep="\t", header=None, usecols=list(USECOLS), names=None,
                         quoting=csv.QUOTE_NONE, dtype=str, keep_default_na=False, encoding="utf-8")
             for f in FILES]
    df = pd.concat(parts, ignore_index=True)
    df.columns = [USECOLS[c] for c in sorted(USECOLS)]
    return df


def label_complaints(df):
    df["top_level"] = df.compdesc.map(top_level).replace(MERGE_MAP)
    n_per = df.groupby("odino").top_level.nunique()
    single_ids = n_per[n_per == 1].index
    one = df[df.odino.isin(single_ids)].drop_duplicates("odino")
    say(f"ODINOs total: {df.odino.nunique():,}; single merged top_level: {len(one):,}; "
        f"multi top_level (dropped): {(n_per > 1).sum():,}")
    one = one[one.top_level.isin(CLASSES)].copy()
    say(f"Single-label and in class list: {len(one):,}")
    one["label"] = one.top_level
    return one.drop(columns=["compdesc", "top_level"])


def clean_text(df):
    n0 = len(df)
    tag_trailing = df.text.str.contains(TAG_RE)
    tag_any = df.text.str.contains(ANY_TAG_RE)
    say(f"\nReviewer tags: trailing tag in {tag_trailing.sum():,} texts ({tag_trailing.mean():.2%}); "
        f"any *XX token anywhere in {tag_any.sum():,}; non-trailing only: {(tag_any & ~tag_trailing).sum():,}")
    say("  trailing tag counts:", df.text[tag_trailing].str.extract(r"(\*[A-Za-z]{2,3})\s*$")[0]
        .str.upper().value_counts().head(12).to_dict())
    ex = df.text[tag_trailing].head(3).str[-70:].tolist()
    say("  examples (last 70 chars):", ex)

    t = df.text.str.replace(TAG_RE, "", regex=True)          # rule 1: strip trailing reviewer tags
    t = t.str.lower()                                         # rule 2: lowercase
    t = t.str.replace(r"\s+", " ", regex=True).str.strip()    # rule 3: collapse whitespace
    df = df.assign(text=t)

    dup = df.duplicated("text", keep="first")                 # rule 4: exact duplicate texts
    say(f"Exact duplicate texts dropped: {dup.sum():,}")
    df = df[~dup]
    nw = df.text.str.split().str.len()
    say(f"Texts under {MIN_WORDS} words dropped: {(nw < MIN_WORDS).sum():,}")  # rule 5
    df = df[nw >= MIN_WORDS]
    say(f"Rows: {n0:,} -> {len(df):,}")
    return df.reset_index(drop=True)


def audit(df, k=20):
    say("\n" + "=" * 80 + "\nLEAKAGE AUDIT (nothing removed yet)\n" + "=" * 80)
    labels = sorted(df.label.unique())
    y = df.label.values

    # A. Distinctive unigrams+bigrams per class: log-odds with informative Dirichlet prior (Monroe et al.)
    cv = CountVectorizer(ngram_range=(1, 2), min_df=30, binary=True, token_pattern=r"(?u)\b\w+\b|[^\w\s]")
    X = cv.fit_transform(df.text)
    vocab = np.array(cv.get_feature_names_out())
    df_all = np.asarray(X.sum(0)).ravel()
    say(f"\nA. Top {k} distinctive terms per class (document-frequency log-odds z-score; "
        f"vocab={len(vocab):,}, min_df=30). Format: term [docs in class / docs overall]")
    prior = df_all / df_all.sum() * 500
    for lab in labels:
        m = y == lab
        a = np.asarray(X[m].sum(0)).ravel()
        b = df_all - a
        na, nb = a.sum(), b.sum()
        la = np.log((a + prior) / (na + prior.sum() - a - prior))
        lb = np.log((b + prior) / (nb + prior.sum() - b - prior))
        z = (la - lb) / np.sqrt(1 / (a + prior) + 1 / (b + prior))
        top = np.argsort(-z)[:k]
        say(f"\n[{lab}]  n={m.sum():,}")
        say("  " + " | ".join(f"{vocab[i]} [{a[i]}/{df_all[i]}]" for i in top))

    # B. Highest-purity terms (appear almost only in one class, and in at least 100 docs)
    say("\nB. Highest-purity terms overall (df >= 100, share of docs in one class >= 90%); "
        "look for form artifacts or inserted component names")
    cls_idx = {lab: (y == lab) for lab in labels}
    rows = []
    for lab, m in cls_idx.items():
        a = np.asarray(X[m].sum(0)).ravel()
        mask = (df_all >= 100) & (a / np.maximum(df_all, 1) >= 0.90)
        for i in np.where(mask)[0]:
            rows.append((lab, vocab[i], int(a[i]), int(df_all[i])))
    rows.sort(key=lambda r: -r[3])
    for lab, term, a, n in rows[:40]:
        say(f"  {lab:28s} {term!r:30s} {a}/{n}")
    if not rows:
        say("  (none)")

    # C. Templated openers / closers (first 5 and last 6 words) with strong class skew
    for name, fn, nwords in [("openers (first 5 words)", lambda s: " ".join(s.split()[:5]), 5),
                             ("closers (last 6 words)", lambda s: " ".join(s.split()[-6:]), 6)]:
        say(f"\nC. Repeated {name}: count >= 150")
        ph = df.text.map(fn)
        vc = ph.value_counts()
        for p_, c in vc[vc >= 150].head(12).items():
            dist = df.label[ph == p_].value_counts(normalize=True)
            say(f"  {c:6d}  top class {dist.index[0]} ({dist.iloc[0]:.0%})  {p_!r}")

    # D. Odd tokens: redaction markers, urls, long digit runs, leftover uppercase-style markers
    say("\nD. Form-artifact patterns (docs containing / class distribution of those docs)")
    pats = {"redaction (foia/(b)(6))": r"foia|\(b\)\(6\)|redacted", "xxx run": r"xxx+",
            "nhtsa mention": r"\bnhtsa\b", "url": r"https?://|www\.", "vin mention": r"\bvin\b",
            "'component' wording": r"\bcomponent\b", "bracketed [..]": r"\[[^\]]{1,40}\]",
            "owner/'consumer' voice": r"the consumer|consumer stated|customer states"}
    for name, p_ in pats.items():
        hit = df.text.str.contains(p_, regex=True)
        if hit.sum():
            dist = df.label[hit].value_counts(normalize=True).head(3)
            say(f"  {name:26s} {hit.sum():6d} docs ({hit.mean():.1%});  top: "
                + ", ".join(f"{i} {v:.0%}" for i, v in dist.items()))

    # E. Class balance by year, to spot time-dependent label/text drift
    say("\nE. Complaints per class per year")
    say(pd.crosstab(df.label, df.ldate.str[:4]).to_string())


def main():
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    df = load()
    say(f"Raw rows: {len(df):,}")
    df = label_complaints(df)
    df = clean_text(df)
    out = df[["odino", "ldate", "make", "model", "year", "label", "text"]]
    out.to_parquet(INTERIM_DIR / "complaints_clean.parquet", index=False)
    say(f"\nSaved {INTERIM_DIR / 'complaints_clean.parquet'}  shape={out.shape}")
    say("\nFinal counts per class:\n" + out.label.value_counts().to_string())

    d = pd.to_datetime(out.ldate, format="%Y%m%d")
    w = np.where(d >= TEST_START, "test", np.where(d >= VAL_START, "val", "train"))
    w = np.where(d >= "2026-11-01", "excluded (partial Nov)", w)
    say("\nCounts per class per temporal window (after cleaning):\n"
        + pd.crosstab(out.label, w).reindex(columns=["train", "val", "test", "excluded (partial Nov)"]).to_string())

    audit(out)
    (RESULTS_DIR).mkdir(exist_ok=True)
    (RESULTS_DIR / "leakage_audit.txt").write_text("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
