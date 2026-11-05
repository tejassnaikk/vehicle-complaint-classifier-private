"""Phase 3: label, clean, and audit for leakage.

Steps (in order):
 1. top_level from COMPDESC (text before first ':', then before first ',', strip, upper).
 2. Apply MERGE_MAP to top_level.
 3. Multi-component rule: keep ODINOs whose rows map to exactly ONE merged top_level (strict:
    rows with out-of-list labels count too), and that label must be in CLASSES. One row per ODINO.
 4. Text cleaning (see RULES below, applied in order; each rule logs docs changed + examples):
    reviewer tags, lowercase, whitespace, is_transcript flag, TL* marker, FOIA text, VIN-tool
    sentences, NHTSA campaign phrase + component name, parenthesized component names, standalone
    recall IDs, partial whitespace collapse, then MASKING of contact details (rule 14: phone numbers, e-mail addresses,
    VIN-like strings, street addresses, URLs; see src/masking.py). Then exact-duplicate texts dropped and texts under
    20 words dropped (after stripping and masking).
 5. Leakage audit (printed + saved to results/leakage_audit.txt).
Output: data/interim/complaints_clean.parquet, results/cleaning_rules_log.txt
"""
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CLASSES, INTERIM_DIR, MERGE_MAP, RAW_DIR, RESULTS_DIR, TEST_START, VAL_START
sys.path.insert(0, str(Path(__file__).resolve().parent))
from masking import mask_contact_details, tag_counts

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


def component_vocab(df):
    """Names NHTSA uses for components, from the raw COMPDESC values (lowercase)."""
    comp = {c.strip().lower() for c in df.compdesc.unique()}
    tops = {top_level(c).lower() for c in comp}
    names = comp | tops | {k.lower() for k in MERGE_MAP} | {v.lower() for v in MERGE_MAP.values()}
    stems = {stem(w) for c in comp for w in re.findall(r"[a-z]+", c)} | {"and", "electric"}
    return names, stems


def stem(w):
    return w[:-1] if len(w) > 3 and w.endswith("s") else w


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


# ---- stripping rules (applied to lowercased text) ----
ID = r"\d{2,3}[ve]-?\d{6,7}"                      # NHTSA recall id, e.g. 19v757000
FOIA_RE = re.compile(r"\s*\[?\s*(?:information\s+)?redacted[^.]{0,60}?pursuant to the freedom of information act"
                     r"(?:\s*\(foia\))?[\s,]*(?:5\s*u\.?s\.?c\.?)?\s*(?:552\s*(?:\(b\)\s*(?:\(\d\))?)?)?\.?")  # tail optional: NHTSA truncates at 2048 chars
VIN_RE = re.compile(r"[^.!?]*\bvin tools?\b[^.!?]*(?:[.!?]|$)")
ITEM = rf"{ID}(?:\s*\([^()]*\))?"
CAMPAIGN_RE = re.compile(rf"\s*\bnhtsa\s+(?:campaign|recall)(?:\s+(?:numbers?|nos?\.?|#))?\s*:?\s*{ITEM}(?:\s*(?:,|and|&)?\s*{ITEM})*")
PAREN_RE = re.compile(r"\s*\(\s*([^()]{2,150}?)\s*\)")
ID_RE = re.compile(rf"\s*\b(?:{ID}|\d{{2,3}}[ve]-?\d{{3,7}})\b")
# Recall-availability calls: owner/contact reports no failure at all -> drop the whole complaint.
NOFAIL_RE = re.compile(r"\b(?:contact|client|vehicle|consumer|owner)\s+(?:had|has|have|did)\s+not\s+(?:yet\s+)?experience[d]?\s+"
                       r"(?:a|an|any|the)?\s*(?:failures?|issues?|problems?)\b")
# Template sentence "parts distribution disconnect" (and close variants) -> strip the phrase, keep the text.
PDD_RE = re.compile(r"\s*\b(?:the\s+)?parts?\s+distribution\s+disconnect\w*(?:\s+for\b)?[\u201d\"]?\.?")
TL_RE = re.compile(r"(?:^|\s)tl\*\s*")

RULE_LOG = {}  # rule name -> (n changed, [(before, after), ...])


def snippet_pair(b, a, w=70):
    i = next((k for k, (x, y) in enumerate(zip(b, a)) if x != y), min(len(a), len(b)))
    j = len(b) - len(a)
    return b[max(0, i - w): i + w + max(j, 0) + 20], a[max(0, i - w): i + w]


def run_rule(name, series, fn):
    new = series.map(fn)
    ch = new != series
    ex = series[ch].sample(min(3, ch.sum()), random_state=42).index if ch.any() else []
    RULE_LOG[name] = (int(ch.sum()), [snippet_pair(series[i], new[i]) for i in ex])
    say(f"  rule {name:34s} changed {ch.sum():7,} texts")
    return new


def clean_text(df, names, stems):
    n0 = len(df)
    say(f"\nReviewer tags: trailing tag in {df.text.str.contains(TAG_RE).sum():,} texts; "
        f"mid-text *XX tokens (left alone): "
        f"{(df.text.str.contains(ANY_TAG_RE) & ~df.text.str.contains(TAG_RE)).sum():,}")
    say("  trailing tag counts:", df.text.str.extract(r"(\*[A-Za-z]{2,3})\s*$")[0]
        .str.upper().value_counts().head(12).to_dict())
    say("Applying rules in order:")
    t = df.text
    t = run_rule("1 strip trailing reviewer tags", t, lambda x: TAG_RE.sub("", x))
    t = run_rule("2 lowercase", t, str.lower)
    t = run_rule("3 collapse whitespace", t, lambda x: re.sub(r"\s+", " ", x).strip())
    df = df.assign(text=t)
    df["is_transcript"] = t.str.contains(TL_RE) | t.str.startswith("the contact")
    say(f"  is_transcript flag (TL* marker or starts 'the contact'): {df.is_transcript.sum():,} "
        f"({df.is_transcript.mean():.1%})")
    t = run_rule("4 strip leading TL* marker", t, lambda x: TL_RE.sub("", x))
    t = run_rule("5 strip FOIA redaction text", t, lambda x: FOIA_RE.sub("", x))
    t = run_rule("6 strip VIN-tool sentences", t, lambda x: VIN_RE.sub("", x))

    def campaign(m):
        keep = []
        for inner in re.findall(r"\(([^()]*)\)", m.group(0)):
            toks = re.findall(r"[a-z]+", inner.lower())
            comp_like = inner.strip().lower() in names or (toks and all(stem(w) in stems for w in toks) and len(inner) < 150)
            if not comp_like:
                keep.append(f" ({inner})")
        return "".join(keep)

    def paren(m):
        return "" if re.sub(r"\s+", " ", m.group(1)).strip().lower() in names else m.group(0)

    t = run_rule("7 strip 'nhtsa campaign number: ID (COMPONENT)'", t, lambda x: CAMPAIGN_RE.sub(campaign, x))
    t = run_rule("8 strip parenthesized component names", t, lambda x: PAREN_RE.sub(paren, x))
    t = run_rule("9 strip standalone recall IDs", t, lambda x: ID_RE.sub("", x))
    t = run_rule("10 re-collapse whitespace", t, lambda x: re.sub(r"\s+", " ", x).strip())
    df = df.assign(text=t)

    # Rule 11: drop "no failure" recall-availability complaints; rule 12: strip "parts distribution disconnect"
    nf = df.text.str.contains(NOFAIL_RE)
    pdd = df.text.str.contains(PDD_RE)
    exact_nf = df.text.str.contains("the contact had not experienced a failure")
    say(f"\n  rule 11 drop 'contact had not experienced a failure' (+variants): {nf.sum():,} texts "
        f"({exact_nf.sum():,} exact phrase, {(nf & ~exact_nf).sum():,} variants)")
    cls = pd.DataFrame({"dropped": df.label[nf].value_counts(), "class_total": df.label.value_counts()}).fillna(0).astype(int)
    cls["pct_of_class"] = (cls.dropped / cls.class_total * 100).round(1)
    say("  dropped per class:\n" + cls.to_string())
    ex = df.text[nf].sample(3, random_state=42)
    RULE_LOG["11 drop 'contact had not experienced a failure' complaints"] = (
        int(nf.sum()), [(x[max(0, NOFAIL_RE.search(x).start() - 150): NOFAIL_RE.search(x).end() + 20], "[whole complaint dropped]") for x in ex])
    say(f"  rule 12 'parts distribution disconnect' present in {pdd.sum():,} texts; "
        f"overlap with rule 11: {(pdd & nf).sum():,}; remaining to strip after rule 11: {(pdd & ~nf).sum():,}")
    df = df[~nf].reset_index(drop=True)
    t = run_rule("12 strip 'parts distribution disconnect'", df.text, lambda x: PDD_RE.sub("", x))
    t = run_rule("13 re-collapse whitespace", t, lambda x: re.sub(r"\s+", " ", x).strip())
    df = df.assign(text=t)

    # Rule 14: mask contact details typed into the text (placeholders [PHONE] [EMAIL] [VIN] [ADDRESS] [URL]).
    kinds = tag_counts(df.text)
    say("\n  rule 14 mask contact details. Texts containing each kind before masking: " + ", ".join(f"{k} {v:,}" for k, v in kinds.items()))
    t = run_rule("14 mask contact details", df.text, mask_contact_details)
    name14 = "14 mask contact details"
    n14, ex14 = RULE_LOG[name14]
    # the "before" side of this rule would show real contact details, so the log keeps only the masked side
    RULE_LOG[name14] = (n14, [("(original not saved: it contains contact details)", a) for _, a in ex14])
    df = df.assign(text=t)
    (RESULTS_DIR / "masking_counts.json").write_text(json.dumps(
        {"texts_before_masking": int(len(df)), "texts_containing_each_kind": kinds, "texts_changed": n14,
         "placeholders_inserted": {k: int(df.text.str.count(tag).sum()) for k, tag in
                                   {"phone": "\\[PHONE\\]", "email": "\\[EMAIL\\]", "address": "\\[ADDRESS\\]", "vin": "\\[VIN\\]", "url": "\\[URL\\]"}.items()}}, indent=2))

    dup = df.duplicated("text", keep="first")
    say(f"Exact duplicate texts dropped (after stripping and masking): {dup.sum():,}")
    df = df[~dup]
    nw = df.text.str.split().str.len()
    say(f"Texts under {MIN_WORDS} words dropped (after stripping): {(nw < MIN_WORDS).sum():,}")
    df = df[nw >= MIN_WORDS]
    say(f"Rows: {n0:,} -> {len(df):,}")
    return df.reset_index(drop=True)


def write_rule_log():
    lines = []
    for name, (n, exs) in RULE_LOG.items():
        lines.append(f"### {name}  ({n:,} texts changed)")
        for b, a in exs:
            lines.append(f"  BEFORE: ...{b}...")
            lines.append(f"  AFTER : ...{a}...")
        lines.append("")
    (RESULTS_DIR / "cleaning_rules_log.txt").write_text("\n".join(lines))


def audit(df, names, stems, k=20):
    say("\n" + "=" * 80 + "\nLEAKAGE AUDIT (after stripping)\n" + "=" * 80)
    labels = sorted(df.label.unique())
    y = df.label.values

    # 0. Residual component-name parentheticals and check bigrams
    pa = df.text.str.findall(r"\(\s*([^()]{2,150}?)\s*\)")
    exact = pa.map(lambda l: any(re.sub(r"\s+", " ", x).strip() in names for x in l))
    like = pa.map(lambda l: any((lambda tk: tk and all(stem(w) in stems for w in tk))(re.findall(r"[a-z]+", x)) and len(x) < 60 for x in l))
    say(f"\n0. Texts still containing a parenthesized string equal to a COMPDESC/top-level name: {exact.sum():,}")
    say(f"   Texts containing a parenthesized string made only of component vocabulary words: {like.sum():,}")
    resid = Counter(x.strip() for l in pa[like] for x in l if (lambda tk: tk and all(stem(w) in stems for w in tk))(re.findall(r"[a-z]+", x)))
    say("   most common such parentheticals:", resid.most_common(8))
    nhtsa_ph = df.text.str.contains(r"nhtsa (?:campaign|recall)").sum()
    say(f"   Texts still containing 'nhtsa campaign/recall' wording: {nhtsa_ph:,}; recall-id pattern {ID}: "
        f"{df.text.str.contains(ID).sum():,}; 'foia': {df.text.str.contains('foia').sum():,}; "
        f"'vin tool': {df.text.str.contains('vin tool').sum():,}; 'tl*': {df.text.str.contains(r'tl[*]').sum():,}")

    # A. Distinctive unigrams+bigrams per class: log-odds with informative Dirichlet prior (Monroe et al.)
    cv = CountVectorizer(ngram_range=(2, 2), min_df=30, binary=True, token_pattern=r"(?u)\b\w+\b|[^\w\s]")
    X = cv.fit_transform(df.text)
    vocab = np.array(cv.get_feature_names_out())
    df_all = np.asarray(X.sum(0)).ravel()
    say(f"\nA. Top {k} distinctive BIGRAMS per class (document-frequency log-odds z-score; "
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
    RESULTS_DIR.mkdir(exist_ok=True)
    raw = load()
    names, stems = component_vocab(raw)
    say(f"Raw rows: {len(raw):,}")
    df = label_complaints(raw)
    df = clean_text(df, names, stems)
    out = df[["odino", "ldate", "make", "model", "year", "label", "text", "is_transcript"]]
    out.to_parquet(INTERIM_DIR / "complaints_clean.parquet", index=False)
    say(f"\nSaved {INTERIM_DIR / 'complaints_clean.parquet'}  shape={out.shape}")
    say("\nFinal counts per class:\n" + out.label.value_counts().to_string())

    d = pd.to_datetime(out.ldate, format="%Y%m%d")
    w = np.where(d >= TEST_START, "test", np.where(d >= VAL_START, "val", "train"))
    w = np.where(d >= "2026-11-01", "excluded (partial Nov)", w)
    say("\nCounts per class per temporal window (after cleaning):\n"
        + pd.crosstab(out.label, w).reindex(columns=["train", "val", "test", "excluded (partial Nov)"]).to_string())
    say("\nis_transcript share per class:\n" + out.groupby("label").is_transcript.mean().round(3).to_string())

    audit(out, names, stems)
    (RESULTS_DIR / "leakage_audit.txt").write_text("\n".join(OUT) + "\n")
    write_rule_log()


if __name__ == "__main__":
    main()
