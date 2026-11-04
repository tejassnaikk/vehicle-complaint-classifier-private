"""Phase 4: temporal split (main) and stratified random split (comparison), with per-class caps.

Input : data/interim/complaints_clean.parquet
Output: data/clean/complaints_temporal.csv, data/clean/complaints_random.csv
Columns: odino, ldate, make, model, year, label, text, is_transcript, split
Pool excludes the partial month after TEST_END (Nov 2026) in both schemes.
"""
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CLASSES, CLEAN_DIR, EVAL_CAP, INTERIM_DIR, SEED, TEST_END, TEST_START, TRAIN_CAP, VAL_START

COLS = ["odino", "ldate", "make", "model", "year", "label", "text", "is_transcript", "split"]
CAPS = {"train": TRAIN_CAP, "val": EVAL_CAP, "test": EVAL_CAP}
MAX_MB = 50


def cap_per_class(df):
    """Randomly keep at most CAPS[split] rows per (split, label); keep everything if fewer exist."""
    parts = []
    for (split, label), g in df.groupby(["split", "label"]):
        parts.append(g.sample(n=min(len(g), CAPS[split]), random_state=SEED))
    return pd.concat(parts)


def temporal_split(pool):
    d = pd.to_datetime(pool.ldate, format="%Y%m%d")
    split = pd.Series("train", index=pool.index)
    split[d >= VAL_START] = "val"
    split[d >= TEST_START] = "test"
    return pool.assign(split=split)


def random_split(pool):
    idx_train, idx_rest = train_test_split(pool.index, test_size=0.2, stratify=pool.label, random_state=SEED)
    idx_val, idx_test = train_test_split(idx_rest, test_size=0.5, stratify=pool.label[idx_rest], random_state=SEED)
    split = pd.Series("train", index=pool.index)
    split[idx_val] = "val"
    split[idx_test] = "test"
    return pool.assign(split=split)


def report(name, before, after, path):
    order = ["train", "val", "test"]
    pre = pd.crosstab(before.label, before.split).reindex(index=CLASSES, columns=order)
    post = pd.crosstab(after.label, after.split).reindex(index=CLASSES, columns=order)
    print(f"\n=== {name}: available (before caps) -> kept (after caps) ===")
    tbl = pre.astype(str) + " -> " + post.astype(str)
    tbl.loc["TOTAL"] = [f"{pre[c].sum()} -> {post[c].sum()}" for c in order]
    print(tbl.to_string())
    short = [(c, s, int(pre.loc[c, s]), CAPS[s]) for c in CLASSES for s in order if pre.loc[c, s] < CAPS[s]]
    print("Below cap (kept all):", "; ".join(f"{c} {s} {n}<{cap}" for c, s, n, cap in short) or "none")
    mb = path.stat().st_size / 1e6
    print(f"{path.name}: {len(after):,} rows, {mb:.1f} MB ({'OK' if mb < MAX_MB else 'TOO BIG'}, limit {MAX_MB} MB)")
    return after


def checks(df, temporal):
    assert df.odino.is_unique, "duplicate ODINO"
    assert not df.text.duplicated().any(), "duplicate text"
    assert set(df.label) == set(CLASSES)
    if temporal:
        g = df.groupby("split").ldate.agg(["min", "max"])
        print("Temporal date ranges:\n" + g.loc[["train", "val", "test"]].to_string())
        assert g.loc["train", "max"] < g.loc["val", "min"] and g.loc["val", "max"] < g.loc["test", "min"]
        assert g.loc["test", "max"] <= TEST_END.replace("-", "")


def main():
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(INTERIM_DIR / "complaints_clean.parquet")
    pool = df[df.ldate <= TEST_END.replace("-", "")].reset_index(drop=True)
    print(f"Clean pool: {len(df):,} rows; after excluding dates after {TEST_END}: {len(pool):,}")

    tmp = temporal_split(pool)
    out = cap_per_class(tmp).sort_values(["ldate", "odino"])[COLS]
    path = CLEAN_DIR / "complaints_temporal.csv"
    out.to_csv(path, index=False)
    report("TEMPORAL", tmp, out, path)
    checks(out, temporal=True)

    rnd = random_split(pool)
    out = cap_per_class(rnd).sort_values(["split", "ldate", "odino"])[COLS]
    path = CLEAN_DIR / "complaints_random.csv"
    out.to_csv(path, index=False)
    report("RANDOM (stratified 80/10/10)", rnd, out, path)
    checks(out, temporal=False)

    # Re-read from disk to confirm what was published
    for f in ["complaints_temporal.csv", "complaints_random.csv"]:
        back = pd.read_csv(CLEAN_DIR / f, dtype={"odino": str, "ldate": str, "year": str}, keep_default_na=False)
        print(f"re-read {f}: shape={back.shape}, columns={list(back.columns)}")
        assert list(back.columns) == COLS


if __name__ == "__main__":
    main()
