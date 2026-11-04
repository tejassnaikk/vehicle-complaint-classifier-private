"""Phase 2: explore the raw NHTSA complaints. Prints stats and saves figures to results/eda/."""
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import RAW_DIR, RESULTS_DIR

EDA_DIR = RESULTS_DIR / "eda"
FILES = ["COMPLAINTS_RECEIVED_2020-2024.txt", "COMPLAINTS_RECEIVED_2025-2026.txt"]

# 51 fields from https://static.nhtsa.gov/odi/ffdd/cmpl/CMPL.txt
COLS = [
    "CMPLID", "ODINO", "MFR_NAME", "MAKETXT", "MODELTXT", "YEARTXT", "CRASH", "FAILDATE", "FIRE",
    "INJURED", "DEATHS", "COMPDESC", "CITY", "STATE", "VIN", "DATEA", "LDATE", "MILES", "OCCURENCES",
    "CDESCR", "CMPL_TYPE", "POLICE_RPT_YN", "PURCH_DT", "ORIG_OWNER_YN", "ANTI_BRAKES_YN",
    "CRUISE_CONT_YN", "NUM_CYLS", "DRIVE_TRAIN", "FUEL_SYS", "FUEL_TYPE", "TRANS_TYPE", "VEH_SPEED",
    "DOT", "TIRE_SIZE", "LOC_OF_TIRE", "TIRE_FAIL_TYPE", "ORIG_EQUIP_YN", "MANUF_DT", "SEAT_TYPE",
    "RESTRAINT_TYPE", "DEALER_NAME", "DEALER_TEL", "DEALER_CITY", "DEALER_STATE", "DEALER_ZIP",
    "PROD_TYPE", "REPAIRED_YN", "MEDICAL_ATTN", "VEHICLES_TOWED_YN", "STATE_OF_INCIDENT",
    "VEHICLE_OPERATOR",
]
# Only read what we need (privacy fields are never loaded).
USE = ["ODINO", "MAKETXT", "MODELTXT", "YEARTXT", "COMPDESC", "LDATE", "CDESCR"]


def load(path):
    n_lines = sum(chunk.count(b"\n") for chunk in iter(lambda f=open(path, "rb"): f.read(1 << 24), b""))
    enc = "utf-8"
    kw = dict(sep="\t", names=COLS, header=None, usecols=USE, quoting=csv.QUOTE_NONE,
              dtype=str, keep_default_na=False, on_bad_lines="warn")
    try:
        df = pd.read_csv(path, encoding="utf-8", **kw)
    except UnicodeDecodeError:
        enc = "latin-1"
        df = pd.read_csv(path, encoding="latin-1", **kw)
    print(f"{path.name}: encoding={enc}, lines in file={n_lines:,}, rows loaded={len(df):,}")
    return df


def top_level(s):
    return s.split(":")[0].split(",")[0].strip().upper()


def main():
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.concat([load(RAW_DIR / f) for f in FILES], ignore_index=True)
    print(f"\nTotal rows: {len(df):,}   unique ODINO: {df.ODINO.nunique():,}")

    ld = pd.to_datetime(df.LDATE, format="%Y%m%d", errors="coerce")
    print(f"LDATE range: {ld.min().date()} to {ld.max().date()}  (unparseable: {ld.isna().sum()})")

    rows_per = df.groupby("ODINO").size()
    print(f"ODINOs with >1 COMPDESC row: {(rows_per > 1).mean():.1%} "
          f"({(rows_per > 1).sum():,} of {len(rows_per):,}); max rows per ODINO: {rows_per.max()}")

    df["top_level"] = df.COMPDESC.map(top_level)
    tl_per = df.groupby("ODINO").top_level.nunique()
    print(f"ODINOs mapping to >1 distinct top_level: {(tl_per > 1).mean():.1%} "
          f"({(tl_per > 1).sum():,}); exactly 1: {(tl_per == 1).sum():,}")

    # Complaint level: one row per ODINO with a single top_level
    single = df[df.ODINO.isin(tl_per[tl_per == 1].index)].drop_duplicates("ODINO").copy()
    text_check = df.groupby("ODINO").CDESCR.nunique()
    print(f"ODINOs whose rows have differing CDESCR: {(text_check > 1).sum():,}")

    counts = single.top_level.value_counts()
    print(f"\nDistinct top_level values (complaint level, single-component only; n={len(single):,}):")
    print(pd.DataFrame({"count": counts, "pct": (counts / counts.sum() * 100).round(2)}).to_string())

    # Text stats on the same complaint-level set
    chars = single.CDESCR.str.len()
    words = single.CDESCR.str.split().str.len().fillna(0)
    q = [0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    print("\nText length (single-component complaints):")
    print("  chars:", {f"p{int(p*100)}": int(chars.quantile(p)) for p in q}, "mean", round(chars.mean()), "max", chars.max())
    print("  words:", {f"p{int(p*100)}": int(words.quantile(p)) for p in q}, "mean", round(words.mean()), "max", int(words.max()))
    print(f"  empty/whitespace: {(single.CDESCR.str.strip() == '').mean():.2%}   "
          f"<5 words: {(words < 5).mean():.2%}   <10 words: {(words < 10).mean():.2%}   "
          f"<20 words: {(words < 20).mean():.2%}")

    # Figures
    fig, ax = plt.subplots(figsize=(8, 5))
    counts.sort_values().plot.barh(ax=ax, color="#3b6ea5")
    ax.set_xlabel("Complaints"); ax.set_title("Complaints per top-level component (single-component only)")
    fig.tight_layout(); fig.savefig(EDA_DIR / "class_counts.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    words.clip(upper=500).plot.hist(bins=60, ax=ax, color="#3b6ea5")
    ax.set_xlabel("Words per complaint (clipped at 500)"); ax.set_ylabel("Complaints")
    ax.set_title("Complaint text length")
    fig.tight_layout(); fig.savefig(EDA_DIR / "text_length_hist.png", dpi=150); plt.close(fig)

    monthly = pd.to_datetime(single.LDATE, format="%Y%m%d", errors="coerce").dt.to_period("M").value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(monthly.index.to_timestamp(), monthly.values, color="#3b6ea5")
    ax.set_ylabel("Complaints"); ax.set_title("Complaints received per month")
    fig.tight_layout(); fig.savefig(EDA_DIR / "complaints_per_month.png", dpi=150); plt.close(fig)

    counts.rename_axis("top_level").rename("count").to_csv(EDA_DIR / "top_level_counts.csv")
    monthly.rename_axis("month").rename("complaints").to_csv(EDA_DIR / "complaints_per_month.csv")
    print(f"\nSaved figures and CSVs to {EDA_DIR}")
    print(f"Last month in data: {monthly.index[-1]} ({monthly.iloc[-1]:,} complaints; may be partial)")


if __name__ == "__main__":
    main()
