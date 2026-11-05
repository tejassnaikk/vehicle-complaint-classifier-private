"""Scan the published CSVs for contact details inside the complaint text. Writes results/privacy_scan.json.

Patterns: US-style phone numbers, e-mail addresses, VIN-like 17-character strings, URLs, and "state ZIP" address endings.
Run before any public release: the counts should be 0 for phone, email and VIN once the text has been masked.
"""
import html
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CLEAN_DIR, RESULTS_DIR, ROOT

PATTERNS = {
    "phone_number": r"(?<![\d,])(?:\+?1[ .-]?)?\(?\d{3}\)?[ .-]\d{3}[ .-]\d{4}(?![\d,])",
    "email_address": r"[\w.+-]+@[\w-]+\.[\w.]+",
    "vin_like_17_chars": r"\b(?=[a-z0-9]*\d)(?=[a-z0-9]*[a-z])[a-hj-npr-z0-9]{17}\b(?![-+]\d)",  # the last part skips floats written like 1.5e-05
    "url": r"https?://|www\.",
    "state_and_zip": r"[a-z]{3,},?\s(?:al|ak|az|ar|ca|co|ct|de|dc|fl|ga|ia|il|ks|ky|la|ma|md|mi|mn|ms|mo|mt|ne|nv|nh|nj|nm|ny|nc|nd|pa|ri|sc|sd|tn|tx|ut|vt|va|wa|wv|wi|wy|pr)\s\d{5}\b",
}


def main():
    out = {"patterns": PATTERNS, "files": {}}
    for f in ["complaints_temporal.csv", "complaints_random.csv"]:
        d = pd.read_csv(CLEAN_DIR / f, dtype=str, keep_default_na=False)
        res = {"rows": len(d)}
        for k, p in PATTERNS.items():
            m = d.text.str.contains(p, regex=True, case=False)
            res[k] = {"texts": int(m.sum()), "pct_of_texts": round(float(m.mean() * 100), 2)}
        out["files"][f] = res
        print(f, {k: v["texts"] for k, v in res.items() if k != "rows"})
    # the website (visible text only, so links to github.com / nhtsa.gov are not counted) and the example CSVs/JSON in results/
    docs = {f.name: html.unescape(re.sub(r"<[^>]+>", " ", re.sub(r"(?s)<(script|style).*?</\1>", "", f.read_text()))) for f in sorted((ROOT / "docs").glob("*.html"))}
    res = {str(f.relative_to(ROOT)): f.read_text(errors="ignore") for f in sorted(RESULTS_DIR.rglob("*")) if f.suffix in {".csv", ".json", ".html"}
           and f.name not in {"privacy_scan.json", "masking_preview.json"} and "site_screenshots" not in f.parts}
    for label, texts in [("docs/*.html (visible text)", docs), ("results/**/*.csv|json|html", res)]:
        hits = {k: sum(len(re.findall(p, t, flags=re.I)) for t in texts.values()) for k, p in PATTERNS.items()}
        out["files"][label] = {"files": len(texts), **{k: {"matches": v} for k, v in hits.items()}}
        print(label, hits)
    (RESULTS_DIR / "privacy_scan.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
