"""Check that numbers on the website trace to files in results/.

For every number printed on a page, look for it in the text of results/ (csv, json, txt, md, log), allowing rounding and
percent forms. Numbers that are not found are listed per page: they should be derived values (differences, shares) that the
site builder computes from those files, and they need a human look. Run: python src/check_site.py
"""
import html
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS, RES = ROOT / "docs", ROOT / "results"
NUM = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?")
NUM_PLAIN = re.compile(r"\d+(?:\.\d+)?")  # numbers inside csv/json: commas are separators
NUM_GROUPED = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?")  # 1,234 style numbers in text files


def norm(tok):
    tok = tok.replace(",", "")
    return tok.rstrip(".")


def corpus_numbers():
    vals = set()
    for f in RES.rglob("*"):
        if f.suffix in {".csv", ".json", ".txt", ".md", ".log"} and f.stat().st_size < 30_000_000:
            text = f.read_text(errors="ignore")
            for t in NUM_PLAIN.findall(text) + [norm(x) for x in NUM_GROUPED.findall(text)]:
                t = norm(t)
                vals.add(t)
                try:
                    x = float(t)
                except ValueError:
                    continue
                for d in range(0, 5):
                    vals.add(f"{x:.{d}f}")
                    vals.add(f"{x * 100:.{d}f}")
                vals.add(str(int(round(x))))
    return vals


def page_numbers(h):
    h = re.sub(r"(?s)<(script|style).*?</\1>", "", h)
    h = re.sub(r"<[^>]*>", " ", h)  # drop tags and attributes (alt text is checked separately)
    return [norm(t) for t in NUM.findall(html.unescape(h))]


VOID = {"meta", "link", "img", "br", "hr", "input"}


def tag_balance(h):
    """Return a list of unbalanced tags (simple stack check, ignoring void elements)."""
    from html.parser import HTMLParser

    problems, stack = [], []

    class P(HTMLParser):
        def handle_starttag(self, tag, attrs):
            if tag not in VOID:
                stack.append(tag)

        def handle_endtag(self, tag):
            if tag in VOID:
                return
            if not stack or stack[-1] != tag:
                problems.append(f"</{tag}> closes {stack[-1] if stack else 'nothing'}")
                if tag in stack:
                    while stack and stack.pop() != tag:
                        pass
            else:
                stack.pop()

    P().feed(h)
    return problems + [f"<{t}> never closed" for t in stack]


def hex_lum(h):
    h = h.lstrip("#")
    r, g, b = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    f = lambda v: v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
    return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b)


def ratio(a, b):
    la, lb = hex_lum(a), hex_lum(b)
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)


def contrast_check():
    """WCAG AA: 4.5:1 for text pairs, 3:1 for large text and graphics. Tokens are read from docs/css/style.css (:root)."""
    css = (DOCS / "css" / "style.css").read_text()
    tok = dict(re.findall(r"--([\w-]+):\s*(#[0-9a-fA-F]{6})", css))
    text_pairs = [("ink", "paper"), ("ink", "card"), ("ink-2", "paper"), ("muted", "paper"), ("muted", "card"), ("on-dark", "asphalt"),
                  ("on-dark-2", "asphalt"), ("on-dark-2", "asphalt-2"), ("amber", "asphalt"), ("asphalt", "amber"), ("asphalt", "amber-soft"),
                  ("red-text", "card"), ("red-text", "red-soft"), ("red-text", "paper"), ("amber-deep", "paper"), ("amber-deep", "amber-wash"),
                  ("asphalt", "amber-wash"), ("on-dark", "asphalt-2")]
    graphic_pairs = [("asphalt", "paper"), ("steel", "paper"), ("asphalt", "card")]
    bad = []
    for a, b in text_pairs:
        r = ratio(tok[a], tok[b])
        print(f"  text    {a:10s} on {b:10s} {r:5.2f} {'ok' if r >= 4.5 else 'FAIL'}")
        if r < 4.5:
            bad.append((a, b, r))
    for a, b in graphic_pairs:
        r = ratio(tok[a], tok[b])
        print(f"  graphic {a:10s} on {b:10s} {r:5.2f} {'ok' if r >= 3 else 'FAIL'}")
        if r < 3:
            bad.append((a, b, r))
    return bad


def asset_scan():
    """No dash characters in css/js, no external resources other than Google Fonts."""
    out = []
    for f in [DOCS / "css" / "style.css", DOCS / "js" / "site.js"]:
        t = f.read_text()
        for ch in ["\u2014", "\u2013"]:
            if ch in t:
                out.append(f"{f.name}: dash character")
    for page in DOCS.glob("*.html"):
        for u in re.findall(r'(?:src|href)="(https?://[^"]+)"', page.read_text()):
            if not u.startswith(("https://fonts.googleapis.com", "https://fonts.gstatic.com")) and "nhtsa.gov" not in u and "github.com/tejassnaikk" not in u:
                out.append(f"{page.name}: external resource {u}")
    return out


def brace_scan():
    """Catch template text that was never filled in, e.g. a literal {NHTSA_REPORT} or {n(...)} in a page."""
    out = []
    for page in DOCS.glob("*.html"):
        t = re.sub(r"(?s)<style.*?</style>", "", page.read_text())
        t = html.unescape(re.sub(r"<[^>]+>", " ", t))
        out += [f"{page.name}: {b}" for b in re.findall(r"\{[A-Za-z_][^{}]{0,40}\}", t)]
        out += [f"{page.name}: attribute {b}" for b in re.findall(r'(?:href|src)="[^"]*\{[^"]*"', page.read_text())]
    return out



def placeholder_scan():
    """Bracketed placeholders still to be filled in by the team (a warning, not a failure)."""
    out = []
    for page in DOCS.glob("*.html"):
        txt = html.unescape(re.sub(r"<[^>]+>", " ", page.read_text()))
        out += [f"{page.name}: {m}" for m in re.findall(r"\[(?:Teammate \d|Course name|Semester)[^\]]*\]", txt)]
    return out


def main():
    strict = "--strict" in sys.argv[1:]
    failures, warnings = [], []
    print("Contrast (WCAG AA):")
    cb = contrast_check()
    print("contrast failures:", cb or "none")
    failures += [f"contrast {x}" for x in cb]
    assets = asset_scan()
    print("asset scan:", assets or "clean")
    failures += assets
    braces = brace_scan()
    print("unfilled template text:", braces or "none")
    failures += braces
    holders = placeholder_scan()
    if holders:
        # a warning in the normal run (run_all.sh), a hard failure with --strict (final pre-publish check)
        (failures if strict else warnings).extend(f"placeholder {h}" for h in holders)
        print(f"{'FAIL' if strict else 'WARNING'}: placeholders still to fill before release: {holders}")
    else:
        print("placeholders still to fill before release: none")
    for page in sorted(DOCS.glob("*.html")):
        bad = tag_balance(page.read_text())
        print(f"{page.name}: tag balance {'OK' if not bad else bad[:5]}")
        failures += [f"{page.name}: {b}" for b in bad]
    corpus = corpus_numbers()
    total = unmatched = 0
    for page in sorted(DOCS.glob("*.html")):
        nums = page_numbers(page.read_text())
        miss = sorted({n for n in nums if n not in corpus and not re.fullmatch(r"(19|20)\d\d|\d", n)}, key=lambda s: (len(s), s))
        total += len(nums)
        unmatched += len(miss)
        print(f"{page.name}: {len(nums)} numbers, {len(miss)} not found literally in results/: {miss}")
    print(f"\nTotal numbers {total}; distinct not found {unmatched}. Check the unmatched ones are computed values (shares, differences, rounding).")
    if strict:
        print("\nSTRICT (pre-publish) CHECK:", "FAILED" if failures else "PASSED")
        for f in failures:
            print("  -", f)
        return 1 if failures else 0
    if warnings:
        print(f"\n{len(warnings)} warning(s) (not failing this run). Use `python src/check_site.py --strict` for the final pre-publish check.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
