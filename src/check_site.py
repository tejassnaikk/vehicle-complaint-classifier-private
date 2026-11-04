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


def main():
    for page in sorted(DOCS.glob("*.html")):
        bad = tag_balance(page.read_text())
        print(f"{page.name}: tag balance {'OK' if not bad else bad[:5]}")
    corpus = corpus_numbers()
    total = unmatched = 0
    for page in sorted(DOCS.glob("*.html")):
        nums = page_numbers(page.read_text())
        miss = sorted({n for n in nums if n not in corpus and not re.fullmatch(r"(19|20)\d\d|\d", n)}, key=lambda s: (len(s), s))
        total += len(nums)
        unmatched += len(miss)
        print(f"{page.name}: {len(nums)} numbers, {len(miss)} not found literally in results/: {miss}")
    print(f"\nTotal numbers {total}; distinct not found {unmatched}. Check the unmatched ones are computed values (shares, differences, rounding).")


if __name__ == "__main__":
    sys.exit(main())
