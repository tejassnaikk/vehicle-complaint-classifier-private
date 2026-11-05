"""Full-page screenshots of every page at 1440 px and 390 px wide with headless Chrome -> results/site_screenshots/.

Headless Chrome will not render narrower than about 500 px, so each page is shown inside an iframe of the target width.
Tall pages are captured in tiles and stitched. Needs Google Chrome. Run: python src/site_screenshots.py [page ...]
"""
import math
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
OUT = ROOT / "results" / "site_screenshots"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
PAGES = ["index", "introduction", "data", "models", "results", "conclusion", "about"]
WIDTHS = [1440, 390]
TILE = 7000


def chrome(args, timeout=120):
    return subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--allow-file-access-from-files",
                           "--virtual-time-budget=9000", *args], capture_output=True, text=True, timeout=timeout)


def measure(tmp, page, width):
    w = tmp / f"m_{page}_{width}.html"
    w.write_text(f'<!doctype html><title>x</title><iframe id=f src="file://{DOCS / (page + ".html")}" width="{width}" height="900" style="border:0"></iframe>'
                 '<script>var f=document.getElementById("f");f.onload=function(){setTimeout(function(){'
                 'document.title="H"+f.contentDocument.documentElement.scrollHeight;},1500)}</script>')
    out = chrome(["--dump-dom", f"file://{w}"]).stdout
    import re
    m = re.search(r"<title>H(\d+)</title>", out)
    return int(m.group(1)) if m else None


def shoot(page, width, tmp):
    h = measure(tmp, page, width)
    if not h:
        print(f"  could not measure {page} at {width}")
        return None
    tiles = []
    for k in range(math.ceil(h / TILE)):
        th = min(TILE, h - k * TILE)
        w = tmp / f"t_{page}_{width}_{k}.html"
        w.write_text(f'<!doctype html><body style="margin:0;background:#fff"><div style="width:{width}px;height:{th}px;overflow:hidden">'
                     f'<iframe src="file://{DOCS / (page + ".html")}" width="{width}" height="{h}" style="border:0;display:block;margin-top:-{k * TILE}px"></iframe></div>')
        png = tmp / f"t_{page}_{width}_{k}.png"
        chrome([f"--window-size={width},{th}", f"--screenshot={png}", f"file://{w}"])
        tiles.append(Image.open(png).convert("RGB").crop((0, 0, width, th)))
    full = Image.new("RGB", (width, h), "white")
    y = 0
    for t in tiles:
        full.paste(t, (0, y))
        y += t.size[1]
    return full


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pages = sys.argv[1:] or PAGES
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for page in pages:
            for width in WIDTHS:
                im = shoot(page, width, tmp)
                if im is None:
                    continue
                path = OUT / f"{page}_{width}.png"
                im.convert("P", palette=Image.ADAPTIVE, colors=160).save(path, optimize=True)
                print(f"{path.name}: {im.size[0]}x{im.size[1]}, {path.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
