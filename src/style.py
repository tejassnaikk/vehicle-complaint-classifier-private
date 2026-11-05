"""Shared look for every matplotlib figure: palette, fonts, rcParams. Matches docs/css/style.css.

Palette: asphalt charcoal, warm paper, one amber accent, safety red only for errors and misclassifications.
Fonts: Barlow Condensed (headings), Inter (labels). Files in assets/fonts/ (Open Font License).
The amber is too light for text or thin marks on white (contrast about 2:1), so amber marks get a dark outline
and every chart carries direct labels or a legend.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

ROOT = Path(__file__).resolve().parents[1]
FONT_DIR = ROOT / "assets" / "fonts"

ASPHALT, ASPHALT_2, STEEL, STEEL_LIGHT = "#1c1f24", "#2b3138", "#7d8896", "#c5cbd3"
AMBER, AMBER_SOFT, AMBER_DEEP = "#f2a900", "#fbe3a0", "#9a6b00"
RED, RED_SOFT = "#d64545", "#f6d3d3"
PAPER, CARD = "#f7f5f0", "#ffffff"
INK, INK_2, MUTED, GRID = "#1c1f24", "#444a52", "#6a707a", "#e4e1d8"

MODEL_COLOR = {"mlp": ASPHALT_2, "bilstm": STEEL, "distilbert": AMBER}
MODEL_MARKER = {"mlp": "o", "bilstm": "s", "distilbert": "D"}
MODEL_EDGE = {"mlp": ASPHALT_2, "bilstm": ASPHALT_2, "distilbert": ASPHALT}  # amber needs a dark outline

HEAD = HEAD_BOLD = None
BODY_FAMILY = "Inter"


def apply_style():
    """Register fonts and set rcParams. Safe to call more than once."""
    global HEAD, HEAD_BOLD
    for f in sorted(FONT_DIR.glob("*.ttf")):
        fm.fontManager.addfont(str(f))
    HEAD = fm.FontProperties(fname=str(FONT_DIR / "BarlowCondensed-SemiBold.ttf"))
    HEAD_BOLD = fm.FontProperties(fname=str(FONT_DIR / "BarlowCondensed-Bold.ttf"))
    plt.rcParams.update({
        "font.family": [BODY_FAMILY, "DejaVu Sans"], "font.size": 12,
        "figure.facecolor": CARD, "axes.facecolor": CARD, "savefig.facecolor": CARD, "savefig.dpi": 200,
        "text.color": INK, "axes.labelcolor": INK_2, "axes.labelsize": 12.5, "xtick.color": INK_2, "ytick.color": INK_2,
        "xtick.labelsize": 11.5, "ytick.labelsize": 11.5, "axes.edgecolor": GRID, "axes.linewidth": 1.0,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.9,
        "legend.frameon": False, "legend.fontsize": 11.5, "axes.titlelocation": "left",
    })


def title(ax_or_fig, text, size=21, x=None, y=None):
    """Left-aligned heading in Barlow Condensed. For an Axes uses set_title, for a Figure fig.text."""
    if hasattr(ax_or_fig, "set_title"):
        ax_or_fig.set_title(text, loc="left", fontproperties=HEAD, fontsize=size, color=INK, pad=14)
    else:
        ax_or_fig.text(x if x is not None else 0.01, y if y is not None else 0.985, text, fontproperties=HEAD, fontsize=size, color=INK, ha="left", va="top")


def bold_num(ax, x, y, text, size=14, color=INK, **kw):
    """Bold-looking number label (Barlow Condensed Bold); Inter's variable font has no bold here."""
    return ax.text(x, y, text, fontproperties=HEAD_BOLD, fontsize=size, color=color, **kw)


def style_axes(ax, grid_axis="x"):
    ax.grid(axis=grid_axis, color=GRID, lw=0.9)
    ax.set_axisbelow(True)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0)


def footnote(fig, text, y=0.012, size=9.5):
    fig.text(0.012, y, text, fontsize=size, color=MUTED, ha="left", va="bottom")


apply_style()
