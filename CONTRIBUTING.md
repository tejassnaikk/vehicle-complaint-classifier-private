# Contributing

For teammates working on the website text, design or code. Questions about the project itself are answered in [README.md](README.md), [CLAUDE.md](CLAUDE.md) and [PROGRESS.md](PROGRESS.md).

## The one rule

**`docs/*.html` is generated. Never edit it by hand.** Your edit is lost the next time anyone builds the site.

- Website text lives in `src/10_build_site.py`.
- Styles live in `docs/css/style.css` and scripts in `docs/js/` (these two you edit directly).
- Data for the Explore pages is written by `src/11_export_web.py` into `docs/data/`.

## Setup

You need Git, Python 3.12 and [uv](https://docs.astral.sh/uv/). No GPU and no data download are needed for text or design edits.

macOS or Linux:

```zsh
git clone https://github.com/tejassnaikk/vehicle-complaint-classifier.git
cd vehicle-complaint-classifier
uv venv --python 3.12 venv
source venv/bin/activate
uv pip install -r requirements.txt
```

Windows (PowerShell):

```powershell
git clone https://github.com/tejassnaikk/vehicle-complaint-classifier.git
cd vehicle-complaint-classifier
uv venv --python 3.12 venv
venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
```

If `uv` is not installed: `pip install uv` (or see the uv page). The pinned torch version may not install on every platform. For text and design edits you only need `pandas`, `numpy`, `pillow` and `scikit-learn`, so install those if the full install fails.

## After every edit

```zsh
python src/10_build_site.py     # regenerates docs/*.html from results/ and docs/data/
python src/check_site.py        # links, contrast, tags, and every number traced to results/
cd docs && python3 -m http.server 8000
```

Then open <http://localhost:8000> in a browser and look at the pages you changed, on a wide window and a narrow one. Stop the server with Ctrl+C. (On Windows use `python -m http.server 8000` if `python3` is not found.)

`check_site.py` must report no failures. Before the final submission we also run `python src/check_site.py --strict`.

## Git workflow

One branch per change. Never push to `main` directly. Tejas reviews and merges.

```zsh
git checkout main
git pull                                  # always first
git checkout -b short-name-for-change     # for example: fix-intro-typo
# ... edit, build, check, preview ...
git status                                # only the files you meant to change
git add src/10_build_site.py docs/        # name the files; the generated docs/ changes go in the same commit
git commit -m "Say what changed and why"
git push -u origin short-name-for-change
```

Then open a pull request on GitHub (the push prints a link, or run `gh pr create --fill`). Tejas reviews it and merges it. After the merge:

```zsh
git checkout main
git pull
```

Commit the regenerated `docs/` files together with the source change, so the site on GitHub Pages matches the code.

## What not to commit

- `models/` (checkpoints, about 700 MB), `data/raw/` and `data/interim/`. They are in `.gitignore`. Do not force-add them.
- No retraining is needed for text or design edits. Do not re-run `run_all.sh` unless the team agrees to it; it takes about 3 hours and changes every number.
- `data/clean/*.csv` are the published dataset. Do not regenerate or edit them by hand.

## Rules for site content

- **Every number on the site must come from a file in `results/`.** Do not type a number into the page text. Read it from a results file in `src/10_build_site.py`, as the existing text does. `check_site.py` lists numbers it cannot find.
- **Any new complaint text must be masked text from `data/clean/`.** Never paste a complaint from the raw NHTSA files. After adding any complaint text, run the privacy scan. It must report zero matches for every pattern:

  ```zsh
  python src/privacy_scan.py
  ```

- Writing rules: plain, direct sentences; specific numbers instead of adjectives; no hype words; no em-dashes; "we" and "our", never "I" or "my". The Introduction and Conclusion use no model names and no metric jargon. The full list is in [CLAUDE.md](CLAUDE.md).
- Colors: asphalt, paper and amber from `docs/css/style.css`. Red is only for errors and mix-ups.

## Using Claude Code

Claude Code works well in this repository. `CLAUDE.md` holds the project rules and `PROGRESS.md` holds the history, decisions and numbers, so it can pick up the project without a long briefing. Ask it to follow the rules above (generated `docs/`, numbers from `results/`, privacy scan) and check its diff before you commit.
