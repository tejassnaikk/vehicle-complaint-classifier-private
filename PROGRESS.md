# Progress

## Phase 0: Setup (done)

**Done**
- Repo scaffold at `~/Developer/vehicle-complaint-classifier` (layout per CLAUDE.md), `.gitignore`, `config.py` (seed 42, caps 5000/800, paths).
- Installed uv 0.13.7 and Python 3.12.16; venv rebuilt on 3.12 (`venv/`, gitignored).
- Packages at current stable versions, pinned in `requirements.txt`: torch 2.15.0, transformers 5.22.1, pandas 3.1.0, numpy 2.5.4, scikit-learn 1.9.1, datasets 5.1.0, matplotlib 3.11.2, seaborn 0.13.2, tqdm 4.70.1, pyarrow 26.0.1.
- `torch.backends.mps.is_available()` = True.
- Private GitHub repo `tejassnaikk/vehicle-complaint-classifier` created, set as origin, main pushed.

**Decisions (Tejas)**
- Python 3.12 via uv instead of system 3.9.
- Raw data lives in `data/raw/` on the internal disk (`RAW_DIR` in config.py). No external SSD.
- Single CLAUDE.md, in the repo (the ~/Downloads copy was deleted).
- GitHub repo created in Phase 0 as private; Phase 9 makes it public and enables Pages.
- End of every phase: commit, push, update this file.

## Phase 1: Download (done)

**Done**
- Chunk URLs under `https://static.nhtsa.gov/odi/ffdd/cmpl/`: `COMPLAINTS_RECEIVED_2020-2024.zip` and `COMPLAINTS_RECEIVED_2025-2026.zip` exist. `2025-2029` returns 404 (the later chunk is named `2025-2026`).
- `src/01_download.py` downloads and unzips both into `data/raw/` (gitignored).
- Sizes: 2020-2024 zip 75,874,098 B, txt 326,690,943 B; 2025-2026 zip 43,215,634 B, txt 192,196,195 B.
- Lines (= rows; every line has exactly 51 tab-separated fields): 418,884 (2020-2024) + 210,347 (2025-2026) = 629,231 rows.
- LDATE range: 20200101 to 20261101.

**Notes for Phase 2**
- Rows repeat ODINO (one per component), so complaint count will be lower than 629,231.
- Newer rows show mojibake (e.g. "dealerâs") when read as latin-1; the file is probably UTF-8 in places. Check encoding handling in Phase 2.

**Decisions (Tejas):** download the two chunks above.

**Next:** Phase 2 (explore). Waiting for "proceed".
