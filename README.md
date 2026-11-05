# What Failed on This Car?

A deep learning class project by a team. It classifies vehicle owner complaints filed with the US National Highway Traffic Safety Administration (NHTSA) by the component that failed, using only the complaint text. Three models are compared (an MLP on TF-IDF features, a BiLSTM, and a fine-tuned DistilBERT) and everything is published as a static website.

- Website: <https://tejassnaikk.github.io/vehicle-complaint-classifier/> (live once GitHub Pages is enabled from `/docs`)
- Data source: NHTSA complaint flat files, <https://www.nhtsa.gov/nhtsa-datasets-and-apis> (field dictionary: <https://static.nhtsa.gov/odi/ffdd/cmpl/CMPL.txt>)

## Results at a glance

Test set scored once per model with its best validation checkpoint. Main result = temporal split (train before 2025-11-01, validation 2025-11 to 2026-04, test 2026-05 to 2026-10). Full tables, intervals and significance tests: `results/summary_table.csv`, `results/significance.json`, and the Results page.

<!-- RESULTS_START -->
| Model | Split | Test accuracy | Test macro-F1 (95% bootstrap interval) | Training time |
|---|---|---|---|---|
| MLP (TF-IDF) | temporal | 0.8762 | 0.8771 (0.8687 to 0.8843) | 24 s |
| BiLSTM | temporal | 0.8687 | 0.8690 (0.8604 to 0.8764) | 34 min |
| DistilBERT | temporal | 0.8913 | 0.8921 (0.8849 to 0.8988) | 32 min |
| MLP (TF-IDF) | random | 0.8851 | 0.8860 (0.8786 to 0.8929) | 21 s |
| BiLSTM | random | 0.8835 | 0.8840 (0.8772 to 0.8908) | 30 min |
| DistilBERT | random | 0.9060 | 0.9066 (0.9000 to 0.9128) | 43 min |

10 classes, 198,505 cleaned complaints, 64,445 in the published temporal sample. Electrical system is the weakest class for every model on the temporal test set.
<!-- RESULTS_END -->

## Repository layout

```
config.py              paths, seed (42), class list, merge map, split dates, caps
requirements.txt       pinned versions
src/
  01_download.py       download and unzip the two NHTSA chunks into data/raw/
  02_explore.py        exploration figures and counts (results/eda/)
  03_clean_and_label.py  labels, cleaning rules, leakage audit -> data/interim/complaints_clean.parquet
  04_split_and_sample.py temporal and random splits with caps -> data/clean/*.csv
  05_train_mlp.py  06_train_bilstm.py  07_train_distilbert.py   models (results/<model>/<split>/)
  08_evaluate.py       metrics, confusion matrices, significance tests, word importance (results/eval/)
  09_make_figures.py   all website figures (results/figures/, docs/img/)
  10_build_site.py     writes docs/*.html from results/
  check_site.py        checks site numbers trace to results/, tag balance
  common.py            shared data loading, metrics, plotting helpers
  style.py             shared figure style (palette, fonts) used by every matplotlib figure
  site_screenshots.py  full-page screenshots of every site page at 1440 and 390 px (needs Google Chrome)
assets/fonts/          Barlow Condensed and Inter (Open Font License) used in the figures
data/clean/            published sampled datasets (about 43 MB each)
results/               every number and figure the website uses
docs/                  the website: HTML, css/style.css, js/site.js, img/ (GitHub Pages serves from /docs)
PROGRESS.md            log of each phase: what was done, numbers, decisions
```

`data/raw/`, `data/interim/`, `models/` and `venv/` are not in git.

## Reproduce from a fresh clone

Developed and run step by step on a MacBook Air (M4, 16 GB), macOS, Python 3.12. The whole sequence has not been re-run end to end from a fresh clone; what was checked from a fresh clone is described in the notes below. Free disk space needed: about 1.5 GB (raw data 0.6 GB, interim data 0.1 GB, checkpoints 0.7 GB). Run every command from the repository root.

```zsh
git clone https://github.com/tejassnaikk/vehicle-complaint-classifier.git
cd vehicle-complaint-classifier

# 1. Environment (uv: https://docs.astral.sh/uv/)
uv venv --python 3.12 venv
source venv/bin/activate
uv pip install -r requirements.txt
python -c "import torch; print(torch.backends.mps.is_available())"   # True on Apple silicon; scripts fall back to CPU otherwise

# 2. Data (download 119 MB, about 0.6 GB unzipped)
python src/01_download.py                 # a few seconds on a fast connection
python src/02_explore.py                  # about 40 s; expect 629,231 rows
python src/03_clean_and_label.py          # about 2.5 min; expect 198,505 cleaned complaints
python src/04_split_and_sample.py         # under 1 min; expect 64,445 (temporal) and 65,416 (random) rows

# 3. Models (keep the Mac awake for the long runs: caffeinate -i)
python src/05_train_mlp.py --split temporal     # about 25 s
python src/05_train_mlp.py --split random       # about 25 s
caffeinate -i python src/06_train_bilstm.py --split temporal --device mps   # about 34 min
caffeinate -i python src/06_train_bilstm.py --split random --device mps     # about 30 min
caffeinate -i python src/07_train_distilbert.py --full --split temporal     # about 32 min
caffeinate -i python src/07_train_distilbert.py --full --split random       # about 43 min

# 4. Evaluation, figures, website
python src/08_evaluate.py                 # about 30 s (needs the checkpoints in models/)
python src/09_make_figures.py             # about 30 s (needs data/raw/ and data/interim/)
python src/10_build_site.py               # under 1 min
python src/site_screenshots.py            # optional; about 1.5 min; screenshots to results/site_screenshots/
python src/check_site.py                  # checks numbers, tags
```

One command for steps 2 to 4 (about 2.5 to 3 hours; run it from the Terminal app so closing the editor cannot stop it):

```zsh
mkdir -p results/run_all && nohup caffeinate -ims ./run_all.sh > results/run_all/launch.out 2>&1 &
tail -f results/run_all/summary.log      # progress; each step also logs to results/run_all/<step>.log
```

`run_all.sh` runs 03 to 10 in order, stops at the first error and sets `TOKENIZERS_PARALLELISM=false`. It also refreshes the results table above (`src/update_readme_table.py`).

Notes:

- Seed 42 is used everywhere. Time-based splits are deterministic. In a fresh clone of this repository, re-training the MLP on the temporal split gave exactly the same validation and test metrics as the committed ones, and rebuilding the site changed nothing in `docs/`. Training the BiLSTM and DistilBERT on the Apple GPU (MPS) may differ in the last digits across PyTorch versions and machines.
- You can skip training and rebuild the site and figures from the committed `results/` and `data/clean/` alone: `python src/10_build_site.py` regenerates `docs/` without any data download. `08_evaluate.py` needs the checkpoints, and `09_make_figures.py` needs the raw and interim data.
- The website needs no build tools. Fonts for the page text come from Google Fonts; every page also works without JavaScript.
- DistilBERT downloads `distilbert-base-uncased` (about 270 MB) on first use. If it is too slow on your machine, a Colab notebook can load the CSVs from this repository.
- Runtimes above were measured on the machine named above; the BiLSTM and DistilBERT times scale with the hardware.

## Data, privacy and license

- The complaints are NHTSA data, published by the US government for public use. This project is not made by or endorsed by NHTSA.
- The raw files also contain the VIN, city, dealer name, phone and address, and vehicle operator name. The scripts never read those columns, and the published CSVs keep only `odino, ldate, make, model, year, label, text, is_transcript, split`.
- The `text` column is NHTSA's complaint text. NHTSA hides some personal details in it as `[xxx]`. Contact details that people typed into the text (phone numbers, e-mail addresses, VIN-like strings, street addresses and URLs) are replaced with `[PHONE]`, `[EMAIL]`, `[VIN]`, `[ADDRESS]` and `[URL]` by rule 14 of `src/masking.py`, applied in `03_clean_and_label.py` before the data is split. `src/privacy_scan.py` checks the published CSVs for what is left (`results/privacy_scan.json`). Status of this commit: the rule is written and tested on a copy of the cleaned data, and `run_all.sh` regenerates the published CSVs and every result with the rule applied (see `PROGRESS.md`).
- Code license: MIT (see `LICENSE`). The NHTSA complaint data is US government data and is in the public domain. The cleaned samples in `data/clean/` are derived from it. Fonts in `assets/fonts/` are under the SIL Open Font License.

## Limits

One dataset and one six-month test window. Only complaints that map to a single component group are used (about 30% of complaints list more than one group and are left out). Classes are capped, so the test set is balanced and does not match the real mix of complaints. Labels are NHTSA's own and are not independently checked. See the Data and Results pages for the leakage audit, the stripped template text, and where the models fail.
