# Project: What Failed on This Car?

Deep learning class project (open project, individual). Classify NHTSA vehicle owner complaints by the vehicle component that failed, using only the complaint text. Compare three neural network models and publish everything as a public website.

## Hard deadlines

- Video presentation posted: **Dec 2, 2026** (late = -50% per day). Target: all results done by Nov 24.
- Final website submission: **Dec 4, 2026, 11:59pm MT**. No late submissions possible.

## Working rules (follow these every session)

1. Work **one phase at a time**. At the end of each phase, STOP. Report what you did, show the key numbers or outputs, and list anything I need to decide. Wait for me to say "proceed".
2. Before any long-running job (download, training run over ~5 minutes), tell me the expected time and size first.
3. Give shell commands as copy-paste zsh. Verify each step worked (file exists, row counts, shapes) before moving on.
4. Never invent results or numbers. Every number on the website must come from a file in `results/`.
5. At the end of every phase: commit with a clear message, push to origin, and update `PROGRESS.md` with what was done, key numbers, decisions I made, and what's next. The GitHub repo `tejassnaikk/vehicle-complaint-classifier` is private from Phase 0; Phase 9 makes it public and enables Pages.
6. Keep it simple. No frameworks, databases, Docker, or cloud services unless I ask. Plain Python scripts + a static HTML site.
7. Fixed random seed 42 everywhere.

## Machine

- MacBook Air M4, 16 GB RAM, zsh. Internal disk space is tight.
- The repo lives on the internal disk. Do not put the git repo, venv, or any SQLite file on the external SSD.
- The raw NHTSA download goes in `data/raw/` on the internal disk (gitignored). Do not use the external SSD. The raw path is a config value (`RAW_DIR` in `config.py`).
- Python 3.12 venv built with uv (`venv/`). Use PyTorch with the MPS backend. Fall back to CPU automatically if MPS fails. If DistilBERT is too slow locally, prepare a Colab notebook instead (see Phase 5).

## Repo layout

```
vehicle-complaint-classifier/
  CLAUDE.md
  README.md                 # how to reproduce, step by step
  requirements.txt          # pinned versions
  config.py                 # paths, seed, class list, sample sizes
  src/
    01_download.py
    02_explore.py
    03_clean_and_label.py
    04_split_and_sample.py
    05_train_mlp.py
    06_train_bilstm.py
    07_train_distilbert.py
    08_evaluate.py
    09_make_figures.py
    common.py               # shared: dataset loading, metrics, plotting helpers
  notebooks/
    distilbert_colab.ipynb  # only if needed
  data/
    raw/                    # gitignored
    interim/                # gitignored
    clean/                  # published sampled dataset (must be < 50 MB)
  models/                   # gitignored
  results/                  # metrics json, confusion matrices, curves, error examples
  docs/                     # the website (GitHub Pages serves from /docs)
    index.html  introduction.html  data.html  models.html  results.html  conclusion.html  about.html
    css/style.css
    img/
```

## Data source

- NHTSA Office of Defects Investigation complaints flat files. Data dictionary: https://static.nhtsa.gov/odi/ffdd/cmpl/CMPL.txt
- Tab-delimited, no header row, 51 fields (see dictionary). Key fields:
  - 2 `ODINO`: complaint reference number. **Repeats when a complaint lists multiple components.**
  - 4 `MAKETXT`, 5 `MODELTXT`, 6 `YEARTXT`
  - 12 `COMPDESC`: component description (the label source), hierarchical, e.g. `SERVICE BRAKES, HYDRAULIC: FOUNDATION COMPONENTS`
  - 17 `LDATE`: date complaint received (YYYYMMDD)
  - 20 `CDESCR`: complaint narrative text (the model input)
- NHTSA publishes the full file `FLAT_CMPL.zip` and 5-year chunks `COMPLAINTS_RECEIVED_YYYY-YYYY.zip`. Use only the most recent chunk(s), roughly 2020 to present. Do not download the full file unless the chunk files are unavailable.

## Privacy rule for published data

The raw file contains VIN, city, dealer name/phone/address, and vehicle operator name. The published cleaned CSV keeps ONLY: `odino, ldate, make, model, year, label, text`. Drop everything else.

## Phases

### Phase 0: Setup
- Create the repo structure above, `.gitignore` (data/raw, data/interim, models, venv, .DS_Store), Python venv, `requirements.txt` (pandas, numpy, scikit-learn, torch, transformers, datasets, matplotlib, seaborn, tqdm). Pin versions after install.
- Verify `torch.backends.mps.is_available()`.
- `git init`, first commit.
- STOP and report.

### Phase 1: Download
- Find the exact chunk filenames. Try `curl -sI` on candidate URLs under `https://static.nhtsa.gov/odi/ffdd/cmpl/` (e.g. `COMPLAINTS_RECEIVED_2020-2024.zip`, `COMPLAINTS_RECEIVED_2025-2029.zip`). Report what exists and file sizes BEFORE downloading.
- If nothing resolves, STOP and ask me to download manually from NHTSA's datasets page.
- Download, unzip, report row counts and the first 3 rows (text truncated).
- STOP.

### Phase 2: Explore (`02_explore.py`)
- Load with explicit column names from the dictionary, `sep="\t"`, `quoting=csv.QUOTE_NONE`, encoding fallback (latin-1 if utf-8 fails), `on_bad_lines="warn"`. Report rows loaded vs lines in file.
- Report: date range of LDATE; % of ODINOs with more than one COMPDESC row; text length distribution (characters and words); % empty or very short texts.
- Derive `top_level` from COMPDESC: take text before the first `:`, then before the first `,`, strip, uppercase. Print a table of distinct top_level values with counts (complaint-level, after the multi-component rule below).
- Save figures to `results/eda/`: class counts bar chart, text length histogram, complaints per month line chart.
- STOP. I will choose the final class list (expected: 8 to 10 classes, drop UNKNOWN OR OTHER).

### Phase 3: Clean and label (`03_clean_and_label.py`)
- Multi-component rule: group by ODINO; keep only complaints whose rows map to exactly ONE `top_level` value in the chosen class list. Collapse to one row per ODINO.
- Text cleaning (keep it light, document every rule): lowercase; collapse whitespace; strip trailing reviewer tag tokens like `*TR`, `*JS`, `*BF` (verify the pattern on real examples first); drop exact duplicate texts; drop texts under 20 words.
- **Leakage audit:** for each class, print the top 20 distinctive unigrams/bigrams. Flag anything that looks like a form artifact or a component name inserted by NHTSA rather than written by the owner. STOP and show me the audit before removing anything.
- Save `data/interim/complaints_clean.parquet`. Report final counts per class.

### Phase 4: Split and sample (`04_split_and_sample.py`)
- **Temporal split (main):** sort by LDATE. Test = most recent 6 months; validation = the 6 months before that; train = everything earlier.
- Balance: cap train at 5,000 per class, validation and test at 800 per class (sample randomly within each class). If a class can't reach the cap, keep what exists and report it.
- **Random split (comparison):** a stratified random 80/10/10 split from the same pool with the same caps. Used only for MLP and BiLSTM to show how much a random split inflates accuracy vs the temporal split.
- Save to `data/clean/` as CSV with only the published columns + a `split` column (separate files for temporal and random). Check file size < 50 MB.
- Report class counts per split for both schemes. STOP.

### Phase 5: Models
All models: same label encoding, same splits, early stopping on validation macro-F1, save best checkpoint to `models/`, save training curves (loss and val macro-F1 per epoch) and metrics json to `results/<model>/`.

1. `05_train_mlp.py`: TF-IDF (unigrams + bigrams, max 30k features, fit on train only) into a PyTorch MLP: 2 hidden layers (512, 128), ReLU, dropout 0.3, Adam, batch 128.
2. `06_train_bilstm.py`: own tokenizer (vocab from train only, min freq 2, max 30k), max length 256 tokens, trainable embedding 128-d, 2-layer bidirectional LSTM (hidden 128), max-pool + mean-pool over time, dropout 0.3, linear output. Also save attention-free word importance later via occlusion or gradient saliency in Phase 6.
3. `07_train_distilbert.py`: `distilbert-base-uncased`, max length 256, batch 16, lr 2e-5, weight decay 0.01, 3 epochs max with linear warmup.
   - **Before the full run:** time 50 training steps on MPS and report the projected total time. If it is over ~2 hours, STOP and build `notebooks/distilbert_colab.ipynb` instead (loads the CSV from the repo, trains, writes the same results files).

Run MLP and BiLSTM on both temporal and random splits. Run DistilBERT on the temporal split only. STOP after each model and report validation and test numbers.

### Phase 6: Evaluate (`08_evaluate.py`)
For each model on the temporal test set, save to `results/`:
- accuracy, macro-F1, weighted-F1, per-class precision/recall/F1 (json + csv)
- confusion matrix PNG, both counts and row-normalized, readable class labels
- 15 misclassified examples per model with true label, predicted label, and the first 300 characters of text (csv), plus 5 confidently correct examples
- one summary table: model x split scheme (temporal vs random) x accuracy x macro-F1
- for BiLSTM and DistilBERT: word-importance highlights for 6 example complaints (saved as small HTML snippets or PNGs) for the website

STOP and report the summary table. Be honest about weak classes.

### Phase 7: Figures (`09_make_figures.py`)
- Architecture diagrams for all three models (clean block diagrams, PNG/SVG, matplotlib or hand-written SVG; no external services).
- Pipeline diagram: raw NHTSA file to cleaned dataset to splits to models to evaluation.
- Before/after cleaning examples (one raw row vs cleaned row), class balance before and after sampling.
- Copy all website images into `docs/img/`.

### Phase 8: Website (`docs/`)
Static HTML + one CSS file, no frameworks, works on phone width. Same nav bar on every page: Introduction | Data & Prep | Models | Results | Conclusion | About. Light, clean, readable design; visuals on every page.

- `introduction.html`: the topic only, no models or methods. **Leave the prose for me to write.** Provide the page with headings, an outline in HTML comments, and image slots (e.g. complaints per year chart, a few example complaints shown as quote cards). The course requires 100% non-technical writing in paragraphs of 7 to 10 sentences.
- `data.html`: data source with link to NHTSA, the field dictionary subset used, the cleaning steps with before/after examples, label mapping table, multi-component rule, leakage audit, split design, class balance charts, links to the raw source and the cleaned CSVs on GitHub.
- `models.html`: what each model is and how it works, the three architecture diagrams, what we predict and why each model was chosen, hyperparameters table, links to each training script.
- `results.html`: summary table, all confusion matrices, per-class F1 chart, training curves, temporal vs random split comparison, misclassified examples, word-importance highlights, honest discussion of where models fail.
- `conclusion.html`: **Leave the prose for me to write.** Headings, outline in comments, image slots for 2 to 3 simple takeaway visuals (e.g. which car problems are easiest/hardest to identify from an owner's words).
- `about.html`: placeholder for me.

Draft explanatory text on the Data, Models, and Results pages, but wrap every drafted paragraph in `<!-- DRAFT: rewrite in my own words -->` comments so I can find and rewrite them. Plain, direct language. No hype words.

### Phase 9: Reproducibility and release
- README: exact steps to reproduce from a fresh clone (venv, download, scripts in order, expected outputs and runtimes).
- Check every link on the site, every image loads, every number matches `results/`.
- Ask me before making the GitHub repo public and enabling Pages from `/docs`.
