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
- (Resolved in Phase 2: both files are valid UTF-8; the mojibake seen in Phase 1 came from my latin-1 preview, 0% of rows affected.)

**Decisions (Tejas):** download the two chunks above.


## Phase 2: Explore (done, awaiting class list)

**Done:** `src/02_explore.py` (reads only the 7 needed columns, so privacy fields are never loaded). Figures and CSVs in `results/eda/`.

**Key numbers**
- Rows loaded = lines in file for both files (418,884 + 210,347 = 629,231). Both UTF-8. No bad lines.
- 432,893 unique ODINOs. LDATE 2020-01-01 to 2026-11-01 (Nov 2026 has only 63 complaints, partial month).
- 31.9% of ODINOs have more than one COMPDESC row (max 9). 29.9% (129,329) map to more than one distinct top_level; 303,564 map to exactly one.
- CDESCR is identical across all rows of an ODINO.
- Text (303,564 single-component complaints): median 83 words / 484 chars, p95 250 words, max 430 words. 0% empty; 1.0% under 5 words; 9.35% under 20 words.
- Top-level values: see `results/eda/top_level_counts.csv`. NHTSA renamed some categories over time, so near-duplicates exist (ENGINE vs ENGINE AND ENGINE COOLING, FUEL SYSTEM vs FUEL/PROPULSION SYSTEM, VISIBILITY vs VISIBILITY/WIPER).

**Decisions needed (Tejas):** final class list, and whether to merge the renamed near-duplicates.

## Phase 3: Clean and label (done, awaiting 'proceed' for Phase 4)

**Decisions (Tejas), recorded also in `config.py` as `MERGE_MAP` / `CLASSES`**
- Merges applied to top_level BEFORE the multi-component rule: ENGINE AND ENGINE COOLING -> ENGINE; FUEL SYSTEM -> FUEL/PROPULSION SYSTEM (same physical systems, NHTSA renamed categories).
- VISIBILITY -> VISIBILITY/WIPER only if one fades as the other grows. Per-year complaint counts (2020..2026): VISIBILITY 379, 462, 775, 981, 594, 523, 609; VISIBILITY/WIPER 2518, 1911, 1658, 1839, 1543, 1557, 1513. They coexist, and VISIBILITY content differs (sunroof, windshield, glass, mirrors). So VISIBILITY is dropped, not merged.
- Final 10 classes: ENGINE, POWER TRAIN, ELECTRICAL SYSTEM, STEERING, SERVICE BRAKES, AIR BAGS, STRUCTURE, FUEL/PROPULSION SYSTEM, VISIBILITY/WIPER, FORWARD COLLISION AVOIDANCE. Everything else dropped.
- Windows (partial Nov 2026 excluded): test = 2026-05-01..2026-10-31, val = 2025-11-01..2026-04-30, train = earlier. FORWARD COLLISION AVOIDANCE has 606 test complaints before cleaning (>= 400), so it stays; EXTERIOR LIGHTING not used.
- Multi-component rule is strict: an ODINO is kept only if ALL its rows map to one merged top_level (rows with out-of-list labels count), and that label is in the class list.

**Numbers (`src/03_clean_and_label.py`, `results/leakage_audit.txt`)**
- 432,893 ODINOs -> 304,879 single-label -> 229,063 in class list -> minus 3,439 exact duplicates -> minus 18,529 under 20 words = **207,095** rows saved to `data/interim/complaints_clean.parquet`.
- Trailing reviewer tags in 2.3% of texts (*TR 3,519, *JB 938, *JS 426, *DT 159, *AS 82, ...), stripped; 254 more texts have a tag mid-text (left alone).
- Per class: ENGINE 52,628; POWER TRAIN 32,790; ELECTRICAL 25,190; STEERING 21,293; SERVICE BRAKES 19,660; FUEL/PROPULSION 15,238; AIR BAGS 13,994; STRUCTURE 12,745; VISIBILITY/WIPER 7,418; FORWARD COLLISION 6,139.
- After cleaning, windows (train/val/test): AIR BAGS 12534/697/762; ELECTRICAL 19931/2804/2449; ENGINE 43075/4761/4780; FCA 5113/443/580; FUEL 12656/1207/1372; POWER TRAIN 26072/3183/3529; BRAKES 16861/1388/1408; STEERING 18577/1213/1498; STRUCTURE 10725/951/1068; VISIBILITY/WIPER 6552/446/419.

**Leakage audit findings (nothing removed yet; awaiting Tejas)**
- 11.9% of texts (24,656) contain NHTSA's call-centre phrasing "NHTSA campaign number: 19V757000 (ENGINE AND ENGINE COOLING)": the component name in parentheses, not written by the owner. The parenthetical matches the label 96.9% of the time. Rate by class: AIR BAGS 24.8%, FUEL/PROPULSION 22.3%, SERVICE BRAKES 16.8%, STRUCTURE 13.5%, STEERING 11.2%, POWER TRAIN 9.8%, ELECTRICAL 9.6%, ENGINE 8.8%, VISIBILITY/WIPER 1.0%, FCA 0.4%.
- 29.2% of texts are call-centre transcripts ("The contact owns a 2018 ...", "TL* The contact ..."), not owner-written; rate varies by class (VISIBILITY/WIPER 3.9%, FCA 12.9%, AIR BAGS 34.8%). Also templated closers ("VIN tool confirms parts not available" 6,615; FOIA redaction text 6,915 docs).
- Recall IDs (e.g. 19V757000) appear in about 12.9% of texts; they can map to one component.

### Phase 3: leakage decisions and final cleaning (Tejas)

Decisions: (1) strip NHTSA-inserted component names and the "NHTSA campaign number: ID (COMPONENT)" phrase; (2) strip standalone recall IDs, keep the word "recall"; (3) keep call-centre transcripts, strip the TL* marker only, add `is_transcript` column; (4) strip VIN-tool and FOIA boilerplate, keep the "failure mileage was approximately" sentence; (5) keep brand and model names; (6) keep the strict multi-component rule. After stripping, the 20-word minimum and the duplicate check are re-applied.

**Cleaning rules, in the order applied** (`src/03_clean_and_label.py`; full examples in `results/cleaning_rules_log.txt`). Texts changed out of 229,063:
1. Strip trailing reviewer tags (`*TR`, `*JB`, `*JS`, `*DT`, ...; one or more at the end): 5,266.
   Before: `...EXTEND THE VIN'S TO ALL 2018 MODELS.  *TR` After: `...extend the vin's to all 2018 models.`
2. Lowercase: 227,721.
3. Collapse whitespace: 124,563.
   Before: `...in oakland, california.  i had only driven...` After: `...in oakland, california. i had only driven...`
4. `is_transcript` flag set (TL* marker or text starts "the contact"): 62,756 texts (27.4%). Then strip the TL* marker: 8,249.
   Before: `tl* the contact owns a 2015 jeep grand cherokee.` After: `the contact owns a 2015 jeep grand cherokee.`
5. Strip FOIA redaction text (including truncated versions; NHTSA cuts text at 2,048 characters): 6,759.
   Before: `...while driving. information redacted pursuant to the freedom of information act (foia), 5 u.s.c. 552(b)(6).` After: `...while driving.`
   The `[xxx]` redaction markers are left in (3.4% of texts, spread like the class base rates).
6. Strip any sentence containing "vin tool" (covers "vin tool confirms parts not available" and variants like "vin tools confirms", "vin tool showed no open recall"): 8,521.
   Before: `...the contact had not experienced a failure. vin tool confirms parts not available.` After: `...the contact had not experienced a failure.`
7. Strip "nhtsa campaign/recall number: ID" plus the parenthesized component names that follow (singular or lists, with or without a space before the parenthesis; any parenthesis made only of component words, e.g. "(service brakes, hydraulics)"): 27,424.
   Before: `...related the failure to nhtsa campaign number: 25v274000 (engine and engine cooling), but the vin was not included.` After: `...related the failure to, but the vin was not included.`
   Known side effect: dangling words such as "related the failure to," stay behind.
8. Strip any parenthesized string anywhere that exactly equals an NHTSA COMPDESC value or top-level name (before or after merging): 930.
   Before: `...recall number: 197 - oil leak on alternator (engine and engine cooling). the vehicle was repaired` After: `...oil leak on alternator. the vehicle was repaired`
9. Strip standalone recall IDs: `NNvNNNNNN` (2 or 3 digits, v or e, optional dash, 6 or 7 digits) and the short forms `NNv-NNN` or `NNvNNN...` (3 to 7 digits): 4,188.
   Before: `...concerning nhtsa recall 22v-430 / honda campaign lcd.` After: `...concerning nhtsa recall / honda campaign lcd.`
10. Re-collapse whitespace: 22.
Then: exact duplicate texts dropped 3,671; texts under 20 words dropped 18,573.

**Funnel:** 629,231 rows -> 432,893 ODINOs -> 304,879 single merged label -> 229,063 in the 10 classes -> 206,819 after duplicates and 20-word minimum. (Before stripping it was 207,095.)

**Final counts per class:** ENGINE 52,589; POWER TRAIN 32,768; ELECTRICAL SYSTEM 25,153; STEERING 21,285; SERVICE BRAKES 19,616; FUEL/PROPULSION SYSTEM 15,202; AIR BAGS 13,952; STRUCTURE 12,698; VISIBILITY/WIPER 7,417; FORWARD COLLISION AVOIDANCE 6,139.

**Per-class train/val/test (partial Nov 2026 excluded):** AIR BAGS 12,493/696/762; ELECTRICAL 19,900/2,799/2,448; ENGINE 43,042/4,758/4,777; FCA 5,113/443/580; FUEL/PROPULSION 12,627/1,203/1,369; POWER TRAIN 26,057/3,179/3,526; SERVICE BRAKES 16,820/1,385/1,408; STEERING 18,571/1,212/1,497; STRUCTURE 10,687/942/1,068; VISIBILITY/WIPER 6,551/446/419.

**Residuals after stripping (honest list for the Data page):**
- Parentheses equal to a COMPDESC/top-level name: 0. Parentheses made only of component words: 3,562, but these are owner abbreviations (tcm 138, abs 125, cvt 91, pcm 87, esc 73), kept.
- Still present: "nhtsa campaign/recall" wording 1,908 texts (owner mentions), 7 malformed recall IDs, 18 texts with partial FOIA wording.
- Call-centre template sentences remain in some classes, e.g. "parts distribution disconnect" (3,678 texts; FUEL/PROPULSION 991, AIR BAGS 655) and "the contact had not experienced a failure" (5,207, AIR BAGS 35%). These are kept per decision 3 but are a possible shortcut; Phase 6 reports transcript vs owner-written accuracy.
- `is_transcript` share per class: AIR BAGS 38.9%, ENGINE 32.6%, FUEL/PROPULSION 32.5%, STRUCTURE 32.5%, POWER TRAIN 31.2%, SERVICE BRAKES 30.9%, STEERING 27.3%, ELECTRICAL 23.8%, FCA 12.9%, VISIBILITY/WIPER 3.9%.

**Brand bias note (for the website discussion):** Brand and model names are kept. They appear among the distinctive terms: "kia" in ENGINE, "tesla" in FORWARD COLLISION AVOIDANCE, "ford explorer" and "subaru" in STRUCTURE and VISIBILITY/WIPER, "ford f-150" and "jeep cherokee" in POWER TRAIN. Some well-known problems are tied to specific models, so part of the accuracy may come from recognising a model's known defect instead of reading the symptom. Results can differ for models not seen in training.

**Next:** superseded by the second cleaning round below.

### Phase 3: second cleaning round (Tejas) - supersedes the funnel and per-class numbers above

Two more rules, added after rule 10 and before the duplicate / 20-word checks:

11. **Drop** complaints containing "the contact had not experienced a failure" or close variants (subject contact/client/vehicle/consumer/owner; had/has/have/did not experience(d) a/an/any/the failure/issue/problem). Reason: these are recall-availability calls, not failure descriptions. **10,295 complaints dropped** (10,132 exact phrase, 163 variants).
    Example (whole complaint dropped): `...the manufacturer had exceeded a reasonable amount of time for the recall repair. the manufacturer was made aware of the issue. the contact had not experienced a failure.`
    Dropped per class (share of the class before this rule): AIR BAGS 3,201 (19.5%); FUEL/PROPULSION 1,690 (10.0%); SERVICE BRAKES 1,673 (7.8%); ELECTRICAL 1,192 (4.3%); ENGINE 1,195 (2.1%); STRUCTURE 576 (3.9%); POWER TRAIN 547 (1.5%); STEERING 210 (0.9%); VISIBILITY/WIPER 10 (0.1%); FORWARD COLLISION AVOIDANCE 1 (0.0%).
12. **Strip** the "parts distribution disconnect" template (and variants: part/parts, disconnect/disconnects/disconnection, optional leading "the", optional trailing "for"), keep the rest of the text. The phrase was in 4,145 texts; **3,067 of them were also dropped by rule 11**; the other 1,078 were stripped.
    Before: `...the failure mileage was approximately 53,000. parts distribution disconnect.` After: `...the failure mileage was approximately 53,000.`
13. Re-collapse whitespace (0 changed).

Then duplicates dropped: 1,690 (was 3,671; most duplicates were the template calls now removed). Texts under 20 words dropped: 18,573.
Residuals: 18 texts still contain "not experienced a failure" in other constructions, 6 contain "parts distribution" in other wording. Left as is.

**Updated funnel:** 629,231 rows -> 432,893 ODINOs -> 304,879 single merged label -> 229,063 in the 10 classes -> 218,768 after rule 11 -> 217,078 after duplicates -> **198,505** after the 20-word minimum.

**Updated per-class totals:** ENGINE 51,627; POWER TRAIN 32,267; ELECTRICAL 24,141; STEERING 21,108; SERVICE BRAKES 18,246; FUEL/PROPULSION 13,868; STRUCTURE 12,314; AIR BAGS 11,389; VISIBILITY/WIPER 7,407; FORWARD COLLISION AVOIDANCE 6,138.

**Updated per-class train/val/test (temporal windows, before caps):**

| Class | train | val | test |
|---|---|---|---|
| ENGINE | 42,156 | 4,695 | 4,764 |
| POWER TRAIN | 25,654 | 3,117 | 3,490 |
| ELECTRICAL SYSTEM | 19,073 | 2,669 | 2,393 |
| STEERING | 18,437 | 1,194 | 1,472 |
| SERVICE BRAKES | 15,548 | 1,321 | 1,374 |
| FUEL/PROPULSION SYSTEM | 11,448 | 1,114 | 1,303 |
| STRUCTURE | 10,350 | 907 | 1,056 |
| AIR BAGS | 10,031 | 636 | 721 |
| VISIBILITY/WIPER | 6,541 | 446 | 419 |
| FORWARD COLLISION AVOIDANCE | 5,112 | 443 | 580 |

`is_transcript` share per class now: ENGINE 31.4%, STRUCTURE 30.3%, POWER TRAIN 30.1%, AIR BAGS 27.9%, STEERING 26.7%, FUEL/PROPULSION 26.0%, SERVICE BRAKES 25.7%, ELECTRICAL 20.7%, FCA 12.9%, VISIBILITY/WIPER 3.8%.

## Phase 4: Split and sample (done, awaiting 'proceed')

`src/04_split_and_sample.py` reads `data/interim/complaints_clean.parquet`, drops the 41 complaints dated after 2026-10-31 (partial November) from the pool for both schemes (198,505 -> 198,464), and writes `data/clean/complaints_temporal.csv` and `data/clean/complaints_random.csv`. Columns: odino, ldate, make, model, year, label, text, is_transcript, split. Seed 42. Caps: train 5,000, val/test 800 per class, sampled randomly within each class.

- **Temporal:** train 2020-01-01..2025-10-31, val 2025-11-01..2026-04-30, test 2026-05-01..2026-10-31 (checked: ranges do not overlap). 64,445 rows, 43.2 MB.
- **Random:** stratified 80/10/10 on the same pool (two `train_test_split` calls), then the same caps. 65,416 rows, 43.3 MB.
- Both files are under 50 MB. Checked: ODINOs unique, texts unique, all 10 classes present, CSV re-read gives 9 columns.

Per-class counts kept (available before caps in brackets):

| Class | Temporal train | val | test | Random train | val | test |
|---|---|---|---|---|---|---|
| ENGINE | 5,000 | 800 | 800 | 5,000 | 800 | 800 |
| POWER TRAIN | 5,000 | 800 | 800 | 5,000 | 800 | 800 |
| ELECTRICAL SYSTEM | 5,000 | 800 | 800 | 5,000 | 800 | 800 |
| STEERING | 5,000 | 800 | 800 | 5,000 | 800 | 800 |
| SERVICE BRAKES | 5,000 | 800 | 800 | 5,000 | 800 | 800 |
| AIR BAGS | 5,000 | 636 | 721 | 5,000 | 800 | 800 |
| STRUCTURE | 5,000 | 800 | 800 | 5,000 | 800 | 800 |
| FUEL/PROPULSION SYSTEM | 5,000 | 800 | 800 | 5,000 | 800 | 800 |
| VISIBILITY/WIPER | 5,000 | 446 | 419 | 5,000 | 741 | 740 |
| FORWARD COLLISION AVOIDANCE | 5,000 | 443 | 580 | 4,908 | 614 | 613 |
| **Total** | **50,000** | **7,125** | **7,320** | **49,908** | **7,755** | **7,753** |

Classes below the cap (all rows kept): temporal AIR BAGS val 636 / test 721, VISIBILITY/WIPER val 446 / test 419, FORWARD COLLISION AVOIDANCE val 443 / test 580; random VISIBILITY/WIPER val 741 / test 740, FORWARD COLLISION AVOIDANCE train 4,908 / val 614 / test 613.

Notes for later phases: the temporal and random sets are not the same size or class mix for small classes, so report macro-F1 as well as accuracy. Both CSVs are committed to git (about 86 MB together); regenerating and re-committing them will grow the repo history, so avoid re-committing unless the data changes.

**Next:** Phase 5 (models). Waiting for "proceed".

## Phase 5.1: MLP (done, awaiting 'proceed' for the BiLSTM)

**Built:** `src/common.py` (data loading, label encoding from `config.CLASSES`, device choice with MPS-to-CPU fallback, metrics, per-class report, curve plot, `save_run`) and `src/05_train_mlp.py --split temporal|random`.

**Setup (also in `results/mlp/<split>/config.json`):** TF-IDF unigrams+bigrams, 30,000 features, fit on train only; MLP 30000 -> 512 -> 128 -> 10, ReLU, dropout 0.3, Adam lr 1e-3, batch 128, max 30 epochs, early stopping on validation macro-F1 (patience 3), seed 42, device MPS. Test evaluated once per run, with the best checkpoint. Nothing was tuned on test.

**Results (accuracy / macro-F1):**

| Split scheme | Val acc | Val macro-F1 | Test acc | Test macro-F1 | Best epoch / run | Wall time |
|---|---|---|---|---|---|---|
| Temporal | 0.8698 | 0.8727 | 0.8762 | 0.8771 | 1 / 4 | 24 s |
| Random | 0.8864 | 0.8874 | 0.8851 | 0.8860 | 1 / 4 | 21 s |

Random-split test macro-F1 is only 0.9 points above temporal (0.8860 vs 0.8771) for the MLP.

**3 weakest classes on test (F1):**
- Temporal: ELECTRICAL SYSTEM 0.771 (precision 0.729, recall 0.818), FUEL/PROPULSION SYSTEM 0.832 (precision 0.897, recall 0.776), VISIBILITY/WIPER 0.863.
- Random: ELECTRICAL SYSTEM 0.769, ENGINE 0.846, POWER TRAIN 0.862.

**Observations:** The best epoch is 1 in both runs. Train loss keeps falling (0.72 -> 0.03) while validation loss rises from epoch 2 on, so the MLP overfits fast with 30k sparse features. Early stopping handled it. Any change to regularisation (weight decay, smaller feature set) would be a decision for Tejas, to be judged on validation only.

**Files:** per run in `results/mlp/<split>/`: `config.json`, `metrics.json`, `history.csv`, `per_class_test.csv`, `test_predictions.csv` (with confidence and is_transcript, for Phase 6), `curves.png`, `train.log`. Checkpoints and the fitted TF-IDF in `models/` (gitignored, about 120 MB): `mlp_<split>.pt`, `mlp_<split>_tfidf.pkl`.

**Next:** Phase 5.2 BiLSTM (`06_train_bilstm.py`). Waiting for "proceed".

## Phase 5.2: BiLSTM (done, awaiting 'proceed' for DistilBERT)

**Built:** `src/06_train_bilstm.py` (`--bench`, or `--split temporal|random --device mps|cpu`), reusing `src/common.py`.

**Setup (also in `results/bilstm/<split>/config.json`):** own regex tokenizer (lowercase, words and numbers only, punctuation dropped); vocab from train only, min freq 2 (cap 30k, not reached: 20,860 tokens temporal, 21,277 random, including PAD and UNK); max length 256 tokens; trainable embedding 128; 2-layer bidirectional LSTM, hidden 128, dropout 0.3; packed sequences, and max-pool and mean-pool over time are masked so padding is excluded; dropout 0.3; linear output. Adam lr 1e-3, batch 64 (my choice, not in the spec), gradient clipping 1.0, max 15 epochs, early stopping on validation macro-F1 with patience 3, seed 42. About 3.3M parameters. Test evaluated once per run with the best checkpoint; nothing tuned on test.

**Device timing (1 full training epoch, temporal train, batch 64):** MPS 252 s, CPU 380 s. MPS was used for both runs (about 250 s per epoch).

**Truncation at 256 tokens:** train 4.4% (temporal) / 5.1% (random). Temporal val 6.8%, test 9.1%: later complaints are longer, so the temporal test set is truncated more than train.

**Results (accuracy / macro-F1):**

| Split scheme | Val acc | Val macro-F1 | Test acc | Test macro-F1 | Best epoch / run | Wall time |
|---|---|---|---|---|---|---|
| Temporal | 0.8648 | 0.8683 | 0.8687 | 0.8690 | 5 / 8 | 2,024 s (34 min) |
| Random | 0.8852 | 0.8860 | 0.8835 | 0.8840 | 4 / 7 | 1,786 s (30 min) |

Compared with the MLP (test macro-F1 temporal 0.8771, random 0.8860): the BiLSTM is lower by 0.8 points (temporal) and 0.2 points (random). It is about 80 times slower to train. The random-minus-temporal gap is 1.5 points for the BiLSTM (0.8840 vs 0.8690) and 0.9 points for the MLP.

**3 weakest classes on test (F1):**
- Temporal: ELECTRICAL SYSTEM 0.757 (precision 0.774, recall 0.741), FUEL/PROPULSION SYSTEM 0.826 (0.838, 0.814), ENGINE 0.841 (0.816, 0.866).
- Random: ELECTRICAL SYSTEM 0.771, ENGINE 0.840, POWER TRAIN 0.856.

**Did ELECTRICAL SYSTEM improve?** No on the temporal split: F1 0.757 vs 0.771 for the MLP. On the random split it is about the same (0.771 vs 0.769).

**Top 3 confused class pairs, temporal test set (BiLSTM; 961 errors out of 7,320):**
1. ENGINE and FUEL/PROPULSION SYSTEM: 93 errors (60 true FUEL/PROPULSION predicted ENGINE, 33 the other way).
2. ELECTRICAL SYSTEM and FUEL/PROPULSION SYSTEM: 78 errors (40 and 38).
3. ENGINE and POWER TRAIN: 77 errors (48 true POWER TRAIN predicted ENGINE, 29 the other way).
For the MLP the same top pair is ENGINE / FUEL/PROPULSION (81), then ELECTRICAL / FUEL/PROPULSION (76), then STRUCTURE / VISIBILITY/WIPER (72; for the BiLSTM this pair is fourth in size, with 49 STRUCTURE predicted VISIBILITY/WIPER as its largest direction).

**Early look at transcripts (temporal test, accuracy):** BiLSTM owner-written 0.866 vs transcript 0.878; MLP 0.875 vs 0.881. Small gaps; the full breakdown is for Phase 6.

**Files:** `results/bilstm/<split>/` (config.json, metrics.json, history.csv with per-epoch times, per_class_test.csv, test_predictions.csv, curves.png, train.log) and `results/bilstm/bench.log`. Checkpoints in `models/` (gitignored): `bilstm_<split>.pt` and `bilstm_<split>_vocab.json` (needed for Phase 6 word importance).


## Phase 5.3: DistilBERT (done, Phase 5 complete, awaiting 'proceed' for Phase 6)

**Built:** `src/07_train_distilbert.py` (`--bench`, or `--full --split temporal|random`). The spec said temporal only; Tejas asked for both splits.

**Setup (also in `results/distilbert/<split>/config.json`):** `distilbert-base-uncased`, max length 256 (7.1% of temporal train texts truncated), batch 16, dynamic padding to the longest text in the batch, AdamW lr 2e-5, weight decay 0.01 (not on bias and LayerNorm), 10% linear warmup then linear decay, gradient clipping 1.0, 3 epochs, seed 42, MPS, fp32. Best epoch chosen by validation macro-F1; test scored once with the best checkpoint.

**Benchmark (50 steps, MPS):** 0.174 s/step steady, 3,125 steps/epoch, projected 28 min for 3 epochs (`results/distilbert/temporal/bench.log`). Actual: 32 min temporal, 43 min random (epochs took 9.7 to 11.3 min temporal and 12.5 to 15.5 min random).

**Stall and rerun (random split):** The first random-split run, chained after the temporal run with nohup, stalled overnight: the Mac went to sleep and swap filled up. Tejas killed it, restarted the Mac, and relaunched the random run from Terminal with `caffeinate` and `TOKENIZERS_PARALLELISM=false`. Same script, settings and seed. The log and outputs in `results/distilbert/random/` are from that rerun only (one clean run, 3 epochs), and its slower epochs are probably a leftover effect of the memory pressure. Test was scored once, at the end of the final run. The temporal run was not affected. Lesson for the README: run long jobs under `caffeinate -i`.

### Full comparison (accuracy / macro-F1; test n = 7,320 temporal, 7,753 random)

| Model | Split | Val acc | Val macro-F1 | Test acc | Test macro-F1 | Best epoch / run | Train time |
|---|---|---|---|---|---|---|---|
| MLP | Temporal | 0.8698 | 0.8727 | 0.8762 | 0.8771 | 1 / 4 | 24 s |
| MLP | Random | 0.8864 | 0.8874 | 0.8851 | 0.8860 | 1 / 4 | 21 s |
| BiLSTM | Temporal | 0.8648 | 0.8683 | 0.8687 | 0.8690 | 5 / 8 | 34 min |
| BiLSTM | Random | 0.8852 | 0.8860 | 0.8835 | 0.8840 | 4 / 7 | 30 min |
| DistilBERT | Temporal | 0.8901 | 0.8941 | 0.8913 | 0.8921 | 3 / 3 | 32 min |
| DistilBERT | Random | 0.9023 | 0.9029 | 0.9060 | 0.9066 | 3 / 3 | 43 min |

**Random minus temporal gap (test):** MLP +0.88 pts accuracy / +0.90 pts macro-F1; BiLSTM +1.48 / +1.50; DistilBERT +1.47 / +1.45. (Validation macro-F1 gap: MLP +1.47, BiLSTM +1.78, DistilBERT +0.88.) A random split overstates performance by about 1 to 1.5 points for every model, which is a small inflation.

**Reading the table:** DistilBERT is best on both splits: +1.5 pts test macro-F1 over the MLP on temporal and +2.1 pts on random. The BiLSTM is last on temporal and slightly behind the MLP on random. With about 7,300 test texts the standard error on accuracy is roughly 0.4 pts, so MLP vs BiLSTM (0.8 pts on temporal, 0.2 on random) is not a clear difference, while DistilBERT's lead is larger.

**DistilBERT weakest classes on test (F1):**
- Temporal: ELECTRICAL SYSTEM 0.801 (precision 0.791, recall 0.812), FUEL/PROPULSION SYSTEM 0.848 (0.874, 0.824), ENGINE 0.851 (0.821, 0.882).
- Random: ELECTRICAL SYSTEM 0.815, ENGINE 0.863, FUEL/PROPULSION SYSTEM 0.879.

**ELECTRICAL SYSTEM across models (test F1, temporal / random):** MLP 0.771 / 0.769; BiLSTM 0.757 / 0.771; DistilBERT 0.801 / 0.815. DistilBERT improves it by about 3 to 4.5 points, but it is still the weakest class for every model on both splits.

**Top 3 confused pairs, temporal test (errors in both directions combined):**
- DistilBERT (796 errors of 7,320): ENGINE and FUEL/PROPULSION 79 (54 true FUEL/PROPULSION predicted ENGINE); STRUCTURE and VISIBILITY/WIPER 71 (52 true STRUCTURE predicted VISIBILITY/WIPER); ELECTRICAL SYSTEM and ENGINE 68.
- BiLSTM (961): ENGINE / FUEL/PROPULSION 93; ELECTRICAL / FUEL/PROPULSION 78; ENGINE / POWER TRAIN 77.
- MLP (906): ENGINE / FUEL/PROPULSION 81; ELECTRICAL / FUEL/PROPULSION 76; STRUCTURE / VISIBILITY/WIPER 72.
ENGINE / FUEL/PROPULSION is the top or near-top confusion for all three models, so it is probably a label overlap and not a model weakness. DistilBERT cuts the ELECTRICAL / FUEL/PROPULSION confusion out of its top 3.

**Transcript vs owner-written accuracy (temporal test, early look for Phase 6):** DistilBERT 0.8915 owner-written vs 0.8904 transcript; BiLSTM 0.866 vs 0.878; MLP 0.875 vs 0.881. No sign that transcripts are easier for DistilBERT.

**Files:** `results/distilbert/<split>/` (config.json, metrics.json, history.csv, per_class_test.csv, test_predictions.csv, curves.png, train.log; temporal also bench.log). Checkpoints in `models/` (gitignored, about 268 MB each): `distilbert_temporal.pt`, `distilbert_random.pt`.

**Next:** Phase 6 (evaluate, `08_evaluate.py`): confusion matrices, error examples, summary table, word-importance highlights. Waiting for "proceed".

## Phase 6: Evaluate (done, awaiting 'proceed' for Phase 7)

`src/08_evaluate.py` uses only saved predictions (`results/<model>/<split>/test_predictions.csv`) and saved checkpoints. Nothing was retrained. Check built in: the accuracy and macro-F1 recomputed from the predictions match the training-run metrics exactly for all 6 runs, and the reloaded BiLSTM and DistilBERT checkpoints reproduce the saved prediction and confidence for all 6 word-importance examples. Added `scipy==1.18.1` to `requirements.txt`.

**Per model and split** (`results/eval/<model>/<split>/`, all 6 combinations): `metrics.json` (accuracy, macro-F1, weighted-F1, per-class precision/recall/F1, top confused pairs), `per_class.csv`, `confusion_matrix.csv`, `confusion_counts.png`, `confusion_normalized.png`, `error_examples.csv` (15 random misclassified, seed 42, first 300 characters, with make/model/year, confidence, is_transcript), `correct_examples.csv` (5 random correct with confidence >= 0.95). Also `results/eval/per_class_f1_all.csv` (per-class F1, every model x split) and `results/eval/errors_all_models.csv` (15 sampled of the 497 temporal test complaints, 6.8%, that all three models got wrong).

### Summary table (`results/summary_table.csv`, `.md`, `.json`)

| Model | Split | Val acc | Val macro-F1 | Test acc | Test macro-F1 (95% CI) | Train time |
|---|---|---|---|---|---|---|
| MLP (TF-IDF) | temporal | 0.8698 | 0.8727 | 0.8762 | 0.8771 (0.8687 to 0.8843) | 24 s |
| MLP (TF-IDF) | random | 0.8864 | 0.8874 | 0.8851 | 0.8860 (0.8786 to 0.8929) | 21 s |
| BiLSTM | temporal | 0.8648 | 0.8683 | 0.8687 | 0.8690 (0.8604 to 0.8764) | 34 min |
| BiLSTM | random | 0.8852 | 0.8860 | 0.8835 | 0.8840 (0.8772 to 0.8908) | 30 min |
| DistilBERT | temporal | 0.8901 | 0.8941 | 0.8913 | 0.8921 (0.8849 to 0.8988) | 32 min |
| DistilBERT | random | 0.9023 | 0.9029 | 0.9060 | 0.9066 (0.9000 to 0.9128) | 43 min |

### Significance (`results/significance.json`; 1,000 bootstrap resamples of the test texts, seed 42, same resamples for all models within a split)

Paired bootstrap difference in test macro-F1 (95% CI):

| Comparison | Temporal | Random |
|---|---|---|
| BiLSTM minus MLP | -0.0081 (-0.0146 to -0.0016) | -0.0020 (-0.0076 to +0.0037), includes 0 |
| DistilBERT minus MLP | +0.0150 (+0.0093 to +0.0215) | +0.0206 (+0.0148 to +0.0263) |
| DistilBERT minus BiLSTM | +0.0231 (+0.0173 to +0.0296) | +0.0226 (+0.0169 to +0.0281) |

McNemar's test (exact, two-sided), temporal test, on paired per-text correctness:

| Pair | Only first right | Only second right | p-value |
|---|---|---|---|
| MLP vs BiLSTM | 325 | 270 | 0.027 |
| MLP vs DistilBERT | 209 | 319 | 1.9e-06 |
| BiLSTM vs DistilBERT | 189 | 354 | 1.3e-12 |

Reading: DistilBERT is clearly better than both other models on both splits (CIs exclude 0; p < 1e-5). MLP vs BiLSTM is a weak result: on temporal the BiLSTM is slightly worse (p = 0.027 would not survive a Bonferroni correction for 3 tests, threshold 0.017; the bootstrap CI barely excludes 0), and on random there is no difference. The bootstrap CIs for single models overlap between MLP and BiLSTM, so those two should be described as roughly equal, with the MLP slightly ahead on temporal.

### Transcript vs owner-written, temporal test (`results/transcript_vs_owner.csv`, `.json`)
1,596 transcripts (21.8%) and 5,724 owner-written texts. Macro-F1 (95% bootstrap CI within group):

| Model | Owner-written | Transcript | Transcript, classes with 30+ texts only (9 classes) |
|---|---|---|---|
| MLP | 0.8770 (0.8688 to 0.8853) | 0.8574 (0.8229 to 0.8827) | 0.8786 |
| BiLSTM | 0.8679 (0.8590 to 0.8762) | 0.8598 (0.8259 to 0.8836) | 0.8734 |
| DistilBERT | 0.8937 (0.8859 to 0.9014) | 0.8633 (0.8273 to 0.8882) | 0.8885 |

Accuracy is similar or higher on transcripts (MLP 0.881, BiLSTM 0.878, DistilBERT 0.890 vs 0.875, 0.866, 0.892 owner-written). The lower transcript macro-F1 is a small-class artifact: only 7 transcripts are VISIBILITY/WIPER (412 owner-written), so that class's F1 is very noisy. Without it, transcript macro-F1 is within the owner-written range for every model. Conclusion: no evidence that transcripts are easier or harder; a template shortcut does not appear to be driving the scores.

### Word importance (`results/eval/word_importance/`)
Occlusion on CPU: remove one whitespace-separated word, importance = drop in the log-odds of the predicted class (log-odds, because probability-based occlusion saturates at 0 for confident predictions). 6 temporal-test complaints (25 to 90 words, seed 42): 2 right for both models, 2 wrong for both, 2 right for DistilBERT but wrong for BiLSTM. Files: `example<N>_<model>.html` (small snippets for the website; blue = supports the predicted class, orange = argues against; hover shows the value), `preview.html` (all 12 on one page), `word_importance.json` (all values and the selection rule).
- Example 4 (true ELECTRICAL SYSTEM, both models predict FORWARD COLLISION AVOIDANCE with 95% and 99% confidence): the word "autopilot" drives it (log-odds drop 6.3 for BiLSTM, 4.0 for DistilBERT); the BiLSTM also leans on "tesla" (2.5). A clear case of a keyword pulling a complaint into a plausible but different class.
- Example 6 (true ENGINE, BiLSTM predicts POWER TRAIN): "transmission" (4.6) and "first" (2.2) drive the BiLSTM; DistilBERT, correct, relies on "hoses", "engine".
- Example 3 (true SERVICE BRAKES, predicted POWER TRAIN by BiLSTM and STEERING by DistilBERT): "fishtailed" for BiLSTM; "barrier." and "language" for DistilBERT; the wrong words are accident wording, not brake wording.
- Caveats for the website: single-word occlusion underestimates words that are repeated or have close substitutes (in confident correct examples the DistilBERT importances are small because several words each carry the same evidence); the examples are illustrations, not proof.

### Weak classes (all models, test F1; full table in `results/eval/per_class_f1_all.csv`)
ELECTRICAL SYSTEM is the weakest class for every model on both splits (temporal MLP 0.771, BiLSTM 0.757, DistilBERT 0.801). Next are FUEL/PROPULSION SYSTEM and ENGINE. The pair ENGINE / FUEL/PROPULSION SYSTEM is the top or near-top confusion for all models; STRUCTURE predicted as VISIBILITY/WIPER is the other recurring one. 497 of 7,320 temporal test complaints (6.8%) are wrong for all three models, so these errors are not model-specific; many are plausibly ambiguous or multi-system complaints (see `errors_all_models.csv`).

**Next:** Phase 7 (figures: architecture diagrams, pipeline diagram, before/after cleaning, class balance, copy to `docs/img/`). Waiting for "proceed".
