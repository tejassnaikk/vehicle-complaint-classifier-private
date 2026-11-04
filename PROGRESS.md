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

**Next:** Phase 4 (split and sample). Waiting for "proceed".
