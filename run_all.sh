#!/usr/bin/env bash
# Re-run the whole pipeline from the cleaning step (03) through the website (10), then the checks.
#
# Launch it from the Terminal app (not from an editor), so closing the editor cannot stop it:
#   cd /Users/tena7850/Developer/vehicle-complaint-classifier
#   mkdir -p results/run_all && nohup caffeinate -ims ./run_all.sh > results/run_all/launch.out 2>&1 &
# Watch progress:   tail -f results/run_all/summary.log
# Each step logs to results/run_all/<step>.log. The script stops at the first failing step.
# It overwrites data/interim, data/clean, models/, results/ and docs/ with freshly generated files.
# Resume after a stop:  ./run_all.sh --from <step>   (skips every step before <step>; the summary log is appended to, not reset)
#   Step names are the first word after "step" below, e.g. 07_train_distilbert_temporal. Use --list to print them.
# Expected time on a MacBook Air M4: about 2.5 to 3 hours (BiLSTM about 65 min, DistilBERT about 75 to 90 min, the rest a few minutes).
set -euo pipefail
cd "$(dirname "$0")"

export TOKENIZERS_PARALLELISM=false
PY=venv/bin/python
LOG=results/run_all
mkdir -p "$LOG"
SUMMARY="$LOG/summary.log"

FROM=""
case "${1:-}" in
  --list) grep -E '^step ' "$0" | awk '{print $2}'; exit 0 ;;
  --from)
    FROM="${2:-}"
    if [ -z "$FROM" ] || ! grep -qE "^step +$FROM( |$)" "$0"; then
      echo "Unknown or missing step '$FROM'. Valid steps:" >&2; grep -E '^step ' "$0" | awk '{print "  " $2}' >&2; exit 2
    fi ;;
  "") ;;
  *) echo "Usage: $0 [--from <step> | --list]" >&2; exit 2 ;;
esac
if [ -z "$FROM" ]; then : > "$SUMMARY"; else echo "$(date '+%F %T') RESUME from $FROM" | tee -a "$SUMMARY"; fi

step() {
  local name="$1"; shift
  if [ -n "$FROM" ]; then
    if [ "$name" = "$FROM" ]; then FROM=""; else echo "$(date '+%F %T') SKIP   $name (before --from)" | tee -a "$SUMMARY"; return 0; fi
  fi
  local start; start=$(date +%s)
  echo "$(date '+%F %T') START  $name" | tee -a "$SUMMARY"
  if "$@" > "$LOG/$name.log" 2>&1; then
    echo "$(date '+%F %T') DONE   $name ($(( $(date +%s) - start )) s)" | tee -a "$SUMMARY"
  else
    local code=$?
    echo "$(date '+%F %T') FAILED $name (exit $code). See $LOG/$name.log" | tee -a "$SUMMARY"
    exit "$code"
  fi
}

# data: cleaning with the contact-detail masking rule (rule 14), then the splits
step 03_clean_and_label       $PY -u src/03_clean_and_label.py
step 04_split_and_sample      $PY -u src/04_split_and_sample.py
step privacy_scan_data        $PY -u src/privacy_scan.py

# models (temporal split is the main result; random split is the comparison)
step 05_train_mlp_temporal        $PY -u src/05_train_mlp.py --split temporal
step 05_train_mlp_random          $PY -u src/05_train_mlp.py --split random
step 06_train_bilstm_temporal     $PY -u src/06_train_bilstm.py --split temporal --device mps
step 06_train_bilstm_random       $PY -u src/06_train_bilstm.py --split random --device mps
step 07_train_distilbert_temporal $PY -u src/07_train_distilbert.py --full --split temporal
step 07_train_distilbert_random   $PY -u src/07_train_distilbert.py --full --split random

# evaluation, README table, figures, website
step 08_evaluate              $PY -u src/08_evaluate.py
step update_readme_table      $PY -u src/update_readme_table.py
step 09_make_figures          $PY -u src/09_make_figures.py
step 11_export_web            $PY -u src/11_export_web.py
step 10_build_site            $PY -u src/10_build_site.py

# site check in normal mode: the About-page placeholders are only a WARNING here. The pre-publish check is `python src/check_site.py --strict`.
step check_site               $PY -u src/check_site.py

echo "$(date '+%F %T') ALL STEPS DONE" | tee -a "$SUMMARY"
