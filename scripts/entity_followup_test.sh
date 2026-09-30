#!/usr/bin/env bash
# Follow-up test queue: runs the test split for every addendum that is committed as is.
#
#   bash scripts/entity_followup_test.sh [--stop-pod]
#
# For each part (rounds, bridges) whose prereg/<name>_entity_<part>.json exists: refuse unless the
# file is committed unchanged, then entity_run.py --stage pressure --split test --addendum <part>
# (-> pressure_test_<part>). Then entity_analyze (verdicts in analysis/hypotheses_<part>.json).
# --stop-pod: stop the pod after GRACE seconds (default 1800) unless results/KEEP_POD_FOLLOWUP exists.
# Progress: results/entity_followup_test.log; each step: results/entity_followup_test_<step>.log.
set -u
cd "$(dirname "$0")/.."
C=${C:-configs/qwen35_4b_entity.yaml}
LOG=results/entity_followup_test.log
mkdir -p results

log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }
step() {
  local name=$1; shift
  local t0=$SECONDS
  log "start $name: $*"
  if "$@" >"results/entity_followup_test_$name.log" 2>&1; then
    log "done  $name ($((SECONDS - t0)) s)"
  else
    log "FAILED $name (exit $?, $((SECONDS - t0)) s), see results/entity_followup_test_$name.log"
    return 1
  fi
}

ran=0
for part in rounds bridges; do
  A=$(python -c "from silent_dissent.config import load_config; from silent_dissent.entity_io import addendum_path; print(addendum_path(load_config('$C'), '$part'))")
  if [ ! -f "$A" ]; then
    log "no $A: $part skipped"
    continue
  fi
  if ! git ls-files --error-unmatch "$A" >/dev/null 2>&1 || ! git diff --quiet HEAD -- "$A"; then
    log "$A is not committed as is: $part refused"
    continue
  fi
  log "$part: $A at $(git rev-parse --short HEAD)"
  step "pressure_$part" python scripts/entity_run.py --config "$C" --stage pressure --split test --addendum "$part" \
    --no-extra-lenses && ran=1
done
[ "$ran" = 1 ] && step analyze python scripts/entity_analyze.py --config "$C"
log "follow-up test queue finished"

for a in "$@"; do
  if [ "$a" = "--stop-pod" ]; then
    log "stopping the pod in ${GRACE:-1800} s unless results/KEEP_POD_FOLLOWUP appears"
    for _ in $(seq 1 $(( ${GRACE:-1800} / 30 ))); do
      [ -e results/KEEP_POD_FOLLOWUP ] && { log "results/KEEP_POD_FOLLOWUP found: pod left running"; exit 0; }
      sleep 30
    done
    if command -v runpodctl >/dev/null && [ -n "${RUNPOD_POD_ID:-}" ]; then
      log "stopping pod $RUNPOD_POD_ID"
      runpodctl stop pod "$RUNPOD_POD_ID" 2>&1 | tee -a "$LOG"
    else
      log "cannot stop pod: runpodctl or RUNPOD_POD_ID missing; stop it from the console"
    fi
  fi
done
