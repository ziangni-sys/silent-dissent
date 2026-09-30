#!/usr/bin/env bash
# Exploratory lens refit on a model's dev split: fit a J-lens, then E0 on the dev split again with
# every lens (the refit is picked up via entity.extra_lenses of the entity config).
#
#   C=configs/llama31_8b_it_entity.yaml L=configs/llama31_8b_it_jlens_wiki100.yaml \
#     bash scripts/entity_refit.sh [--after LOG:TEXT] [--stop-pod]
#
# --after LOG:TEXT  first wait until the file LOG contains TEXT (e.g. another queue's end), then
#                   until no other GPU job runs, so the fit gets the whole GPU
# Steps: fit_jlens with L (skipped if its lens file exists), then entity_run --stage e0 --split dev
# --formats plain --tag refit (-> e0_dev_refit). Nothing touches the test split.
# --stop-pod: afterwards stop the pod unless, within GRACE seconds (default 3600), another GPU job
#             starts or results/KEEP_POD_REFIT appears.
# Progress: results/refit_<config>.log; each step: results/refit_<config>_<step>.log.
set -u
cd "$(dirname "$0")/.."
C=${C:?set C to the entity config}
L=${L:?set L to the lens-fit config}
NAME=$(basename "$C" .yaml)
LOG=results/refit_$NAME.log
mkdir -p results lenses
GPU_JOBS="python scripts/(entity_|fit_jlens|check_model|lens_benchmark)"

log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }
step() {
  local name=$1; shift
  local t0=$SECONDS
  log "start $name: $*"
  if "$@" >"results/refit_${NAME}_$name.log" 2>&1; then
    log "done  $name ($((SECONDS - t0)) s)"
  else
    log "FAILED $name (exit $?, $((SECONDS - t0)) s), see results/refit_${NAME}_$name.log"
    return 1
  fi
}

args=("$@")
for i in "${!args[@]}"; do
  if [ "${args[$i]}" = "--after" ]; then
    spec=${args[$((i + 1))]}
    log "waiting until ${spec%%:*} contains '${spec#*:}'"
    until grep -qF "${spec#*:}" "${spec%%:*}" 2>/dev/null; do sleep 30; done
    while pgrep -f "$GPU_JOBS" >/dev/null; do sleep 30; done
    log "GPU free"
  fi
done

LENS=$(python -c "from silent_dissent.config import load_config; print(load_config('$L')['lens']['kwargs']['path'])")
if [ -s "$LENS" ]; then
  log "$LENS exists: fit skipped"
else
  step fit python scripts/fit_jlens.py --config "$L"
fi
step e0_dev python scripts/entity_run.py --config "$C" --stage e0 --split dev --formats plain --tag refit
log "refit queue finished"

for a in "$@"; do
  if [ "$a" = "--stop-pod" ]; then
    log "stopping the pod in ${GRACE:-3600} s unless another GPU job starts or results/KEEP_POD_REFIT appears"
    for _ in $(seq 1 $(( ${GRACE:-3600} / 30 ))); do
      [ -e results/KEEP_POD_REFIT ] && { log "results/KEEP_POD_REFIT found: pod left running"; exit 0; }
      pgrep -f "$GPU_JOBS" >/dev/null && { log "another GPU job is running: pod left running"; exit 0; }
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
