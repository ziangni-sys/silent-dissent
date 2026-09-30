#!/usr/bin/env bash
# Addendum "own": with the agent's round-0 answer hidden or absent in round 1, is the original
# bridge still read? (entity.addenda.own of the config; scripts/entity_addendum.py --part own)
#
#   C=configs/qwen35_4b_entity.yaml bash scripts/entity_own.sh dev
#   ... read the log, then commit and push prereg/<name>_entity_own.json ...
#   C=configs/qwen35_4b_entity.yaml bash scripts/entity_own.sh test
#   C=configs/qwen35_4b_entity.yaml bash scripts/entity_own.sh analyze
#
# dev      refuses unless the main prereg and the config are committed unchanged. Skips (exit 0) if the
#          addendum exists. Runs the two pilots on the dev split (entity_run --own hidden|absent: the
#          declared cells, n_peers, one round, pilot_limit items), each skipped if its directory is
#          complete (info.json), then entity_addendum --part own, which writes the addendum once if the
#          hidden variant passes. If it does not ("NO ADDENDUM", exit code 3 of entity_addendum), that
#          is logged and the phase ends with exit 0: the model has no own addendum, nothing to retry.
# test     skips (exit 0) if the model has no addendum; refuses unless the addendum is committed
#          unchanged and pushed; runs entity_run --addendum own on the test split
#          (-> pressure_test_own), skipped if that directory is complete. GPU.
# analyze  entity_analyze --addendum own: reads pressure_test_own only and writes only its own files
#          (analysis/pressure_test_own_*, hypotheses_own.json, own_summary.json). CPU; safe to rerun.
# A directory that exists without info.json (an interrupted run) is never overwritten: the phase
# refuses and the directory is left for the user. Progress: results/own_<config>.log; each step:
# results/own_<config>_<step>.log. A failing step stops the phase (exit 1).
set -u
cd "$(dirname "$0")/.."
C=${C:?set C to an entity config}
PHASE=${1:?usage: C=<config> entity_own.sh dev|test|analyze}
NAME=$(basename "$C" .yaml)
LOG=results/own_$NAME.log
mkdir -p results
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }
run_step() {  # returns the step's exit code
  local name=$1; shift
  local t0=$SECONDS rc
  log "start $name: $*"
  "$@" >"results/own_${NAME}_$name.log" 2>&1
  rc=$?
  log "end   $name (exit $rc, $((SECONDS - t0)) s), log results/own_${NAME}_$name.log"
  return $rc
}
step() { run_step "$@" || { log "FAILED $1"; tail -n 20 "results/own_${NAME}_$1.log" | tee -a "$LOG"; exit 1; }; }
py() { python -c "from silent_dissent.config import load_config; from silent_dissent.entity_io import *; c = load_config('$C'); print($1)"; }
committed() { git ls-files --error-unmatch "$1" >/dev/null 2>&1 && git diff --quiet HEAD -- "$1"; }
pushed() { git fetch -q origin && git cat-file -e "@{upstream}:$1" 2>/dev/null && git diff --quiet "@{upstream}" -- "$1"; }
# 0: complete run (info.json is written last), 1: absent, 2: exists but incomplete
run_state() { [ -e "$1/info.json" ] && return 0; [ -e "$1" ] && return 2; return 1; }

PREREG=$(py "c['entity_prereg']") || { log "FAILED: cannot read $C"; exit 1; }
ADD=$(py "addendum_path(c, 'own')")
ROOT=$(py "entity_dir(c)")

case "$PHASE" in
dev)
  committed "$PREREG" || { log "REFUSED: $PREREG is not committed unchanged"; exit 1; }
  committed "$C" || { log "REFUSED: $C (with entity.addenda.own) is not committed unchanged"; exit 1; }
  [ -e "$ADD" ] && { log "SKIP dev: $ADD exists (an addendum is fixed once)"; exit 0; }
  for v in hidden absent; do
    d="$ROOT/pressure_dev_own_$v"
    run_state "$d"; s=$?
    if [ $s -eq 0 ]; then log "skip pilot_$v: $d is complete"
    elif [ $s -eq 2 ]; then log "REFUSED: $d exists without info.json (interrupted run; not overwritten, report it)"; exit 1
    else step "pilot_$v" python scripts/entity_run.py --config "$C" --stage pressure --split dev --own "$v" --no-extra-lenses
    fi
  done
  run_step addendum python scripts/entity_addendum.py --config "$C" --part own; rc=$?
  tee -a "$LOG" <"results/own_${NAME}_addendum.log"
  case $rc in
  0) log "next: review $ADD, then: git add $ADD && git commit && git push; then: C=$C bash scripts/entity_own.sh test" ;;
  3) log "NO ADDENDUM for $NAME: the hidden variant fails its dev checks (a result to report; do not retry)" ;;
  *) log "FAILED addendum"; exit 1 ;;
  esac
  ;;
test)
  [ -e "$ADD" ] || { log "SKIP test: no addendum $ADD"; exit 0; }
  committed "$ADD" || { log "REFUSED: $ADD is not committed unchanged"; exit 1; }
  pushed "$ADD" || { log "REFUSED: $ADD is not pushed (git push first)"; exit 1; }
  d="$ROOT/pressure_test_own"
  run_state "$d"; s=$?
  if [ $s -eq 0 ]; then log "skip test: $d is complete"
  elif [ $s -eq 2 ]; then log "REFUSED: $d exists without info.json (interrupted run; not overwritten, report it)"; exit 1
  else step test python scripts/entity_run.py --config "$C" --stage pressure --split test --addendum own
  fi
  log "next: C=$C bash scripts/entity_own.sh analyze"
  ;;
analyze)
  [ -e "$ADD" ] || { log "SKIP analyze: no addendum $ADD"; exit 0; }
  run_state "$ROOT/pressure_test_own" || { log "REFUSED: $ROOT/pressure_test_own is not complete (run the test phase)"; exit 1; }
  step analyze python scripts/entity_analyze.py --config "$C" --addendum own
  grep -A40 "addendum own" "results/own_${NAME}_analyze.log" | tee -a "$LOG"
  ;;
*)
  echo "unknown phase $PHASE (dev | test | analyze)"; exit 2 ;;
esac
