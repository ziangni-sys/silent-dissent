#!/usr/bin/env bash
# Replication queue: the entity study on another model, with every rule fixed on Qwen3.5-4B
# (configs/llama31_8b_it_entity.yaml, configs/qwen36_27b_entity.yaml).
#
#   C=configs/llama31_8b_it_entity.yaml bash scripts/entity_replicate.sh dev [--stop-pod]
#   C=configs/llama31_8b_it_entity.yaml bash scripts/entity_replicate.sh test [--stop-pod]
#   C=configs/llama31_8b_it_entity.yaml bash scripts/entity_replicate.sh rounds [--stop-pod]
#
# dev   dev split only: entity_build (which facts the model knows; dev / test split; skipped if the
#       item files exist), E0 on dev, entity_select (writes the entity prereg once, if a format
#       passes), the later-round pilots if entity.addenda.rounds is set (repeat = reference, then its variants) and
#       entity_addendum --part rounds (writes the rounds addendum once, if a variant passes).
#       Then read the logs, commit and push prereg/<name>_entity*.json.
# rounds  the later-round pilots and the rounds addendum alone (dev split; needs the prereg and no
#       addendum yet), e.g. after a pilot failed
# test refuses unless the prereg is committed unchanged (the rounds addendum likewise, or its run is
#       skipped): E1/E2 pressure, the rounds addendum run, E3 timeline, E4 injection, E5 free debate,
#       entity_analyze.
# --stop-pod: stop the pod after GRACE seconds (default 1800) unless results/KEEP_POD_REPLICATE exists.
# Progress: results/replicate_<config>.log; each step: results/replicate_<config>_<step>.log. In dev
# a failing step stops the queue (the next steps need its output); in test the queue moves on.
set -u
cd "$(dirname "$0")/.."
C=${C:?set C to a replication config}
PHASE=${1:?usage: C=<config> entity_replicate.sh dev|test|rounds [--stop-pod]}
NAME=$(basename "$C" .yaml)
LOG=results/replicate_$NAME.log
mkdir -p results
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}  # long later-round prompts

log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }
step() {
  local name=$1; shift
  local t0=$SECONDS
  log "start $name: $*"
  if "$@" >"results/replicate_${NAME}_$name.log" 2>&1; then
    log "done  $name ($((SECONDS - t0)) s)"
  else
    log "FAILED $name (exit $?, $((SECONDS - t0)) s), see results/replicate_${NAME}_$name.log"
    return 1
  fi
}
get() { python -c "from silent_dissent.config import load_config; c = load_config('$C'); print($1)"; }
committed() { git ls-files --error-unmatch "$1" >/dev/null 2>&1 && git diff --quiet HEAD -- "$1"; }
stop_pod() {
  log "stopping the pod in ${GRACE:-1800} s unless results/KEEP_POD_REPLICATE appears"
  for _ in $(seq 1 $(( ${GRACE:-1800} / 30 ))); do
    [ -e results/KEEP_POD_REPLICATE ] && { log "results/KEEP_POD_REPLICATE found: pod left running"; return; }
    sleep 30
  done
  if command -v runpodctl >/dev/null && [ -n "${RUNPOD_POD_ID:-}" ]; then
    log "stopping pod $RUNPOD_POD_ID"
    runpodctl stop pod "$RUNPOD_POD_ID" 2>&1 | tee -a "$LOG"
  else
    log "cannot stop pod: runpodctl or RUNPOD_POD_ID missing; stop it from the console"
  fi
}

PREREG=$(get "c['entity_prereg']")
ADD=$(python -c "from silent_dissent.config import load_config; from silent_dissent.entity_io import addendum_path; print(addendum_path(load_config('$C'), 'rounds'))")

dev() {
  if [ -e "$PREREG" ]; then
    log "$PREREG exists: the dev phase has run"
    return 1
  fi
  log "dev phase ($C at $(git rev-parse --short HEAD))"
  if [ -s "$(get "c['out_dir']")/entity/items_test.jsonl" ]; then
    log "item files exist: entity_build skipped"
  else
    step build python scripts/entity_build.py --config "$C" || return 1
  fi
  step e0_dev python scripts/entity_run.py --config "$C" --stage e0 --split dev || return 1
  step select python scripts/entity_select.py --config "$C" || { log "no prereg written: the replication stops at E0"; return 1; }
  if [ "$(get "'rounds' in (c['entity'].get('addenda') or {})")" = True ]; then
    rounds_pilots || return 1
  else
    log "no entity.addenda.rounds: later-round pilots skipped"
  fi
  log "dev phase finished: read the logs, then commit and push $PREREG and $ADD (if written)"
}

rounds_pilots() {
  if [ ! -f "$PREREG" ] || [ -e "$ADD" ]; then
    log "the rounds pilots need $PREREG and no $ADD yet"
    return 1
  fi
  local variants limit npeers rounds v
  variants=$(get "' '.join(c['entity']['addenda']['rounds']['variants'])")
  limit=$(get "c['entity']['addenda']['rounds']['pilot_limit']")
  npeers=$(get "c['entity']['addenda']['rounds']['n_peers']")
  rounds=$(get "c['entity']['addenda']['rounds']['round']")
  for v in repeat $variants; do
    step "rounds_$v" python scripts/entity_run.py --config "$C" --stage pressure --split dev --later "$v" \
      --n-peers "$npeers" --rounds "$rounds" --limit "$limit" --tag "rounds_$v" --no-extra-lenses || return 1
  done
  step addendum_rounds python scripts/entity_addendum.py --config "$C" --part rounds \
    || log "no rounds addendum: rounds 2-3 will not be tested"
}

test_split() {
  if ! committed "$PREREG"; then
    log "$PREREG is not committed as is: refusing to run the test split"
    return 1
  fi
  log "test phase ($C, prereg at $(git rev-parse --short HEAD))"
  step pressure python scripts/entity_run.py --config "$C" --stage pressure --split test
  if [ ! -f "$ADD" ]; then
    log "no $ADD: rounds 2-3 not run"
  elif committed "$ADD"; then
    step pressure_rounds python scripts/entity_run.py --config "$C" --stage pressure --split test --addendum rounds \
      --no-extra-lenses
  else
    log "$ADD is not committed as is: rounds refused"
  fi
  step timeline python scripts/entity_timeline.py --config "$C" --split test
  step intervene python scripts/entity_intervene.py --config "$C" --split test
  step debate python scripts/entity_free_debate.py --config "$C" --split test
  step analyze python scripts/entity_analyze.py --config "$C"
  log "test phase finished"
}

case "$PHASE" in
  dev) dev ;;
  test) test_split ;;
  rounds) rounds_pilots && log "rounds pilots finished: read the logs, then commit and push $ADD (if written)" ;;
  *) echo "unknown phase $PHASE (dev, test or rounds)"; exit 2 ;;
esac
status=$?
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | sed 's/^/GPU now: /' | tee -a "$LOG"
for a in "${@:2}"; do
  [ "$a" = "--stop-pod" ] && stop_pod
done
exit $status
