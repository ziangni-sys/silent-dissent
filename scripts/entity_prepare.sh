#!/usr/bin/env bash
# Prepare a new model for the entity study: download it, run the pre-flight check, fit its J-lens.
#
#   C=configs/gemma4_e4b_it_entity.yaml L=configs/gemma4_e4b_it_jlens_wiki100.yaml \
#     PREFLIGHT_LENS=jlens_base bash scripts/entity_prepare.sh [--after LOG:TEXT]
#
# --after LOG:TEXT  first wait until the file LOG contains TEXT, then until no other GPU job runs
# 1. download the model (without Meta-style original/ checkpoints)
# 2. scripts/preflight_config.py + scripts/check_model.py; stops unless it ends with ALL CHECKS PASSED
#    (PREFLIGHT_LENS: an entity.extra_lenses entry used as the lens, since the fitted one may not exist yet)
# 3. scripts/fit_jlens.py with L, unless its lens file exists (skipped when L is unset)
# Exit status 0 only if every step passed, so the next queue can be chained with &&.
# Progress: results/prepare_<config>.log; each step: results/prepare_<config>_<step>.log.
set -u
cd "$(dirname "$0")/.."
C=${C:?set C to the entity config}
NAME=$(basename "$C" .yaml)
LOG=results/prepare_$NAME.log
mkdir -p results lenses
GPU_JOBS="python scripts/(entity_|fit_jlens|check_model|lens_benchmark)"

log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }
step() {
  local name=$1; shift
  local t0=$SECONDS
  log "start $name: $*"
  if "$@" >"results/prepare_${NAME}_$name.log" 2>&1; then
    log "done  $name ($((SECONDS - t0)) s)"
  else
    log "FAILED $name (exit $?, $((SECONDS - t0)) s), see results/prepare_${NAME}_$name.log"
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

MODEL=$(python -c "from silent_dissent.config import load_config; print(load_config('$C')['model']['name'])")
step download python -c "from huggingface_hub import snapshot_download; print(snapshot_download('$MODEL', ignore_patterns=['original/*']))" \
  || exit 1
PCFG=$(python scripts/preflight_config.py --config "$C" ${PREFLIGHT_LENS:+--extra-lens "$PREFLIGHT_LENS"}) || exit 1
step preflight python scripts/check_model.py --config "$PCFG" --limit 50 || exit 1
if ! grep -q "ALL CHECKS PASSED" "results/prepare_${NAME}_preflight.log"; then
  log "pre-flight check did not pass: see results/prepare_${NAME}_preflight.log"
  exit 1
fi
if [ -n "${L:-}" ]; then
  LENS=$(python -c "from silent_dissent.config import load_config; print(load_config('$L')['lens']['kwargs']['path'])")
  if [ -s "$LENS" ]; then
    log "$LENS exists: fit skipped"
  else
    step fit python scripts/fit_jlens.py --config "$L" || exit 1
  fi
fi
log "prepare finished"
