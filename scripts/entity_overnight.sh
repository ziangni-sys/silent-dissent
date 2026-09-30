#!/usr/bin/env bash
# Unattended queue for the entity study, run once the entity prereg is committed.
#
#   bash scripts/entity_overnight.sh              # leave the pod running at the end
#   bash scripts/entity_overnight.sh --stop-pod   # stop it after a grace period (/workspace survives)
#
# 1. E6: fitting corpora + the chat and mixed J-lenses (skipped if the lens files exist), so every
#    later readout also carries them
# 2. test split: E1/E2 pressure (PLIMIT items if set), E3 timeline, E4 injection, E5 free debate
# 3. lens benchmark with the refit lenses, then entity_analyze
# 4. --stop-pod: wait GRACE seconds (default 1800) for someone to read the results; touching
#    results/KEEP_POD during that time keeps the pod running
# Each step logs to results/entity_overnight_<step>.log; progress goes to results/entity_overnight.log.
# A failing step is logged and the queue moves on.
set -u
cd "$(dirname "$0")/.."
C=${C:-configs/qwen35_4b_entity.yaml}
LOG=results/entity_overnight.log
mkdir -p results lenses

log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }
step() {
  local name=$1; shift
  local t0=$SECONDS
  log "start $name: $*"
  if "$@" >"results/entity_overnight_$name.log" 2>&1; then
    log "done  $name ($((SECONDS - t0)) s)"
  else
    log "FAILED $name (exit $?, $((SECONDS - t0)) s), see results/entity_overnight_$name.log"
    return 1
  fi
}

PREREG=$(python -c "from silent_dissent.config import load_config; print(load_config('$C')['entity_prereg'])")
if ! git ls-files --error-unmatch "$PREREG" >/dev/null 2>&1 || ! git diff --quiet HEAD -- "$PREREG"; then
  log "$PREREG is not committed as is: refusing to run the test split"
  exit 1
fi
log "queue started ($C, prereg $PREREG at $(git rev-parse --short HEAD))"

# ---- E6: refit J-lenses
if [ ! -s lenses/qwen35_4b_jlens_chat.pt ] || [ ! -s lenses/qwen35_4b_jlens_mixed.pt ]; then
  step lens_corpus python scripts/build_lens_corpus.py --config configs/qwen35_4b_jlens_chat.yaml --n 2000
  [ -s lenses/qwen35_4b_jlens_chat.pt ] || step fit_chat python scripts/fit_jlens.py --config configs/qwen35_4b_jlens_chat.yaml
  [ -s lenses/qwen35_4b_jlens_mixed.pt ] || step fit_mixed python scripts/fit_jlens.py --config configs/qwen35_4b_jlens_mixed.yaml
fi

# ---- test split
step pressure python scripts/entity_run.py --config "$C" --stage pressure --split test ${PLIMIT:+--limit $PLIMIT}
step timeline python scripts/entity_timeline.py --config "$C" --split test
step intervene python scripts/entity_intervene.py --config "$C" --split test
step debate python scripts/entity_free_debate.py --config "$C" --split test

# ---- E6 text benchmark with the refit lenses, then the analysis
extra=()
for n in chat mixed; do
  [ -s "lenses/qwen35_4b_jlens_$n.pt" ] && extra+=(--extra-lens "jlens_$n=lenses/qwen35_4b_jlens_$n.pt")
done
step benchmark python scripts/lens_benchmark.py --config configs/qwen35_4b.yaml --out results/qwen35_4b_entity/lens_benchmark ${extra[@]+"${extra[@]}"}
step analyze python scripts/entity_analyze.py --config "$C"
log "queue finished"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | sed 's/^/GPU now: /' | tee -a "$LOG"

if [ "${1:-}" = "--stop-pod" ]; then
  log "stopping the pod in ${GRACE:-1800} s unless results/KEEP_POD appears"
  for _ in $(seq 1 $(( ${GRACE:-1800} / 30 ))); do
    [ -e results/KEEP_POD ] && { log "results/KEEP_POD found: pod left running"; exit 0; }
    sleep 30
  done
  if command -v runpodctl >/dev/null && [ -n "${RUNPOD_POD_ID:-}" ]; then
    log "stopping pod $RUNPOD_POD_ID"
    runpodctl stop pod "$RUNPOD_POD_ID" 2>&1 | tee -a "$LOG"
  else
    log "cannot stop pod: runpodctl or RUNPOD_POD_ID missing; stop it from the console"
  fi
fi
