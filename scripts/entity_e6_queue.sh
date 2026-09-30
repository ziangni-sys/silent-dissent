#!/usr/bin/env bash
# E6 queue: refit the J-lens on chat / mixed text, then compare every lens on the dev split.
#
#   bash scripts/entity_e6_queue.sh [--after-overnight] [--stop-pod]
#
# --after-overnight  first wait (at most 6 h) until results/entity_overnight.log says "queue finished",
#                    so the fits get the whole GPU (a fit needs ~42 GB next to a 36 GB readout run)
# Then: lens corpora (if missing), fit_jlens for the wiki100 (matched control), chat and mixed lenses (if missing), E0 on the dev
# split again with every lens (tag e6; the refit lenses are picked up via entity.extra_lenses), the
# text / two-hop benchmark with the refit lenses, entity_analyze.
# --stop-pod         stop the pod after GRACE seconds (default 1800) unless results/KEEP_POD_E6 exists.
# Progress: results/entity_e6.log; each step: results/entity_e6_<step>.log.
set -u
cd "$(dirname "$0")/.."
C=${C:-configs/qwen35_4b_entity.yaml}
LOG=results/entity_e6.log
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }
step() {
  local name=$1; shift
  local t0=$SECONDS
  log "start $name: $*"
  if "$@" >"results/entity_e6_$name.log" 2>&1; then
    log "done  $name ($((SECONDS - t0)) s)"
  else
    log "FAILED $name (exit $?, $((SECONDS - t0)) s), see results/entity_e6_$name.log"
    return 1
  fi
}

for a in "$@"; do
  if [ "$a" = "--after-overnight" ]; then
    log "waiting for the overnight queue to finish"
    for _ in $(seq 1 360); do
      grep -q "queue finished" results/entity_overnight.log 2>/dev/null && break
      sleep 60
    done
    while pgrep -f "python scripts/(entity_|lens_benchmark|fit_jlens)" >/dev/null; do sleep 30; done
    log "GPU free"
  fi
done

[ -s results/lens_corpus/qwen35_4b/chat.jsonl ] && [ -s results/lens_corpus/qwen35_4b/mixed.jsonl ] || \
  step lens_corpus python scripts/build_lens_corpus.py --config configs/qwen35_4b_jlens_chat.yaml --n 2000
[ -s lenses/qwen35_4b_jlens_wiki100.pt ] || step fit_wiki100 python scripts/fit_jlens.py --config configs/qwen35_4b_jlens_wiki100.yaml
[ -s lenses/qwen35_4b_jlens_chat.pt ] || step fit_chat python scripts/fit_jlens.py --config configs/qwen35_4b_jlens_chat.yaml
[ -s lenses/qwen35_4b_jlens_mixed.pt ] || step fit_mixed python scripts/fit_jlens.py --config configs/qwen35_4b_jlens_mixed.yaml

step e0_dev python scripts/entity_run.py --config "$C" --stage e0 --split dev --tag e6
extra=()
for n in wiki100 chat mixed; do
  [ -s "lenses/qwen35_4b_jlens_$n.pt" ] && extra+=(--extra-lens "jlens_$n=lenses/qwen35_4b_jlens_$n.pt")
done
step benchmark python scripts/lens_benchmark.py --config configs/qwen35_4b.yaml \
  --out results/qwen35_4b_entity/lens_benchmark_e6 ${extra[@]+"${extra[@]}"}
step analyze python scripts/entity_analyze.py --config "$C"
log "e6 queue finished"

for a in "$@"; do
  if [ "$a" = "--stop-pod" ]; then
    log "stopping the pod in ${GRACE:-1800} s unless results/KEEP_POD_E6 appears"
    for _ in $(seq 1 $(( ${GRACE:-1800} / 30 ))); do
      [ -e results/KEEP_POD_E6 ] && { log "results/KEEP_POD_E6 found: pod left running"; exit 0; }
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
