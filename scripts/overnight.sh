#!/usr/bin/env bash
# Unattended queue for the Qwen3.5-4B run. Start it in its own tmux window:
#
#   bash scripts/overnight.sh              # leave the pod running at the end
#   bash scripts/overnight.sh --stop-pod   # stop the pod when done (/workspace survives)
#
# 1. waits until no experiment script is running (e.g. the current run_intervention);
# 2. run_debate (skipped if debate.jsonl exists) and analyze for the J-lens run;
# 3. logit-lens comparison: same items/reasons/prereg layer, lens swapped
#    (baseline + pressure + analyze into results/qwen35_4b_logit).
# Each step logs to results/overnight_<step>.log; progress goes to results/overnight.log.
# A failing step is logged and the queue moves on.
set -u
cd "$(dirname "$0")/.."
C=${C:-configs/qwen35_4b.yaml}
L=${L:-configs/qwen35_4b_logit.yaml}
R=${R:-results/qwen35_4b}
RL=${RL:-results/qwen35_4b_logit}
mkdir -p results

log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a results/overnight.log; }
# Match only python processes running an experiment script, not shells whose command line
# merely mentions one (e.g. a `while pgrep ...` waiter), which would block the queue forever.
busy() { pgrep -f "^([^ ]*/)?python[0-9.]* ([^ ]+ )*scripts/(run_baseline|run_pressure|run_intervention|run_debate|analyze|select_layer)\.py" >/dev/null; }
step() {
  local name=$1; shift
  local t0=$SECONDS
  log "start $name: $*"
  if "$@" >"results/overnight_$name.log" 2>&1; then
    log "done  $name ($((SECONDS - t0)) s)"
  else
    log "FAILED $name (exit $?, $((SECONDS - t0)) s), see results/overnight_$name.log"
    return 1
  fi
}

log "queue started; waiting for running experiment scripts to finish"
while busy; do sleep 60; done
sleep ${SETTLE:-30}  # in case another chained step was about to start
while busy; do sleep 60; done
log "GPU free"

# --- J-lens run: finish the pipeline --------------------------------------------------
[ -s $R/intervention.jsonl ] || log "WARNING: $R/intervention.jsonl is missing or empty"
if [ -s $R/debate.jsonl ]; then
  log "skip  debate ($R/debate.jsonl exists)"
else
  step debate python scripts/run_debate.py --config $C
fi
step analyze python scripts/analyze.py --config $C

# --- Logit-lens comparison ---------------------------------------------------------------
mkdir -p $RL
for f in items.jsonl reasons.json; do  # identical items and peer reasons, no regeneration
  [ -e $RL/$f ] || cp $R/$f $RL/$f
done
if step logit_baseline python scripts/run_baseline.py --config $L; then
  # The lens does not touch the forward pass, so stated answers must be identical.
  python - "$R" "$RL" <<'PY' 2>&1 | tee -a results/overnight.log
import json, sys
load = lambda p: {r["item_id"]: r["stated"] for r in map(json.loads, open(p))}
a, b = load(f"{sys.argv[1]}/baseline_all.jsonl"), load(f"{sys.argv[2]}/baseline_all.jsonl")
diff = sum(a.get(k) != v for k, v in b.items())
print(f"logit vs jlens baseline: {len(b)} items, {diff} stated answers differ"
      + ("" if diff == 0 else "  <-- UNEXPECTED, check before using the logit run"))
PY
  step logit_pressure python scripts/run_pressure.py --config $L && \
    step logit_analyze python scripts/analyze.py --config $L
fi

log "queue finished"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | sed 's/^/GPU now: /' | tee -a results/overnight.log

if [ "${1:-}" = "--stop-pod" ]; then
  if command -v runpodctl >/dev/null && [ -n "${RUNPOD_POD_ID:-}" ]; then
    log "stopping pod $RUNPOD_POD_ID"
    runpodctl stop pod "$RUNPOD_POD_ID" 2>&1 | tee -a results/overnight.log
  else
    log "cannot stop pod: runpodctl or RUNPOD_POD_ID missing; stop it from the console"
  fi
fi
