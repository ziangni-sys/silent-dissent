#!/usr/bin/env bash
# Follow-up dev queue (dev split only; nothing here touches test data).
#
#   bash scripts/entity_followup_dev.sh
#
# 1. later rounds: pressure_dev_rounds_repeat (the prereg protocol, reference) and one run per
#    candidate in entity.addenda.rounds.variants, pilot_limit items, n_peers peers
# 2. bridges (only if entity.addenda.bridges is set): bridgeprobe_dev (where can the lens read which
#    bridge type), then e0_dev_bridges_<mode> per probe mode (E0 conditions on the non-place bridge
#    items, extended readout)
# 3. entity_addendum.py --part rounds / bridges --dry-run: prints what would be fixed. The addenda
#    themselves are written by hand after reading this (entity_addendum.py without --dry-run),
#    then committed and pushed before scripts/entity_followup_test.sh.
# Only the configured lens and the logit lens are read (--no-extra-lenses). Progress:
# results/entity_followup_dev.log; each step: results/entity_followup_dev_<step>.log. A failing
# step is logged and the queue moves on.
set -u
cd "$(dirname "$0")/.."
C=${C:-configs/qwen35_4b_entity.yaml}
LOG=results/entity_followup_dev.log
mkdir -p results

log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }
step() {
  local name=$1; shift
  local t0=$SECONDS
  log "start $name: $*"
  if "$@" >"results/entity_followup_dev_$name.log" 2>&1; then
    log "done  $name ($((SECONDS - t0)) s)"
  else
    log "FAILED $name (exit $?, $((SECONDS - t0)) s), see results/entity_followup_dev_$name.log"
    return 1
  fi
}
get() { python -c "from silent_dissent.config import load_config; import json; c = load_config('$C'); print($1)"; }

VARIANTS=$(get "' '.join(c['entity']['addenda']['rounds']['variants'])")
LIMIT=$(get "c['entity']['addenda']['rounds']['pilot_limit']")
NPEERS=$(get "c['entity']['addenda']['rounds']['n_peers']")
ROUNDS=$(get "c['entity']['addenda']['rounds']['round']")
BRIDGES=$(get "'bridges' in c['entity']['addenda']")
TYPES= MODES=
if [ "$BRIDGES" = True ]; then
  TYPES=$(get "' '.join(c['entity']['addenda']['bridges']['bridge_types'])")
  MODES=$(get "' '.join(c['entity']['addenda']['bridges']['probe_modes'])")
fi
FMT=$(get "json.load(open(c['entity_prereg']))['format']")
log "queue started ($C at $(git rev-parse --short HEAD)): variants [$VARIANTS], $LIMIT items, $NPEERS peers, $ROUNDS rounds; bridges $BRIDGES${TYPES:+ (types [$TYPES], probe modes [$MODES])}, format $FMT"

for v in repeat $VARIANTS; do
  step "rounds_$v" python scripts/entity_run.py --config "$C" --stage pressure --split dev --later "$v" \
    --n-peers "$NPEERS" --rounds "$ROUNDS" --limit "$LIMIT" --tag "rounds_$v" --no-extra-lenses
done
if [ "$BRIDGES" = True ]; then
  step bridgeprobe python scripts/entity_bridge_probe.py --config "$C" --split dev --no-extra-lenses
  for m in $MODES; do
    # shellcheck disable=SC2086
    step "e0_bridges_$m" python scripts/entity_run.py --config "$C" --stage e0 --split dev --formats "$FMT" \
      --bridge-types $TYPES --probe-mode "$m" --extended --tag "bridges_$m" --no-extra-lenses
  done
fi
step addendum_rounds python scripts/entity_addendum.py --config "$C" --part rounds --dry-run
[ "$BRIDGES" = True ] && step addendum_bridges python scripts/entity_addendum.py --config "$C" --part bridges --dry-run
log "dev queue finished; read results/entity_followup_dev_addendum_*.log, then write the addenda"
