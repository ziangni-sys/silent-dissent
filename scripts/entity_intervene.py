"""Entity study E4: is the latent premise causally upstream of the answer?

Round 0 and round 1 of one pressure cell (entity.intervention.style, .n_peers). The round-1
prompt is then re-run with a lens direction added at the last token of one layer
(h += alpha * ||h|| * d; entity.intervention.layers, default the prereg band or, with
entity.intervention.band_layers = n, n evenly spaced layers of it; each layer separately;
entity.intervention.alphas):
  agents that gave in (answered e3'):   orig_bridge, orig_answer, ctrl_bridge, random
  agents that held (answered e3):       peer_bridge, peer_answer, ctrl_bridge, random
The outcome is the answer's first token (original / peer / control / other) and its ranks. If
injecting the *bridge* moves the answer, the premise the lens reads is used downstream.

Writes <out_dir>/entity/inject_<split>/records.jsonl.

    python scripts/entity_intervene.py --config configs/qwen35_4b_entity.yaml --split test
"""
import argparse
import json
from pathlib import Path

from silent_dissent.config import load_config, setup
from silent_dissent import entity_metrics as EM
from silent_dissent.entity_experiments import classify, generate, run_injection
from silent_dissent.entity_io import entity_dir, lens_set, load_items, prereg_categories, states_after_round0
from silent_dissent.entity_prompts import EntityState

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--split", choices=["dev", "test"], required=True)
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--direction-lens", help="lens whose token directions are injected (default: the configured lens)")
args = ap.parse_args()

cfg = load_config(args.config)
ec, ic = cfg["entity"], cfg["entity"]["intervention"]
prereg = Path(cfg["entity_prereg"])
pr = json.loads(prereg.read_text()) if prereg.exists() else {}
fmt = pr.get("format", ec["formats"][0])
lm, lens = setup(cfg)
gen_bs = ec.get("gen_batch_size", cfg["batch_size"])
band, n = pr.get("band"), ic.get("band_layers")
if band and n:  # n evenly spaced layers of the prereg band, its first and last included
    band = sorted({band[int(i * (len(band) - 1) / max(n - 1, 1) + 0.5)] for i in range(n)})
layers = ic.get("layers") or band or [lm.n_layers // 2]
lenses = lens_set(cfg, lm.model, lens)
dlens = lenses[args.direction_lens or cfg["lens"]["name"]]
items = load_items(cfg, args.split, args.limit or ic.get("limit"), prereg_categories(cfg))

solo = [EntityState(f, p, c, "solo").render(fmt, lm.tok) for f, p, c, _ in items]
answers = generate(lm, solo, ec.get("gen_tokens", 12), gen_bs, desc="round 0")
states, meta, roles = states_after_round0(lm.tok, items, answers,
                                          [{"condition": "pressure", "style": ic["style"]}], [ic["n_peers"]],
                                          require_correct=ec.get("require_correct_round0", True))
for st in states:
    st.advance()
round1 = generate(lm, [st.render(fmt, lm.tok) for st in states], ec.get("gen_tokens", 12), gen_bs,
                  desc="round 1")
for m, st, a in zip(meta, states, round1):
    m.update({"split": args.split, "format": fmt, "round": 1, "answer": a, "outcome": classify(a, st.fact, st.peer)})

records = []
for outcome, kinds in (("peer", ic["kinds_flipped"]), ("original", ic["kinds_held"])):
    idx = [i for i, m in enumerate(meta) if m["outcome"] == outcome
           or (ic.get("debug_any_outcome") and outcome == "peer")]  # debug configs: a random model never flips
    print(f"{outcome}: {len(idx)} agents, kinds {kinds}, layers {layers}, alphas {ic['alphas']}")
    if idx:
        records += run_injection(lm, dlens, [states[i] for i in idx], [meta[i] for i in idx], [roles[i] for i in idx],
                                 fmt, layers, ic["alphas"], kinds, cfg["batch_size"], cfg["seed"])
out = entity_dir(cfg) / f"inject_{args.split}"
out.mkdir(parents=True, exist_ok=True)
with open(out / "records.jsonl", "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
if records:
    print(EM.injection_table(records).to_string(index=False))
print(f"-> {out} ({len(records)} records)")
