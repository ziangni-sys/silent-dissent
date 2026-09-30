"""Causal test: inject letter directions at the pre-registered layer."""
import argparse
import json
import random

from silent_dissent.config import get_items, load_config, load_prereg, setup
from silent_dissent.experiments import read_jsonl, run_intervention, write_jsonl

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--grid", default="main", help="pressure file to draw source records from")
ap.add_argument("--limit", type=int)
ap.add_argument("--rounds", type=int, nargs="+", help="override intervention.rounds")
ap.add_argument("--alphas", type=float, nargs="+", help="override intervention.alphas")
ap.add_argument("--tag", help="write intervention_<tag>.jsonl / intervention_baseline_<tag>.jsonl")
args = ap.parse_args()

cfg = load_config(args.config)
icfg = dict(cfg["intervention"])
if args.rounds:
    icfg["rounds"] = args.rounds
if args.alphas:
    icfg["alphas"] = args.alphas
suffix = f"_{args.tag}" if args.tag else ""
layer = icfg["layer"] if icfg["layer"] is not None else load_prereg(cfg)["layer"]
lm, lens = setup(cfg)
items = {it.item_id: it for it in get_items(cfg)}
reasons_path = cfg["out_dir"] / "reasons.json"
reasons = json.loads(reasons_path.read_text()) if reasons_path.exists() else None

src = read_jsonl(cfg["out_dir"] / f"pressure_{args.grid}.jsonl")
src = [r for r in src if r["round"] in icfg["rounds"] and all(r[k] == v for k, v in icfg["source"].items())]
src = src[: args.limit]
kw = dict(layer=layer, alphas=icfg["alphas"], kinds=icfg["kinds"], batch_size=cfg["batch_size"],
          seed=cfg["seed"], contrastive=icfg["contrastive"], reasons=reasons)
print(f"layer {layer}: {len(src)} pressure records, rounds {icfg['rounds']}, alphas {icfg['alphas']}")
write_jsonl(run_intervention(lm, lens, items, src, **kw), cfg["out_dir"] / f"intervention{suffix}.jsonl")

# Unpressured check: does injection alone move answers on round-0 prompts?
base = read_jsonl(cfg["out_dir"] / "baseline.jsonl")
base = random.Random(cfg["seed"]).sample(base, min(icfg["baseline_check"], len(base)))
print(f"baseline check: {len(base)} items")
write_jsonl(run_intervention(lm, lens, items, base, **kw), cfg["out_dir"] / f"intervention_baseline{suffix}.jsonl")
