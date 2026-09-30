"""Real N-agent debate (same model), for the collective-decision experiment."""
import argparse
import random

from silent_dissent.config import get_items, load_config, setup
from silent_dissent.debate import run_debate
from silent_dissent.experiments import read_jsonl, write_jsonl

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
args = ap.parse_args()

cfg = load_config(args.config)
dcfg = cfg["debate"]
lm, lens = setup(cfg)
items = {it.item_id: it for it in get_items(cfg)}
ids = [r["item_id"] for r in read_jsonl(cfg["out_dir"] / "baseline.jsonl")]
ids = random.Random(cfg["seed"]).sample(ids, min(dcfg["n_items"], len(ids)))
records = run_debate(lm, lens, [items[i] for i in ids], dcfg["n_agents"], dcfg["rounds"], dcfg["temperature"],
                     cfg["batch_size"], cfg["seed"])
write_jsonl(records, cfg["out_dir"] / "debate.jsonl")
