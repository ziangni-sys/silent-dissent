"""Entity study E5: free debate among copies of the model, with a latent vote.

n agents answer a two-hop statement (round 0 sampled at entity.debate.temperature so they can
disagree), then see each other's statements for entity.debate.rounds greedy rounds. For every
agent and round the lenses rank each candidate answer (the truth and the distinct round-0
answers) at the last token. Facts: entity.debate.n_items from facts_<split>.jsonl, a share
entity.debate.known_share of them known by the model (the rest are where debate can help or hurt).

Group accuracy per round compares three rules: plurality of stated answers, plurality of latent
choices (each agent's best-ranked candidate over the band), and plurality of round-0 answers.

Writes <out_dir>/entity/debate_<split>/records.jsonl.

    python scripts/entity_free_debate.py --config configs/qwen35_4b_entity.yaml --split test
"""
import argparse
import json
import random
from pathlib import Path

from silent_dissent import entity_metrics as EM
from silent_dissent.config import load_config, setup
from silent_dissent.entity_experiments import run_debate
from silent_dissent.entity_facts import Fact
from silent_dissent.entity_io import entity_dir, lens_set, prereg_categories

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--split", choices=["dev", "test"], required=True)
ap.add_argument("--n-items", type=int, default=None)
args = ap.parse_args()

cfg = load_config(args.config)
ec, dc = cfg["entity"], cfg["entity"]["debate"]
prereg = Path(cfg["entity_prereg"])
pr = json.loads(prereg.read_text()) if prereg.exists() else {}
fmt = pr.get("format", ec["formats"][0])
with open(entity_dir(cfg) / f"facts_{args.split}.jsonl", encoding="utf-8") as f:
    rows = [json.loads(l) for l in f]
cats = prereg_categories(cfg)
if cats is not None:
    rows = [r for r in rows if r["fact"]["category"] in cats]
rng = random.Random(cfg["seed"])
n = args.n_items or dc["n_items"]
known = [r for r in rows if r["known"]]
unknown = [r for r in rows if not r["known"] and r["hop1"]]  # the model knows the bridge at least
n_known = min(len(known), round(n * dc["known_share"]))
chosen = rng.sample(known, n_known) + rng.sample(unknown, min(len(unknown), n - n_known))
facts = [Fact.from_dict(r["fact"]) for r in chosen]
known_uid = {r["fact"]["uid"] for r in known}

lm, lens = setup(cfg)
lenses = lens_set(cfg, lm.model, lens)
layers = pr.get("band") or list(range(lm.n_layers // 2, lm.n_layers))
print(f"{len(facts)} facts ({n_known} known), {dc['n_agents']} agents, {dc['rounds']} rounds, {fmt}")
records = run_debate(lm, lenses, facts, fmt, dc["n_agents"], dc["rounds"], dc["temperature"], cfg["batch_size"],
                     ec.get("gen_tokens", 12), cfg["seed"])
for r in records:
    r["known"] = r["uid"] in known_uid
out = entity_dir(cfg) / f"debate_{args.split}"
out.mkdir(parents=True, exist_ok=True)
with open(out / "records.jsonl", "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
for name in lenses:
    print(f"--- {name} (band {layers})")
    print(EM.debate_table(records, name, layers).to_string(index=False))
print(f"-> {out} ({len(records)} records)")
