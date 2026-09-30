"""Entity study, step 1: which two-hop facts does the model know, and with whom are they paired?

For every fact (TwoHopFact + hand-made, entity.sources) the model completes hop 1 (-> bridge),
hop 2 (-> answer) and the composed statement (-> answer); a fact counts as known if all three
are right. Facts are split dev / test by bridge entity (entity.dev_frac). Within each split,
every known fact gets a peer and a control fact (same category, all bridges and answers
distinct) and, where one exists among the known facts, a retention sibling.

Writes <out_dir>/entity/{known.jsonl, items_dev.jsonl, items_test.jsonl, facts_dev.jsonl,
facts_test.jsonl}; facts_*.jsonl keep all facts with their known flags (for the free debate).

    python scripts/entity_build.py --config configs/qwen35_4b_entity.yaml
"""
import argparse
import json
import random
from collections import Counter

from silent_dissent.config import load_config, setup
from silent_dissent.entity_experiments import known_check
from silent_dissent.entity_facts import assign_partners, load_facts, siblings, split_of
from silent_dissent.entity_io import entity_dir

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--limit", type=int, help="debug: only the first N facts")
args = ap.parse_args()

cfg = load_config(args.config)
ec = cfg["entity"]
facts = load_facts(ec["sources"])
cap = ec.get("max_per_category")
if cap:  # a fixed random subset per category keeps the known-check affordable
    rng = random.Random(cfg["seed"])
    by: dict[str, list] = {}
    for f in facts:
        by.setdefault(f.category, []).append(f)
    facts = [f for fs in by.values() for f in (rng.sample(fs, cap) if len(fs) > cap else fs)]
facts = facts[: args.limit]
print(f"{len(facts)} facts in {len({f.category for f in facts})} categories")

lm, _ = setup(cfg)
kc = known_check(lm, facts, ec.get("gen_batch_size", cfg["batch_size"]), ec.get("gen_tokens", 12))
out = entity_dir(cfg)
out.mkdir(parents=True, exist_ok=True)
with open(out / "known.jsonl", "w", encoding="utf-8") as fh:
    for f, k in zip(facts, kc):
        fh.write(json.dumps({**k, "category": f.category, "source": f.source}, ensure_ascii=False) + "\n")
known_uid = {k["uid"] for k in kc if k["hop1"] and k["hop2"] and k["composed"]}
if not ec.get("known_filter", True):  # debug configs only: a random model knows nothing
    known_uid = {f.uid for f in facts}
print(f"known (hop1, hop2 and composed right): {len(known_uid)}/{len(facts)}; "
      f"hop1 {sum(k['hop1'] for k in kc)}, hop2 {sum(k['hop2'] for k in kc)}, composed {sum(k['composed'] for k in kc)}")

flag = {k["uid"]: k for k in kc}
for split in ("dev", "test"):
    in_split = [f for f in facts if split_of(f, ec["dev_frac"], cfg["seed"]) == split]
    known = [f for f in in_split if f.uid in known_uid]
    triples = assign_partners(known, known, cfg["seed"])
    sib = siblings([f for f, _, _ in triples], known)
    with open(out / f"items_{split}.jsonl", "w", encoding="utf-8") as fh:
        for f, p, c in triples:
            s = sib.get(f.uid)
            fh.write(json.dumps({"fact": f.to_dict(), "peer": p.to_dict(), "control": c.to_dict(),
                                 "sibling": s.to_dict() if s else None}, ensure_ascii=False) + "\n")
    with open(out / f"facts_{split}.jsonl", "w", encoding="utf-8") as fh:
        for f in in_split:
            k = flag[f.uid]
            fh.write(json.dumps({"fact": f.to_dict(), "known": f.uid in known_uid, "hop1": k["hop1"],
                                 "hop2": k["hop2"], "composed": k["composed"]}, ensure_ascii=False) + "\n")
    cats = Counter(f.category for f, _, _ in triples)
    print(f"[{split}] {len(in_split)} facts, {len(known)} known, {len(triples)} items with partners, "
          f"{len(sib)} with a retention sibling; categories: {dict(cats.most_common(8))}")
