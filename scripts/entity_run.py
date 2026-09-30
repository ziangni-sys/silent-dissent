"""Entity study: round 0 plus the validation controls (E0) or the pressure conditions (E1/E2).

  --stage e0        agree, mention, mention_bridge, instructed, hypothetical, retention (own
                    answer visible and hidden); one round; formats from entity.formats
  --stage pressure  pressure x entity.styles x entity.n_peers for entity.rounds rounds, with
                    agree / mention / mention_bridge as in-run baselines; the format is the one
                    fixed in the entity prereg if it exists (else entity.formats / --formats)

Round 0 is run once per item and format; later rounds only for items answered correctly there.
Writes <out_dir>/entity/<stage>_<split>[_<tag>]/{records.jsonl, ranks.npz, info.json}.

Follow-up studies (addenda, README, section "Entity-answer study"). On the dev split these options are free:
  --later V            how rounds >= 2 are written (entity_prompts.LATER; round 1 is the same)
  --n-peers N.. / --rounds R      override entity.n_peers / entity.rounds
  --bridge-types T..   only items whose bridge has one of these types (e.g. person), all categories
  --probe-mode M       probe tokens of an entity (entity_facts.PROBE_MODES)
  --extended           readouts also at the mention's last token and the mention..last span
  --own hidden|absent  pilot of the "own" addendum: the pressure and baseline cells of
                       entity.addenda.own with the agent's round-0 statement hidden (a placeholder)
                       or absent (it answers first after the peers); n_peers and one round from the
                       same config block, --limit defaults to its pilot_limit
On the test split they come only from a committed addendum: --addendum rounds | bridges | own.

    python scripts/entity_run.py --config configs/qwen35_4b_entity.yaml --stage e0 --split dev
    python scripts/entity_run.py --config C --stage pressure --split test --addendum rounds
"""
import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd

from silent_dissent import entity_metrics as EM
from silent_dissent.config import load_config, setup
from silent_dissent.entity_experiments import classify, entity_readout, generate, readout_inputs, run_rounds
from silent_dissent.entity_facts import PROBE_MODES, role_probes
from silent_dissent.entity_io import (ADDENDA, addendum_path, entity_dir, lens_set, load_addendum, load_items,
                                      prereg_categories, save_run, states_after_round0)
from silent_dissent.entity_prompts import LATER, EntityState

OWN = {"hidden": {"hidden_own": True}, "absent": {"absent_own": True}}  # addendum "own" variants

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--stage", choices=["e0", "pressure"], required=True)
ap.add_argument("--split", choices=["dev", "test"], required=True)
ap.add_argument("--formats", nargs="+")
ap.add_argument("--limit", type=int, help="only the first N items")
ap.add_argument("--tag", help="suffix for the output directory")
ap.add_argument("--later", choices=LATER)
ap.add_argument("--n-peers", type=int, nargs="+")
ap.add_argument("--rounds", type=int)
ap.add_argument("--bridge-types", nargs="+")
ap.add_argument("--probe-mode", choices=PROBE_MODES)
ap.add_argument("--extended", action="store_true")
ap.add_argument("--own", choices=list(OWN), help="dev pilot of the own addendum (see above)")
ap.add_argument("--addendum", choices=ADDENDA, help="take the follow-up settings from this addendum")
ap.add_argument("--no-extra-lenses", action="store_true", help="only the configured lens and the logit lens")
args = ap.parse_args()

cfg = load_config(args.config)
ec = cfg["entity"]
manual = [o for o in ("later", "n_peers", "rounds", "bridge_types", "probe_mode", "own") if getattr(args, o) is not None]
manual += ["extended"] if args.extended else []
if args.split == "test" and manual:
    raise SystemExit(f"--{', --'.join(o.replace('_', '-') for o in manual)}: on the test split these settings come "
                     "only from a committed addendum (--addendum)")
prereg = Path(cfg.get("entity_prereg", "")) if cfg.get("entity_prereg") else None
if args.formats:
    formats = args.formats
elif args.stage == "pressure" and prereg and prereg.exists():
    formats = [json.loads(prereg.read_text())["format"]]
else:
    formats = ec["formats"]
if args.stage == "e0":
    specs = [{"condition": c} for c in ("agree", "mention", "mention_bridge", "instructed", "hypothetical")]
    specs += [{"condition": "retention"}, {"condition": "retention", "hidden_own": True}]
    rounds, n_peers = 1, ec.get("e0_n_peers", [3])
else:
    specs = [{"condition": "pressure", "style": s} for s in ec["styles"]]
    specs += [{"condition": c} for c in ("agree", "mention", "mention_bridge")]
    rounds, n_peers = ec["rounds"], ec["n_peers"]
cats = prereg_categories(cfg) if args.stage == "pressure" else None
later, probe_mode, extended, bridge_types = args.later or "repeat", args.probe_mode or "words", args.extended, \
    args.bridge_types
n_peers, rounds = args.n_peers or n_peers, args.rounds or rounds
limit = args.limit
if args.own and args.addendum:
    raise SystemExit("--own is the dev pilot; the test run takes its settings from --addendum own")
if args.own:  # dev pilot of the own addendum (the cells and rules declared in entity.addenda.own)
    oc = ec["addenda"]["own"]
    if args.stage != "pressure":
        raise SystemExit("--own needs --stage pressure")
    specs = [{"condition": "pressure", "style": st, **OWN[args.own]} for st in oc["styles"]]
    specs += [{"condition": c, **OWN[args.own]} for c in oc["baselines"]]
    n_peers, rounds = args.n_peers or [oc["n_peers"]], args.rounds or 1
    limit = args.limit or oc.get("pilot_limit")
if bridge_types:
    cats = None  # all categories with such bridges (the prereg's categories have place bridges)
tag = args.tag
if args.addendum:
    add = load_addendum(cfg, args.addendum)
    if add is None or args.stage != "pressure":
        raise SystemExit(f"--addendum {args.addendum} needs --stage pressure and {addendum_path(cfg, args.addendum)}")
    specs = [{"condition": "pressure", "style": s} for s in add["styles"]]
    specs += [{"condition": c} for c in add["baselines"]]
    formats = [add["format"]]
    if args.addendum == "rounds":
        later, n_peers, rounds = add["later"], [add["n_peers"]], add["round"]
    elif args.addendum == "own":  # every registered variant: the same cells, one round
        specs = [{**sp, **OWN[v]} for v in add["variants"] for sp in specs]
        cats, n_peers, rounds = add["categories"], [add["n_peers"]], 1
    else:
        cats, probe_mode, extended, n_peers, rounds = add["categories"], add["probe_mode"], True, add["n_peers"], 1
    tag = tag or args.addendum
elif args.own:
    tag = tag or f"own_{args.own}"

lm, lens = setup(cfg)
gen_bs = ec.get("gen_batch_size", cfg["batch_size"])
lenses = lens_set(cfg, lm.model, lens, not args.no_extra_lenses)
items = load_items(cfg, args.split, limit, cats, bridge_types)
print(f"{len(items)} items ({args.split}), formats {formats}, lenses {list(lenses)}, {len(specs)} conditions x "
      f"n_peers {n_peers}, {rounds} round(s), later rounds {later}, probe mode {probe_mode}"
      + (", extended readout" if extended else ""))

records, arrays = [], {name: [] for name in lenses}
for fmt in formats:
    solo = [EntityState(f, p, c, "solo") for f, p, c, _ in items]
    roles = [role_probes(lm.tok, f, p, c, probe_mode) for f, p, c, _ in items]
    prompts, offs, *moffs = readout_inputs(lm.tok, solo, fmt, extended)
    arr = entity_readout(lm, lenses, prompts, offs, roles, cfg["batch_size"], moffs[0] if extended else None)
    answers = generate(lm, prompts, ec.get("gen_tokens", 12), gen_bs, desc=f"{fmt} round 0")
    for (f, p, c, _), a, rl in zip(items, answers, roles):
        records.append({"uid": f.uid, "category": f.category, "split": args.split, "format": fmt, "condition": "solo",
                        "style": None, "hidden_own": False, "absent_own": False, "n_peers": 0, "round": 0, "answer": a,
                        "outcome": classify(a, f, p), "roles": rl})
    for name in lenses:
        arrays[name].append(arr[name])
    acc = sum(r["outcome"] == "original" for r in records[-len(items):]) / max(len(items), 1)
    print(f"[{fmt}] round 0 accuracy {acc:.3f}")

    states, meta, sroles = states_after_round0(lm.tok, items, answers, specs, n_peers,
                                               require_correct=ec.get("require_correct_round0", True),
                                               probe_mode=probe_mode, later=later)
    for m, rl in zip(meta, sroles):
        m.update({"split": args.split, "format": fmt, "roles": rl})
    recs, arr = run_rounds(lm, lenses, states, meta, sroles, fmt, rounds, cfg["batch_size"], ec.get("gen_tokens", 12),
                           gen_bs, extended)
    records += recs
    for name in lenses:
        if len(arr[name]):
            arrays[name].append(arr[name])

try:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
except Exception:
    commit = None
out = entity_dir(cfg) / (f"{args.stage}_{args.split}" + (f"_{tag}" if tag else ""))
save_run(out, records, arrays, {"stage": args.stage, "split": args.split, "formats": formats, "rounds": rounds,
                                "n_peers": n_peers, "specs": specs, "lenses": list(lenses), "n_items": len(items),
                                "categories": cats, "bridge_types": bridge_types, "later": later,
                                "probe_mode": probe_mode, "extended": extended, "addendum": args.addendum, "own": args.own,
                                "limit": limit,
                                "config": args.config, "code_commit": commit})
pd.set_option("display.width", 200)
print(EM.outcome_table(records).to_string(index=False))
print(f"-> {out} ({len(records)} records)")
