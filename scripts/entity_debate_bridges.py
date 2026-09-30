"""E5 bridges: is the original bridge still read in agents that give in during free debate? The rules
were recorded in docs/EXPERIMENT_LOG.md (stage 6) before this was run; the analysis is exploratory.

Readout (GPU): nothing is generated. The prompts of the existing free debate (debate_test/records.jsonl)
are rebuilt exactly as run_debate built them, from every agent's recorded answers, and the entity roles
are read at the last token of every agent's prompt in every round, with the registered lens and the
logit lens (entity_readout): the original bridge and answer, and those of a control fact (a random fact
of the same category from facts_<split>.jsonl with a different bridge and answer, drawn with the config
seed in uid order). Before anything is written, the rebuilt round-1 prompts must reproduce the stored
candidate ranks of the registered lens (share of values within 0.1 in log10(rank + 1) >= 0.95); otherwise
the script stops with exit code 3. Writes <out_dir>/entity/debate_<split>_bridges/ (records.jsonl,
ranks.npz, info.json) and refuses to overwrite it.

Analysis (CPU; entity_metrics.debate_bridge_summary): for every round r >= 1, agents correct at r - 1
are labelled gave_in (wrong at r, stating the wrong answer another agent gave at r - 1), switched,
held (correct although another agent gave a wrong, non-degenerate answer at r - 1) or agreed; per
label, the original-bridge readout (hit@k minus control, the registered band, k and lens, last token)
under both lenses, the paired lens difference and logrank for agents that gave in, and gave in minus
held; known facts (primary) and all facts; round 1 primary. Records are resampled by fact (10000
resamples, seed 0); intervals only with >= 20 readable records. -> analysis/debate_bridges.json.

    python scripts/entity_debate_bridges.py --config configs/qwen36_27b_entity.yaml             # GPU
    python scripts/entity_debate_bridges.py --config configs/qwen36_27b_entity.yaml --analyze   # CPU only
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

from silent_dissent import entity_metrics as EM
from silent_dissent.config import load_config
from silent_dissent.entity_facts import Fact, norm, probe_tokens, role_probes
from silent_dissent.entity_io import entity_dir, load_run, save_run

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--split", choices=["dev", "test"], default="test")
ap.add_argument("--analyze", action="store_true", help="only the analysis of an existing readout (CPU)")
args = ap.parse_args()

cfg = load_config(args.config)
root = entity_dir(cfg)
prereg = json.loads(Path(cfg["entity_prereg"]).read_text())
lens_name, band, k, fmt = prereg["lens"], prereg["band"], prereg["k"], prereg["format"]
out = root / f"debate_{args.split}_bridges"
MIN_MATCH = 0.95


def analyze():
    records, arrays, _ = load_run(out)
    summary = EM.debate_bridge_summary(records, arrays, lens_name, band, k)
    (root / "analysis").mkdir(exist_ok=True)
    (root / "analysis" / "debate_bridges.json").write_text(json.dumps(summary, indent=1, default=float) + "\n")
    print(f"transition counts: {summary['counts']}")
    for key, c in summary["cells"].items():
        if "/was_wrong/" not in key and c["n"]:
            ci = f" [{c['ci'][0]:.3f}, {c['ci'][1]:.3f}]" if "ci" in c else ""
            print(f"  {key:42s} {c['mean']:.3f}{ci} n={c['n']} ({c['n_facts']} facts)")
    for key, c in summary["contrasts"].items():
        if c.get("mean") is not None:
            ci = f" [{c['ci'][0]:.3f}, {c['ci'][1]:.3f}]" if "ci" in c else ""
            print(f"  {key:42s} {c['mean']:.3f}{ci} n={c['n']}")
    print(f"-> {root / 'analysis' / 'debate_bridges.json'}")


if args.analyze:
    analyze()
    sys.exit(0)
if out.exists():
    raise SystemExit(f"REFUSED: {out} exists (not overwritten); use --analyze to redo the analysis")

src = root / f"debate_{args.split}" / "records.jsonl"
debate = [json.loads(l) for l in src.open(encoding="utf-8") if l.strip()]
with open(root / f"facts_{args.split}.jsonl", encoding="utf-8") as f:
    rows = [json.loads(l) for l in f]
by_uid = {r["fact"]["uid"]: Fact.from_dict(r["fact"]) for r in rows}
uids = list(dict.fromkeys(r["uid"] for r in debate if r["round"] == 0))  # run_debate's fact order
facts = [by_uid[u] for u in uids]
n_agents = max(r["agent"] for r in debate) + 1
rounds = max(r["round"] for r in debate)
hist = {(r["uid"], r["agent"], r["round"]): r for r in debate}

# controls: a same-category fact with a different bridge and answer, drawn in uid order with the config seed
rng = random.Random(cfg["seed"])
pool: dict[str, list[Fact]] = {}
for r in rows:
    pool.setdefault(r["fact"]["category"], []).append(Fact.from_dict(r["fact"]))
control = {}
for u in sorted(uids):
    f = by_uid[u]
    cands = [c for c in pool[f.category] if norm(c.e2) != norm(f.e2) and norm(c.e3) != norm(f.e3)]
    control[u] = rng.choice(cands)

import torch  # noqa: E402
from silent_dissent.config import setup  # noqa: E402  (loads the model; not needed for --analyze)
from silent_dissent.entity_experiments import _candidate_ranks, debate_prompt, entity_readout, subject_offset  # noqa: E402
from silent_dissent.entity_io import lens_set  # noqa: E402

lm, lens = setup(cfg)
lenses = lens_set(cfg, lm.model, lens, extra_lenses=False)


def prompts_for(r):
    """run_debate's prompts of round r, in its order (facts, then agents)."""
    ps, index = [], []
    for i, f in enumerate(facts):
        for a in range(n_agents):
            own = [hist[(f.uid, a, t)]["answer"] for t in range(r)]
            others = [[hist[(f.uid, b, t)]["answer"] for b in range(n_agents) if b != a] for t in range(r)]
            ps.append(debate_prompt(fmt, lm.tok, f, own, others))
            index.append((f, a))
    return ps, index


# check: the rebuilt round-1 prompts reproduce the stored candidate ranks
if rounds >= 1:
    ps, index = prompts_for(1)
    cand_roles = {}
    for f in facts:
        ids = [probe_tokens(lm.tok, x) for x in hist[(f.uid, 0, 0)]["candidates"]]
        count: dict[int, int] = {}
        for s in ids:
            for t in set(s):
                count[t] = count.get(t, 0) + 1
        cand_roles[f.uid] = [[t for t in s if count[t] == 1] for s in ids]
    with torch.no_grad():  # as inside run_debate; with autograd on, the 27B runs out of memory
        got = _candidate_ranks(lm, {lens_name: lenses[lens_name]}, ps, [cand_roles[f.uid] for f, _ in index],
                               cfg["batch_size"])[lens_name]
    a_ = np.array([x for (f, a), g in zip(index, got) for c in g for x in c], dtype=float)
    b_ = np.array([x for (f, a) in index for c in hist[(f.uid, a, 1)]["candidate_ranks"][lens_name] for x in c],
                  dtype=float)
    if len(a_) != len(b_):
        print(f"RECONSTRUCTION FAILED: {len(a_)} rebuilt ranks against {len(b_)} stored; nothing written")
        sys.exit(3)
    ok = (a_ >= 0) & (b_ >= 0)
    match = float(np.mean(np.abs(np.log10(a_[ok] + 1) - np.log10(b_[ok] + 1)) <= 0.1)) if ok.any() else 0.0
    exact = float(np.mean(a_ == b_))
    print(f"reconstruction check (round 1, {lens_name}): {match:.4f} of {int(ok.sum())} ranks within 0.1 log10, "
          f"{exact:.4f} identical")
    if match < MIN_MATCH:
        print(f"RECONSTRUCTION FAILED (< {MIN_MATCH}): nothing written")
        sys.exit(3)
else:
    match = exact = None

records, arrays = [], {name: [] for name in lenses}
for r in range(rounds + 1):
    ps, index = prompts_for(r)
    roles = [role_probes(lm.tok, f, None, control[f.uid]) for f, _ in index]
    offs = [subject_offset(lm.tok, p, f.e1) for p, (f, _) in zip(ps, index)]
    arr = entity_readout(lm, lenses, ps, [o if o is not None else -1 for o in offs], roles, cfg["batch_size"])
    for (f, a), rl in zip(index, roles):
        d = hist[(f.uid, a, r)]
        records.append({"uid": f.uid, "category": f.category, "known": d.get("known"), "agent": a, "round": r,
                        "answer": d["answer"], "correct": d["correct"], "candidates": d["candidates"],
                        "control_uid": control[f.uid].uid, "roles": rl})
    for name in lenses:
        arrays[name].append(arr[name])
    print(f"round {r}: {len(ps)} prompts read")
save_run(out, records, arrays, {"source": str(src), "lens": lens_name, "band": band, "k": k, "format": fmt,
                                "n_facts": len(facts), "n_agents": n_agents, "rounds": rounds,
                                "reconstruction_match": match, "reconstruction_identical": exact,
                                "config": args.config})
print(f"-> {out} ({len(records)} records)")
analyze()
