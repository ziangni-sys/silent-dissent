"""E5 check, EXPLORATORY and NOT pre-registered: the attribution rule of scripts/entity_debate_check.py
(B07) under broader definitions of a non-answer. Defined on 2026-09-30 after the registered check had been
run and read (Qwen3.6-27B: not attributed; degenerate share 0.082 at round 2, while 'other' answers such as
'the agents', 'an island' or 'loop' were 0.508), before any number of this script was computed.

Why: the registered class 'degenerate' (entity_metrics.debate_kind) matches the word 'agent' only in the
singular, although its description says "a mention of an agent", so 'agents' / 'the agents' fall into
'other'; descriptions and fragments ('an island', 'loop', 'unconfirmed') are 'other' too. Under rule (b)
an 'other' answer is a vote for a wrong answer, not an abstention.

Variants (which answers are abstentions; everything else exactly as in entity_debate_check.py):
  registered   debate_kind == 'degenerate' (must reproduce the registered (b) and degenerate share)
  agent_words  registered, plus 'other' answers with a word starting with 'agent' (agents, agent's, ...)
  candidates   every answer that is neither correct nor another round-0 candidate ('degenerate' and
               'other'); an upper bound, since it also drops real wrong entities nobody proposed
Same rule and thresholds as registered: attributed iff the abstention share is >= 0.10 at round 2 or 3,
and accuracy (b: plurality over the remaining answers, a group with none left excluded) at each of rounds
2 and 3 is within 0.05 of round 1 or has an overlapping CI (2000 bootstrap resamples over facts, seed 0).
Reads debate_test/records.jsonl and results/debate_check/summary.json (no model, CPU); writes
results/debate_check_explore/summary.json and refuses to overwrite it.

    python scripts/entity_debate_check_explore.py [--configs C ...] [--out results/debate_check_explore]
"""
import argparse
import json
from pathlib import Path

import numpy as np

from silent_dissent import entity_metrics as EM
from silent_dissent.config import load_config
from silent_dissent.entity_facts import norm
from silent_dissent.entity_io import entity_dir
from silent_dissent.metrics import bootstrap_ci

CONFIGS = ["configs/qwen35_4b_entity.yaml", "configs/qwen36_27b_entity.yaml", "configs/gemma4_e4b_it_entity.yaml",
           "configs/llama31_8b_it_entity.yaml"]

ap = argparse.ArgumentParser()
ap.add_argument("--configs", nargs="+", default=CONFIGS)
ap.add_argument("--registered", default="results/debate_check/summary.json")
ap.add_argument("--out", default="results/debate_check_explore")
args = ap.parse_args()
dst = Path(args.out)
if (dst / "summary.json").exists():
    raise SystemExit(f"REFUSED: {dst / 'summary.json'} exists (not overwritten)")
registered = json.loads(Path(args.registered).read_text())["models"]


def agent_word(r):
    return any(w.startswith("agent") for w in norm(r["answer"]).split())


VARIANTS = {
    "registered": lambda r: r["kind"] == "degenerate",
    "agent_words": lambda r: r["kind"] == "degenerate" or (r["kind"] == "other" and agent_word(r)),
    "candidates": lambda r: r["kind"] in ("degenerate", "other"),
}


def summ(values):
    v = np.asarray([x for x in values if x is not None], dtype=float)
    lo, hi = bootstrap_ci(v, n_boot=2000, seed=0)
    return {"mean": float(v.mean()) if len(v) else None, "ci": [lo, hi], "n": int(len(v))}


def group_correct(recs, abstain):
    """entity_metrics.debate_group_correct with drop_degenerate, for any abstention rule."""
    truth = recs[0]["candidates"][0]
    p = EM._plurality([truth if r["correct"] else r["answer"] for r in recs if not abstain(r)])
    return None if p is None else p == norm(truth)


out = {"configs": args.configs, "exploratory": True, "rule": "as registered (abstention share >= 0.10 at round 2 "
       "or 3, and accuracy (b, excluded) at rounds 2 and 3 within 0.05 of round 1 or overlapping CI), with the "
       "abstention definition of each variant", "models": {},
       "variants": {"registered": "debate_kind == degenerate",
                    "agent_words": "registered, plus 'other' answers with a word starting with 'agent'",
                    "candidates": "neither correct nor another round-0 candidate (degenerate or other)"}}
for path in args.configs:
    cfg = load_config(path)
    name = Path(path).stem
    f = entity_dir(cfg) / "debate_test" / "records.jsonl"
    if not f.exists():
        print(f"{name}: no debate_test, skipped")
        continue
    records = [json.loads(l) for l in f.open(encoding="utf-8") if l.strip()]
    for r in records:
        r["kind"] = EM.debate_kind(r)
    rounds = sorted({r["round"] for r in records})
    by: dict[tuple, list[dict]] = {}
    for r in records:
        by.setdefault((r["uid"], r["round"]), []).append(r)
    m = {}
    print(f"\n===== {name} ({len(records)} records; EXPLORATORY, not pre-registered)")
    for v, abstain in VARIANTS.items():
        mv = {"share": {}, "b_excluded": {}, "b_no_valid_answer": {}}
        for t in rounds:
            groups = [g for (u, rr), g in by.items() if rr == t]
            rt = [r for r in records if r["round"] == t]
            mv["share"][f"r{t}"] = float(np.mean([abstain(r) for r in rt]))
            b = [group_correct(g, abstain) for g in groups]
            mv["b_excluded"][f"r{t}"] = summ(b)
            mv["b_no_valid_answer"][f"r{t}"] = sum(x is None for x in b)
            if v == "registered":  # must reproduce the registered check exactly
                assert b == [EM.debate_group_correct(g, drop_degenerate=True) for g in groups], f"{name} r{t}: (b)"
                reg = registered[name]
                for ours, theirs in ((mv["b_excluded"][f"r{t}"]["mean"], reg["accuracy"][f"r{t}/all"]["b_excluded"]["mean"]),
                                     (mv["share"][f"r{t}"], reg["kinds"][f"r{t}/all"]["degenerate"])):
                    assert abs(ours - theirs) < 1e-4, f"{name} r{t}: {ours} does not reproduce the registered {theirs}"
        late = [t for t in (2, 3) if t in rounds]
        if 1 in rounds and late:
            share = max(mv["share"][f"r{t}"] for t in late)
            b1 = mv["b_excluded"]["r1"]

            def close(t):
                bt = mv["b_excluded"][f"r{t}"]
                return abs(bt["mean"] - b1["mean"]) <= 0.05 or (bt["ci"][0] <= b1["ci"][1] and b1["ci"][0] <= bt["ci"][1])

            mv["attribution"] = {"max_abstention_share_r2_r3": share, "share_ok": share >= 0.10,
                                 "accuracy_ok": all(close(t) for t in late)}
            mv["attribution"]["attributed"] = mv["attribution"]["share_ok"] and mv["attribution"]["accuracy_ok"]
        m[v] = mv
        print(f"-- {v}")
        for t in rounds:
            b = mv["b_excluded"][f"r{t}"]
            print(f"   round {t}: abstention share {mv['share'][f'r{t}']:.3f} | (b) {b['mean']:.3f} "
                  f"[{b['ci'][0]:.3f}, {b['ci'][1]:.3f}] n={b['n']} (no answer left: {mv['b_no_valid_answer'][f'r{t}']})")
        if "attribution" in mv:
            a = mv["attribution"]
            print(f"   ATTRIBUTED TO THE PROTOCOL ({v}, exploratory): {'yes' if a['attributed'] else 'no'} (max abstention "
                  f"share r2-r3 {a['max_abstention_share_r2_r3']:.3f}, accuracy criterion "
                  f"{'met' if a['accuracy_ok'] else 'not met'})")
    out["models"][name] = m


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if np.isnan(x) else round(float(x), 4)
    if isinstance(x, (np.integer, np.bool_)):
        return x.item()
    return x


dst.mkdir(parents=True, exist_ok=True)
(dst / "summary.json").write_text(json.dumps(clean(out), indent=1, ensure_ascii=False) + "\n")
print(f"-> {dst / 'summary.json'}")
