"""E5 check (docs/EXPERIMENT_BACKLOG.md, B07; rule recorded in docs/EXPERIMENT_LOG.md, stage 6, before any
record was opened): what the free-debate agents answered in each round, and whether Qwen3.6-27B's drop
in stated accuracy at rounds 2-3 comes from degenerate answers. Reads debate_test/records.jsonl of every
model (no model, CPU) and writes results/debate_check/summary.json:

  kinds      share of answers per class and round (entity_metrics.debate_kind: correct, other_candidate,
             degenerate, other), for all, known and unknown facts
  strings    the 20 most frequent degenerate or other answers per round
  accuracy   group accuracy per round with bootstrap CIs over facts (2000 resamples, seed 0):
             (a) plurality of stated answers, exactly as entity_metrics.debate_table (checked against it);
             (b) plurality over non-degenerate answers, a fact with none counted as excluded (primary) or
                 as wrong; (c) mean share of correct agents; latent: debate_table's latent plurality
  attributed the rule: the drop is attributed to the protocol iff (i) the degenerate share is >= 0.10
             at round 2 or 3 and (ii) for each of rounds 2 and 3, accuracy (b, excluded) is within 0.05 of
             round 1 or its CI overlaps round 1's

    python scripts/entity_debate_check.py [--configs C ...] [--out results/debate_check]
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from silent_dissent import entity_metrics as EM
from silent_dissent.config import load_config
from silent_dissent.entity_io import entity_dir
from silent_dissent.metrics import bootstrap_ci

CONFIGS = ["configs/qwen35_4b_entity.yaml", "configs/qwen36_27b_entity.yaml", "configs/gemma4_e4b_it_entity.yaml",
           "configs/llama31_8b_it_entity.yaml"]
KINDS = ["correct", "other_candidate", "degenerate", "other"]

ap = argparse.ArgumentParser()
ap.add_argument("--configs", nargs="+", default=CONFIGS)
ap.add_argument("--out", default="results/debate_check")
args = ap.parse_args()


def summ(values):
    v = np.asarray([x for x in values if x is not None], dtype=float)
    lo, hi = bootstrap_ci(v, n_boot=2000, seed=0)
    return {"mean": float(v.mean()) if len(v) else None, "ci": [lo, hi], "n": int(len(v))}


out = {"configs": args.configs, "rule": "attributed iff degenerate share >= 0.10 at round 2 or 3, and accuracy (b, "
       "excluded) at each of rounds 2 and 3 within 0.05 of round 1 or with overlapping CI", "models": {}}
for path in args.configs:
    cfg = load_config(path)
    name = Path(path).stem
    f = entity_dir(cfg) / "debate_test" / "records.jsonl"
    if not f.exists():
        print(f"{name}: no debate_test, skipped")
        continue
    records = [json.loads(l) for l in f.open(encoding="utf-8") if l.strip()]
    prereg = json.loads(Path(cfg["entity_prereg"]).read_text())
    lens, band = prereg["lens"], prereg["band"]
    rounds = sorted({r["round"] for r in records})
    for r in records:
        r["kind"] = EM.debate_kind(r)
    m = {"kinds": {}, "strings": {}, "accuracy": {}}
    for t in rounds:
        rt = [r for r in records if r["round"] == t]
        for subset, keep in (("all", lambda r: True), ("known", lambda r: r.get("known")),
                             ("unknown", lambda r: not r.get("known"))):
            rs = [r for r in rt if keep(r)]
            m["kinds"][f"r{t}/{subset}"] = {k: (sum(r["kind"] == k for r in rs) / len(rs) if rs else None)
                                            for k in KINDS} | {"n": len(rs)}
        m["strings"][f"r{t}"] = Counter(r["answer"] for r in rt if r["kind"] in ("degenerate", "other")).most_common(20)

    by: dict[tuple, list[dict]] = {}
    for r in records:
        by.setdefault((r["uid"], r["round"]), []).append(r)
    table = EM.debate_table(records, lens, band)
    for t in rounds:
        for subset, keep in (("all", lambda g: True), ("known", lambda g: g[0].get("known")),
                             ("unknown", lambda g: not g[0].get("known"))):
            groups = [g for (u, rr), g in by.items() if rr == t and keep(g)]
            a = [EM.debate_group_correct(g) for g in groups]  # debate_table's rule: no usable answer = wrong
            b = [EM.debate_group_correct(g, drop_degenerate=True) for g in groups]
            m["accuracy"][f"r{t}/{subset}"] = {
                "a_stated": summ(a), "b_excluded": summ(b), "b_wrong": summ([bool(x) for x in b]),
                "b_no_valid_answer": sum(x is None for x in b),
                "c_agents": summ([np.mean([r["correct"] for r in g]) for g in groups])}
        ref = float(table.loc[table["round"] == t, "acc_stated"].iloc[0])
        got = m["accuracy"][f"r{t}/all"]["a_stated"]["mean"]
        assert abs(ref - got) < 1e-9, f"{name} round {t}: (a) {got} does not reproduce debate_table {ref}"
        m["accuracy"][f"r{t}/all"]["latent"] = float(table.loc[table["round"] == t, "acc_latent"].iloc[0])
    late = [t for t in (2, 3) if t in rounds]
    if 1 in rounds and late:
        deg = max(m["kinds"][f"r{t}/all"]["degenerate"] for t in late)
        b1 = m["accuracy"]["r1/all"]["b_excluded"]

        def close(t):
            bt = m["accuracy"][f"r{t}/all"]["b_excluded"]
            return abs(bt["mean"] - b1["mean"]) <= 0.05 or (bt["ci"][0] <= b1["ci"][1] and b1["ci"][0] <= bt["ci"][1])

        m["attribution"] = {"max_degenerate_share_r2_r3": deg, "degenerate_ok": deg >= 0.10,
                            "accuracy_ok": all(close(t) for t in late)}
        m["attribution"]["attributed"] = m["attribution"]["degenerate_ok"] and m["attribution"]["accuracy_ok"]
    out["models"][name] = m

    print(f"\n===== {name} ({len(records)} records)")
    for t in rounds:
        k, acc = m["kinds"][f"r{t}/all"], m["accuracy"][f"r{t}/all"]
        print(f"round {t}: " + ", ".join(f"{x} {k[x]:.3f}" for x in KINDS) + f" | stated (a) {acc['a_stated']['mean']:.3f}"
              f", without degenerate (b) {acc['b_excluded']['mean']:.3f} (no valid answer: {acc['b_no_valid_answer']})"
              f", agents (c) {acc['c_agents']['mean']:.3f}, latent {acc['latent']:.3f}")
        print(f"   top answers (degenerate/other): {m['strings'][f'r{t}'][:8]}")
    if "attribution" in m:
        a = m["attribution"]
        print(f"ATTRIBUTED TO THE PROTOCOL: {'yes' if a['attributed'] else 'no'} (max degenerate share r2-r3 "
              f"{a['max_degenerate_share_r2_r3']:.3f}, accuracy criterion {'met' if a['accuracy_ok'] else 'not met'})")

dst = Path(args.out)
dst.mkdir(parents=True, exist_ok=True)


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


(dst / "summary.json").write_text(json.dumps(clean(out), indent=1, ensure_ascii=False) + "\n")
print(f"-> {dst / 'summary.json'}")
