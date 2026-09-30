"""Entity study across models: one JSON with what the paper's tables and figures need (no model).

For every entity config (default: Qwen3.5-4B, Qwen3.6-27B, Gemma-4-E4B-it, Llama-3.1-8B-Instruct) it reads
the entity prereg, pressure_test, inject_test and debate_test and writes results/crossmodel/summary.json:
  meta         model, lens, rule, format, band, k, categories, test items, round-0 accuracy
  hypotheses   H1-H4 as pre-registered (entity_metrics.evaluate_hypotheses)
  robustness   H1 / H2 / H4 on the same test records under other readouts: the band of the other
               selection rule where it selects one (ALT_BANDS, from the dev split), k in K_GRID, every
               n_peers, and every extra lens (as the lens of H1, compared with the logit lens in H4)
  flips        share of agents giving the peers' answer, per style and n_peers (round 1)
  cells        band means with CIs, per lens, of orig_bridge / peer_bridge in round 0, agree, mention,
               mention_bridge (answer kept) and pressure-answer held / flipped (primary n_peers)
  curves       the same cells per layer (hit@k, last token), per lens
  k_cells      orig_bridge band means of round 0 / agree / held / flipped for every k in K_GRID, per lens
  inject       E4: share of agents that gave in answering the original after injection (and of held
               agents answering the peers'), per kind x alpha, pooled over the injected layers
  debate       E5 group accuracy per round (stated / latent / round-0 plurality), primary lens
  rounds       the rounds addendum verdicts, where one was run
  own          the own addendum (agent's round-0 answer hidden / absent): verdicts O1-O4 and the
               exploratory summary (rates, cells, paired comparisons), where it was run
  debate_bridges  E5 follow-up: the original-bridge readout of free-debate agents per transition
               (scripts/entity_debate_bridges.py, analysis/debate_bridges.json), where it was run
Only the test split is read, and nothing is refit or reselected: the robustness rows are exploratory.

    python scripts/entity_crossmodel.py [--configs C ...] [--out results/crossmodel]
"""
import argparse
import json
import warnings
from pathlib import Path

import numpy as np

from silent_dissent import entity_metrics as EM
from silent_dissent.config import load_config
from silent_dissent.entity_io import entity_dir, load_run
from silent_dissent.metrics import bootstrap_ci

CONFIGS = ["configs/qwen35_4b_entity.yaml", "configs/qwen36_27b_entity.yaml", "configs/gemma4_e4b_it_entity.yaml",
           "configs/llama31_8b_it_entity.yaml"]
# The band the other selection rule gives on each model's dev split (None: it selects nothing there).
# 4B: `entity_select.py` with the v2 settings, dry run; 27B: its v1 prereg (prereg/qwen36_27b_entity.json).
ALT_BANDS = {"qwen35_4b_entity": ("v2", list(range(23, 29))), "qwen36_27b_entity": ("v1", [62]),
             "gemma4_e4b_it_entity": ("v2", None), "llama31_8b_it_entity": ("v1", None)}
K_GRID = [10, 50, 100, 500]
CELLS = {"round0": dict(round=0, outcome="original"),
         "agree": dict(round=1, condition="agree", outcome="original"),
         "mention": dict(round=1, condition="mention", outcome="original"),
         "mention_bridge": dict(round=1, condition="mention_bridge", outcome="original"),
         "held": dict(round=1, condition="pressure", style="answer", outcome="original"),
         "flipped": dict(round=1, condition="pressure", style="answer", outcome="peer")}

ap = argparse.ArgumentParser()
ap.add_argument("--configs", nargs="+", default=CONFIGS)
ap.add_argument("--out", default="results/crossmodel")
args = ap.parse_args()


def clean(x):
    """JSON-safe: numpy scalars to Python, NaN to None."""
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if np.isnan(x) else round(float(x), 4)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def summary(ranks, idx, contrast, k, band):
    s = EM.band_summary(ranks, idx, contrast, "last", k, band)
    return {"mean": s["mean"], "ci": [s["ci_lo"], s["ci_hi"]], "n": s["n"]} if s["n"] else {"mean": None, "n": 0}


def hyp(records, arrays, prereg, n_peers, **over):
    return {h["id"]: {k: h[k] for k in ("estimate", "ci_lo", "ci_hi", "n", "supported")}
            for h in EM.evaluate_hypotheses(records, arrays, {**prereg, **over}, n_peers)}


out = {"configs": args.configs, "models": {}}
for path in args.configs:
    cfg = load_config(path)
    name = Path(path).stem
    root = entity_dir(cfg)
    prereg = json.loads(Path(cfg["entity_prereg"]).read_text())
    if not (root / "pressure_test" / "ranks.npz").exists():
        print(f"{name}: no pressure_test, skipped")
        continue
    records, arrays, info = load_run(root / "pressure_test")
    band, k, lens, n = prereg["band"], prereg["k"], prereg["lens"], prereg["primary_n_peers"]
    rule = "v2" if cfg["entity"]["select"].get("band_rule") == "dialogue" else "v1"
    r0 = [r for r in records if r["round"] == 0]
    m = {"meta": {"model": prereg["model"], "config": path, "prereg": cfg["entity_prereg"], "rule": rule,
                  "format": prereg["format"], "band": band, "k": k, "lens": lens, "categories": prereg.get("categories"),
                  "n_items": info.get("n_items"), "n_layers": int(arrays[lens].shape[2]), "lenses": list(arrays),
                  "round0_accuracy": float(np.mean([r["outcome"] == "original" for r in r0])) if r0 else None}}
    m["hypotheses"] = hyp(records, arrays, prereg, n)

    rob = []
    alt_rule, alt = ALT_BANDS.get(name, (None, None))
    if alt:
        rob.append({"what": f"band of rule {alt_rule}", "band": alt, **hyp(records, arrays, prereg, n, band=alt)})
    for kk in K_GRID:
        if kk != k:
            rob.append({"what": f"k = {kk}", **hyp(records, arrays, prereg, n, k=kk)})
    for nn in sorted({r["n_peers"] for r in records if r["condition"] == "pressure"}):
        if nn != n:
            rob.append({"what": f"{nn} peers", **hyp(records, arrays, prereg, nn)})
    for extra in [x for x in arrays if x not in (lens, "logit")]:
        rob.append({"what": f"lens {extra}", **hyp(records, arrays, prereg, n, lens=extra)})
    m["robustness"] = rob

    flips = {}
    for style in sorted({r["style"] for r in records if r["condition"] == "pressure"}):
        for nn in sorted({r["n_peers"] for r in records if r["condition"] == "pressure"}):
            rr = [r["outcome"] for r in records if r["round"] == 1 and r["condition"] == "pressure"
                  and r["style"] == style and r["n_peers"] == nn]
            flips[f"{style}/{nn}"] = {"peer": float(np.mean([o == "peer" for o in rr])),
                                      "other": float(np.mean([o == "other" for o in rr])), "n": len(rr)}
    m["flips"] = flips

    rows = lambda **kw: [i for i, r in enumerate(records) if all(r.get(a) == b for a, b in kw.items())
                         and (r["round"] == 0 or r.get("n_peers") == n)]
    cells, curves = {}, {}
    for cell, kw in CELLS.items():
        idx = rows(**kw)
        for ln in arrays:
            for c in ("orig_bridge", "peer_bridge"):
                cells[f"{cell}/{ln}/{c}"] = summary(arrays[ln], idx, c, k, band)
                curves[f"{cell}/{ln}/{c}"] = EM.curve(arrays[ln], idx, c, "last", k)
    m["cells"], m["curves"] = cells, {key: list(v) for key, v in curves.items()}
    m["k_cells"] = {f"{kk}/{cell}/{ln}": summary(arrays[ln], rows(**CELLS[cell]), "orig_bridge", kk, band)
                    for kk in K_GRID for cell in ("round0", "agree", "held", "flipped") for ln in arrays}

    inj = root / "inject_test" / "records.jsonl"
    if inj.exists():
        recs = [json.loads(l) for l in inj.open(encoding="utf-8") if l.strip()]
        t = EM.injection_table(recs)
        res = {}
        for outcome, target in (("peer", "original"), ("original", "peer")):
            g = t[t.outcome == outcome]
            if len(g) and target in g:
                w = g.assign(x=g[target] * g.n).groupby(["inject_kind", "alpha"])[["x", "n"]].sum()
                res["gave_in" if outcome == "peer" else "held"] = {
                    f"{kind}/{alpha}": {"share": float(row.x / row.n), "n": int(row.n)} for (kind, alpha), row in w.iterrows()}
        res["layers"] = sorted(int(x) for x in t.inject_layer.unique())
        m["inject"] = res

    deb = root / "debate_test" / "records.jsonl"
    if deb.exists():
        recs = [json.loads(l) for l in deb.open(encoding="utf-8") if l.strip()]
        if recs and lens in recs[0]["candidate_ranks"]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                m["debate"] = EM.debate_table(recs, lens, band).to_dict(orient="records")

    rounds = root / "analysis" / "hypotheses_rounds.json"
    if rounds.exists():
        m["rounds"] = json.loads(rounds.read_text())
    own = root / "analysis" / "hypotheses_own.json"
    if own.exists():
        m["own"] = {"hypotheses": json.loads(own.read_text())}
        if (root / "analysis" / "own_summary.json").exists():
            m["own"]["summary"] = json.loads((root / "analysis" / "own_summary.json").read_text())
    db = root / "analysis" / "debate_bridges.json"
    if db.exists():
        m["debate_bridges"] = json.loads(db.read_text())
    out["models"][name] = m
    h = m["hypotheses"]
    print(f"{name}: " + ", ".join(f"{i} {h[i]['estimate']:.3f}" for i in h) + f"; {len(rob)} robustness rows")

dst = Path(args.out)
dst.mkdir(parents=True, exist_ok=True)
(dst / "summary.json").write_text(json.dumps(clean(out), indent=1) + "\n")
print(f"-> {dst / 'summary.json'}")
