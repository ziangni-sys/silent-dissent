"""Entity study: exploratory robustness checks on the raw test readouts (no model; CPU), answering
review questions. Decided 2026-09-30 before any of these numbers was computed (docs/EXPERIMENT_LOG.md,
stage 5); nothing here is pre-registered, and the pre-registered verdicts stay those of
entity_analyze / entity_crossmodel.

For every entity config it reads the prereg and pressure_test and uses the pre-registered readout
(lens, band, k, last token, format, primary n_peers, round 1); results/robust/summary.json holds:
  threshold_free  readouts without the top-k threshold, per record and band layer from the ranks of
                  the entity and of the control (ranks are clipped at 30000):
                  logrank = log10(rank_control + 1) - log10(rank_entity + 1), in orders of magnitude
                  (0: no preference); win = share of band layers where the entity outranks the control
                  (ties 1/2), given as its margin over 1/2 (0: no preference).
                  cells: round0, agree, mention, mention_bridge, held, flipped; orig and peer bridge;
                  every lens. tests: H1, H2, H4 with these readouts.
  k_curve         hit@k contrast of the same cells for k from 1 to 10000.
  within_item     the same items in two conditions (pre-registered readout "hit" and "logrank"):
                  flipped - agree and flipped - round 0 (original bridge), held - agree, flipped -
                  mention (peers' bridge: H2 on the same items), with the ratio of the two means; and
                  selection: do the items that gave in differ before any pressure (round 0 and agree
                  readouts of items that gave in minus items that held)?
  categories      per category: give-in counts, H1 (hit, logrank) and H2 (hit); intervals only where
                  every sample has >= MIN_N records.
  tests           H1-H4 as pre-registered, with a record bootstrap and a cluster bootstrap (records
                  resampled by bridge entity: the original bridge for H1, H3, H4, the peers' bridge for
                  H2; one draw shared by both samples of a difference), 95% percentile intervals and
                  two-sided bootstrap p-values.
  holm            Holm correction over every pre-registered test with an estimate (models x H1-H4),
                  family-wise alpha 0.05, on the cluster p-values (and, for comparison, the record ones).
Every "test" entry: estimate, n, n_clusters, record {ci, p}, cluster {ci, p}.

    python scripts/entity_robust.py [--configs C ...] [--out results/robust] [--n-boot 10000]
"""
import argparse
import json
from pathlib import Path

import numpy as np

from silent_dissent import entity_metrics as EM
from silent_dissent.config import load_config
from silent_dissent.entity_facts import norm
from silent_dissent.entity_io import entity_dir, load_items, load_run

CONFIGS = ["configs/qwen35_4b_entity.yaml", "configs/qwen36_27b_entity.yaml", "configs/gemma4_e4b_it_entity.yaml",
           "configs/llama31_8b_it_entity.yaml"]
KS = [1, 3, 10, 30, 100, 300, 1000, 3000, 10000]
CELLS = {"round0": dict(round=0, outcome="original"),
         "agree": dict(round=1, condition="agree", outcome="original"),
         "mention": dict(round=1, condition="mention", outcome="original"),
         "mention_bridge": dict(round=1, condition="mention_bridge", outcome="original"),
         "held": dict(round=1, condition="pressure", style="answer", outcome="original"),
         "flipped": dict(round=1, condition="pressure", style="answer", outcome="peer")}
MIN_N = 10

ap = argparse.ArgumentParser()
ap.add_argument("--configs", nargs="+", default=CONFIGS)
ap.add_argument("--out", default="results/robust")
ap.add_argument("--n-boot", type=int, default=10000)
args = ap.parse_args()
B = args.n_boot


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


def test(parts, shift=0.0, intervals=True):
    """parts: [(sign, values, cluster labels)], NaN already dropped; estimate = sum(sign * mean) - shift.
    Record bootstrap: every part resampled on its own; cluster bootstrap: one draw of clusters for all."""
    n = [len(v) for _, v, _ in parts]
    res = {"n": n if len(n) > 1 else n[0], "n_clusters": len({c for _, _, cl in parts for c in cl})}
    if min(n) == 0:
        return {"estimate": None, **res}
    res["estimate"] = sum(s * v.mean() for s, v, _ in parts) - shift
    if intervals:
        rec = sum(s * EM.boot_means([(v, list(range(len(v))))], B, seed=j)[0] for j, (s, v, _) in enumerate(parts))
        clu = EM.boot_means([(v, cl) for _, v, cl in parts], B)
        clu = sum(s * clu[j] for j, (s, _, _) in enumerate(parts))
        res.update(record=EM.boot_summary(rec - shift), cluster=EM.boot_summary(clu - shift))
    return res


out = {"configs": args.configs, "n_boot": B, "min_n": MIN_N, "ks": KS, "models": {}}
family = []  # (model, hypothesis id) of every pre-registered test with an estimate
for path in args.configs:
    cfg = load_config(path)
    name = Path(path).stem
    root = entity_dir(cfg)
    prereg = json.loads(Path(cfg["entity_prereg"]).read_text())
    if not (root / "pressure_test" / "ranks.npz").exists():
        print(f"{name}: no pressure_test, skipped")
        continue
    records, arrays, _ = load_run(root / "pressure_test")
    band, k, lens, fmt = prereg["band"], prereg["k"], prereg["lens"], prereg["format"]
    n = prereg.get("primary_n_peers") or cfg["entity"]["primary_n_peers"]
    assert "logit" in arrays, f"{name}: no logit-lens readout"
    bridges = {f.uid: (norm(f.e2), norm(p.e2)) for f, p, _, _ in load_items(cfg, "test")}
    orig = lambda i: bridges[records[i]["uid"]][0]
    peer = lambda i: bridges[records[i]["uid"]][1]

    def rows(**kw):
        return [i for i, r in enumerate(records) if r["format"] == fmt and all(r.get(a) == b for a, b in kw.items())
                and (r["round"] == 0 or r.get("n_peers") == n)]

    def val(idx, contrast, kind, ln=lens, kk=k):
        """Per-record band mean aligned with idx: kind hit (the pre-registered readout), logrank or win."""
        if kind == "hit":
            return EM.band_rows(arrays[ln], idx, contrast, "last", kk, band)
        return EM.rank_rows(arrays[ln], idx, contrast, "last", band, kind)

    def part(sign, idx, contrast, kind, cluster, ln=lens):
        v = val(idx, contrast, kind, ln)
        ok = ~np.isnan(v) if len(v) else np.zeros(0, bool)
        return sign, v[ok], [cluster(i) for i, o in zip(idx, ok) if o]

    cell = {c: rows(**kw) for c, kw in CELLS.items()}
    flip = lambda style: rows(round=1, condition="pressure", style=style, outcome="peer")
    m = {"meta": {"lens": lens, "band": band, "k": k, "format": fmt, "n_peers": n,
                  "cells_n": {c: len(ix) for c, ix in cell.items()}}}

    # pre-registered tests with record and cluster bootstraps
    def h4_part(kind="hit"):
        idx = flip("answer")
        d = val(idx, "orig_bridge", kind) - val(idx, "orig_bridge", kind, "logit") if idx else np.zeros(0)
        ok = ~np.isnan(d)
        return 1, d[ok], [orig(i) for i, o in zip(idx, ok) if o]

    m["tests"] = {"H1": test([part(1, flip("answer"), "orig_bridge", "hit", orig)]),
                  "H2": test([part(1, flip("answer"), "peer_bridge", "hit", peer),
                              part(-1, cell["mention"], "peer_bridge", "hit", peer)]),
                  "H3": test([part(1, flip("hop2"), "orig_bridge", "hit", orig),
                              part(-1, flip("hop1"), "orig_bridge", "hit", orig)]),
                  "H4": test([h4_part()])}
    family += [(name, h) for h, t in m["tests"].items() if t.get("estimate") is not None]

    # threshold-free readouts
    tf = {"cells": {}, "tests": {}}
    for c, idx in cell.items():
        for ln in arrays:
            for con in ("orig_bridge", "peer_bridge"):
                cl = orig if con == "orig_bridge" else peer
                tf["cells"][f"{c}/{ln}/{con}"] = {"logrank": test([part(1, idx, con, "logrank", cl, ln)]),
                                                  "win": test([part(1, idx, con, "win", cl, ln)], shift=0.5)}
    for kind in ("logrank", "win"):
        tf["tests"][f"H1/{kind}"] = test([part(1, flip("answer"), "orig_bridge", kind, orig)],
                                         shift=0.5 if kind == "win" else 0.0)
        tf["tests"][f"H2/{kind}"] = test([part(1, flip("answer"), "peer_bridge", kind, peer),
                                          part(-1, cell["mention"], "peer_bridge", kind, peer)])
        tf["tests"][f"H4/{kind}"] = test([h4_part(kind)])
    m["threshold_free"] = tf

    m["k_curve"] = {f"{c}/{ln}/{con}": [float(np.nanmean(v)) if len(v) and not np.isnan(v).all() else None
                                         for v in (val(idx, con, "hit", ln, kk) for kk in KS)]
                    for c, idx in cell.items() for ln in arrays for con in ("orig_bridge", "peer_bridge")}

    # the same items in two conditions
    by_uid = {c: {records[i]["uid"]: i for i in idx} for c, idx in cell.items()}
    wi = {}
    for a, b, con in (("flipped", "agree", "orig_bridge"), ("flipped", "round0", "orig_bridge"),
                      ("held", "agree", "orig_bridge"), ("flipped", "mention", "peer_bridge")):
        common = sorted(set(by_uid[a]) & set(by_uid[b]))
        ia, ib = [by_uid[a][u] for u in common], [by_uid[b][u] for u in common]
        for kind in ("hit", "logrank"):
            va, vb = val(ia, con, kind), val(ib, con, kind)
            ok = ~np.isnan(va) & ~np.isnan(vb) if common else np.zeros(0, bool)
            cl = [orig(i) if con == "orig_bridge" else peer(i) for i, o in zip(ia, ok) if o]
            res = test([(1, (va - vb)[ok], cl)])
            if ok.sum():
                uids = [u for u, o in zip(common, ok) if o]
                bm = EM.boot_means([(va[ok], uids), (vb[ok], uids)], B)  # items resampled, pairs kept
                with np.errstate(invalid="ignore", divide="ignore"):
                    r = np.where(bm[1] != 0, bm[0] / bm[1], np.nan)
                ci = EM.boot_summary(r)["ci"]
                res.update(mean_a=va[ok].mean(), mean_b=vb[ok].mean(),
                           ratio={"estimate": va[ok].mean() / vb[ok].mean() if vb[ok].mean() else None, "ci": ci})
            wi[f"{a}-{b}/{con}/{kind}"] = res
    for c in ("round0", "agree"):  # selection: items that gave in vs items that held, before any pressure
        fi = [by_uid[c][u] for u in by_uid["flipped"] if u in by_uid[c]]
        hi = [by_uid[c][u] for u in by_uid["held"] if u in by_uid[c]]
        for kind in ("hit", "logrank"):
            wi[f"selection/{c}/{kind}"] = test([part(1, fi, "orig_bridge", kind, orig),
                                                part(-1, hi, "orig_bridge", kind, orig)])
    m["within_item"] = wi

    # categories
    cats = prereg.get("categories") or sorted({r["category"] for r in records})
    per = {}
    for cat in cats:
        in_cat = lambda idx: [i for i in idx if records[i]["category"] == cat]
        f_, me = in_cat(flip("answer")), in_cat(cell["mention"])
        pressed = in_cat(rows(round=1, condition="pressure", style="answer"))
        big = len(f_) >= MIN_N and len(me) >= MIN_N
        per[cat] = {"n_pressed": len(pressed), "n_flipped": len(f_), "n_held": len(in_cat(cell["held"])),
                    "H1": test([part(1, f_, "orig_bridge", "hit", orig)], intervals=len(f_) >= MIN_N),
                    "H1/logrank": test([part(1, f_, "orig_bridge", "logrank", orig)], intervals=len(f_) >= MIN_N),
                    "H2": test([part(1, f_, "peer_bridge", "hit", peer), part(-1, me, "peer_bridge", "hit", peer)],
                               intervals=big)}
    m["categories"] = per
    out["models"][name] = m

    t = m["tests"]
    print(f"\n===== {name} (lens {lens}, band {band}, k {k}, {n} peers)")
    for h, r in t.items():
        if r.get("estimate") is None:
            print(f"{h}: no estimate (n {r['n']})")
            continue
        print(f"{h}: {r['estimate']:.3f} n={r['n']} clusters={r['n_clusters']}  record CI "
              f"[{r['record']['ci'][0]:.3f}, {r['record']['ci'][1]:.3f}] p={r['record']['p']:.4f}  cluster CI "
              f"[{r['cluster']['ci'][0]:.3f}, {r['cluster']['ci'][1]:.3f}] p={r['cluster']['p']:.4f}")
    for key, r in tf["tests"].items():
        if r.get("estimate") is not None:
            print(f"threshold-free {key}: {r['estimate']:.3f} cluster CI [{r['cluster']['ci'][0]:.3f}, "
                  f"{r['cluster']['ci'][1]:.3f}] n={r['n']}")
    for key, r in wi.items():
        if r.get("estimate") is not None:
            extra = f", ratio {r['ratio']['estimate']:.3f}" if r.get("ratio", {}).get("estimate") is not None else ""
            print(f"within item {key}: {r['estimate']:.3f} cluster CI [{r['cluster']['ci'][0]:.3f}, "
                  f"{r['cluster']['ci'][1]:.3f}] n={r['n']}{extra}")
    for cat, r in per.items():
        h1, h2 = r["H1"].get("estimate"), r["H2"].get("estimate")
        print(f"category {cat}: flipped {r['n_flipped']} of {r['n_pressed']}, H1 "
              f"{'-' if h1 is None else f'{h1:.3f}'}, H2 {'-' if h2 is None else f'{h2:.3f}'}")

for kind in ("cluster", "record"):
    ps = [out["models"][mn]["tests"][h][kind]["p"] for mn, h in family]
    for (mn, h), p_adj in zip(family, EM.holm(ps) if ps else []):
        t = out["models"][mn]["tests"][h]
        t[kind]["p_holm"] = p_adj
        t[kind]["holm_supported"] = bool(p_adj < 0.05 and t["estimate"] > 0)
out["holm"] = {"family": [f"{mn}/{h}" for mn, h in family], "alpha": 0.05,
               "supported_cluster": [f"{mn}/{h}" for mn, h in family
                                     if out["models"][mn]["tests"][h]["cluster"]["holm_supported"]],
               "supported_record": [f"{mn}/{h}" for mn, h in family
                                    if out["models"][mn]["tests"][h]["record"]["holm_supported"]]}
print(f"\nHolm over {len(family)} pre-registered tests (alpha 0.05), cluster p: "
      f"{len(out['holm']['supported_cluster'])} supported; not supported: "
      f"{[f for f in out['holm']['family'] if f not in out['holm']['supported_cluster']]}")
dst = Path(args.out)
dst.mkdir(parents=True, exist_ok=True)
(dst / "summary.json").write_text(json.dumps(clean(out), indent=1) + "\n")
print(f"-> {dst / 'summary.json'}")
