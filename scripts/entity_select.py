"""Entity study, step 3: fix format, readout band and hypotheses from the dev-split E0 run only.

Checks (the analysis values are "orig_bridge - ctrl_bridge" etc., hit@k at the last token,
configured lens; see silent_dissent/entity_metrics.py):
  F1  round-0 accuracy in the format >= entity.select.min_accuracy, and a non-empty band: layers
      where correctly answered round-0 records have orig_bridge - ctrl_bridge >= band_threshold
  F2  retention records that state the sibling's answer (own answer visible and hidden, each
      n >= min_n): orig_bridge - ctrl_bridge averaged over the band >= band_threshold
  F3  mention baselines (reported): peer_bridge in mention / mention_bridge records that keep
      the original answer; plus the hypothetical (adopted premise) and instructed readouts
Options (entity.select), all off by default:
  exclude_last_layer   drop the last layer from the band (it is the model output for every lens)
  category_threshold,  keep only categories whose round-0 band-mean orig_bridge - ctrl_bridge
  min_category_n       reaches the threshold (n >= min_category_n); later runs use only those
  f2: adoption         F2 = hypothetical peer_bridge - mention_bridge peer_bridge >= min_adoption_gap
                       (an adopted premise stands out from a merely mentioned one) and agree
                       orig_bridge >= min_persistence x round-0 orig_bridge (it persists in dialogue);
                       with min_adoption_ratio, the first part is hypothetical >= ratio x mention_bridge
  band_rule: dialogue  instead of band_threshold / category_threshold: the round-0 latent window is the
                       span of layers where the lens beats the logit lens by >= latent_gap (round-0
                       correct records); categories whose window mean of that gap is >= latent_gap
                       (n >= min_category_n); the band is the layers where agree records of those
                       categories read the bridge >= dialogue_min and >= latent_gap above the logit
                       lens (the last layer is always excluded). Selects where the latent premise is
                       readable in dialogue, which is what the hypotheses test.
  supersedes           path of an earlier prereg of the same model whose test split was never run;
                       its sha256 and supersedes_reason are recorded
  notes                free-text dev adjustments copied into the prereg
The plain transcript is used if it passes F1 and F2, otherwise the chat format; if neither
passes, no prereg is written. The prereg is refused if the file exists (fixed once, like
prereg/qwen35_4b.json); commit and push it before running the test split.

    python scripts/entity_select.py --config configs/qwen35_4b_entity.yaml
"""
import argparse
import datetime
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from silent_dissent import entity_metrics as EM
from silent_dissent.config import load_config
from silent_dissent.entity_io import entity_dir, load_run

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--run", default="e0_dev", help="E0 run directory under <out_dir>/entity")
ap.add_argument("--dry-run", action="store_true", help="print the checks, write nothing")
args = ap.parse_args()

cfg = load_config(args.config)
ec, sc = cfg["entity"], cfg["entity"]["select"]
k, thr, lens = sc["k"], sc["band_threshold"], cfg["lens"]["name"]
records, arrays, info = load_run(entity_dir(cfg) / args.run)
ranks = arrays[lens]
rows = lambda **kw: [i for i, r in enumerate(records) if all(r.get(a) == b for a, b in kw.items())]


def band_mean(idx, contrast, band, arr=None):
    s = EM.band_summary(ranks if arr is None else arr, idx, contrast, "last", k, band)
    return {"n": s["n"], "mean": round(s["mean"], 3) if s["n"] else None,
            "ci": [round(s["ci_lo"], 3), round(s["ci_hi"], 3)] if s["n"] else None}


summary, passing = {}, []
n_layers = ranks.shape[2]
f2_mode = sc.get("f2", "retention")
band_rule = sc.get("band_rule", "round0")
cmp = arrays["logit"] if band_rule == "dialogue" else None
for fmt in info["formats"]:
    solo = rows(format=fmt, condition="solo")
    right_all = rows(format=fmt, condition="solo", outcome="original")
    acc = len(right_all) / max(len(solo), 1)
    solo_curve = EM.curve(ranks, right_all, "orig_bridge", "last", k)
    extra = {}
    if band_rule == "round0":
        band = EM.select_band(solo_curve, thr)
        if sc.get("exclude_last_layer"):  # the last layer is the model output itself: every lens agrees there
            band = [l for l in band if l < n_layers - 1]
        cats = None
        if band and sc.get("category_threshold") is not None:  # categories whose bridge is readable in round 0
            cats = []
            for cat in sorted({records[i]["category"] for i in right_all}):
                m = band_mean([i for i in right_all if records[i]["category"] == cat], "orig_bridge", band)
                if m["n"] >= sc["min_category_n"] and m["mean"] is not None and m["mean"] >= sc["category_threshold"]:
                    cats.append(cat)
    else:  # "dialogue": latent (lens - logit lens) in round 0 picks the categories, the agree dialogue the band
        gap, dmin, inner = sc["latent_gap"], sc["dialogue_min"], range(n_layers - 1)  # last layer excluded
        gap0 = np.nan_to_num(solo_curve - EM.curve(cmp, right_all, "orig_bridge", "last", k), nan=-1)
        sel = [l for l in inner if gap0[l] >= gap]
        window = list(range(min(sel), max(sel) + 1)) if sel else []
        cats = []
        for cat in sorted({records[i]["category"] for i in right_all}) if window else []:
            idx = [i for i in right_all if records[i]["category"] == cat]
            mj, ml = band_mean(idx, "orig_bridge", window), band_mean(idx, "orig_bridge", window, cmp)
            if mj["n"] >= sc["min_category_n"] and mj["mean"] is not None and mj["mean"] - ml["mean"] >= gap:
                cats.append(cat)
        agr_idx = [i for i in rows(format=fmt, condition="agree", outcome="original") if records[i]["category"] in cats]
        cj = np.nan_to_num(EM.curve(ranks, agr_idx, "orig_bridge", "last", k), nan=-1)
        cl = np.nan_to_num(EM.curve(cmp, agr_idx, "orig_bridge", "last", k), nan=0)
        band = [l for l in inner if cj[l] >= dmin and cj[l] - cl[l] >= gap] if agr_idx else []
        extra = {"round0_latent_window": window, "agree_orig_bridge_curve": [round(float(x), 3) for x in cj],
                 "agree_orig_bridge_curve_logit": [round(float(x), 3) for x in cl]}
    R = lambda **kw: [i for i in rows(format=fmt, **kw) if cats is None or records[i]["category"] in cats]
    ret_out = {} if sc.get("retention_outcome") == "any" else {"outcome": "sibling"}  # "any": debug configs only
    ret = R(condition="retention", hidden_own=False, **ret_out)
    ret_h = R(condition="retention", hidden_own=True, **ret_out)
    bm = lambda idx, c: band_mean(idx, c, band) if band else None
    s = {
        "round0_accuracy": round(acc, 3), "n_round0": len(solo),
        "round0_orig_bridge_curve": [round(float(x), 3) for x in solo_curve],
        "band": band, "categories": cats,
        "round0_orig_bridge": bm(R(condition="solo", outcome="original"), "orig_bridge"),
        "retention": bm(ret, "orig_bridge"), "retention_hidden": bm(ret_h, "orig_bridge"),
        "agree_orig_bridge": bm(R(condition="agree", outcome="original"), "orig_bridge"),
        "mention_peer_bridge": bm(R(condition="mention", outcome="original"), "peer_bridge"),
        "mention_bridge_peer_bridge": bm(R(condition="mention_bridge", outcome="original"), "peer_bridge"),
        "hypothetical_peer_bridge": bm(R(condition="hypothetical", outcome="peer"), "peer_bridge"),
        "instructed_orig_bridge": bm(R(condition="instructed", outcome="peer"), "orig_bridge"),
        "compliance": {c: float(round(np.mean([records[i]["outcome"] == o for i in R(condition=c)]), 3))
                       if R(condition=c) else None
                       for c, o in (("instructed", "peer"), ("hypothetical", "peer"), ("retention", "sibling"))},
        **extra,
    }
    f1 = acc >= sc["min_accuracy"] and bool(band) and (cats is None or len(cats) > 0)
    ok = lambda x: bool(x) and x["n"] >= sc["min_n"] and x["mean"] is not None
    if f2_mode == "retention":
        f2 = bool(band) and all(ok(x) and x["mean"] >= thr for x in (s["retention"], s["retention_hidden"]))
    else:  # "adoption": an adopted premise must stand out from a merely mentioned one, and persist when agreed
        hyp, men, agr, r0 = (s["hypothetical_peer_bridge"], s["mention_bridge_peer_bridge"], s["agree_orig_bridge"],
                             s["round0_orig_bridge"])
        f2 = bool(band) and all(ok(x) for x in (hyp, men, agr, r0))
        if f2 and sc.get("min_adoption_ratio") is not None:  # adopted >= ratio x merely mentioned
            f2 = hyp["mean"] > 0 and hyp["mean"] >= sc["min_adoption_ratio"] * men["mean"]
        elif f2:
            f2 = hyp["mean"] - men["mean"] >= sc["min_adoption_gap"]
        f2 = f2 and agr["mean"] >= sc["min_persistence"] * r0["mean"]
    s.update({"F1": f1, "F2": f2, "F2_mode": f2_mode})
    summary[fmt] = s
    print(f"[{fmt}] accuracy {acc:.3f}, band {band}, F1 {'pass' if f1 else 'fail'}, F2 ({f2_mode}) "
          f"{'pass' if f2 else 'fail'}")
    for key in ("round0_latent_window", "categories", "round0_orig_bridge", "retention", "retention_hidden",
                "agree_orig_bridge", "mention_peer_bridge", "mention_bridge_peer_bridge", "hypothetical_peer_bridge",
                "instructed_orig_bridge", "compliance"):
        if key in s:
            print(f"    {key}: {s[key]}")
    if f1 and f2:
        passing.append(fmt)

fmt = "plain" if "plain" in passing else (passing[0] if passing else None)
if fmt is None:
    raise SystemExit("no format passes F1 and F2 on the dev split: no prereg written")
print(f"FORMAT: {fmt}, BAND: {summary[fmt]['band']}")
if args.dry_run:
    raise SystemExit

path = Path(cfg["entity_prereg"])
if path.exists():
    raise SystemExit(f"{path} already exists; the entity prereg is fixed")
try:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
except Exception:
    commit = None
cell = lambda style: f"condition pressure, style {style}, n_peers {ec['primary_n_peers']}, round 1"
if band_rule == "round0":
    rules = {
        "band_rule": f"layers where correctly answered dev round-0 records have orig_bridge - ctrl_bridge >= {thr} "
                     f"(hit@{k}, {lens}, last token)"
                     + (", excluding the last layer" if sc.get("exclude_last_layer") else ""),
        "category_rule": (f"categories whose dev round-0 records have band-mean orig_bridge - ctrl_bridge >= "
                          f"{sc['category_threshold']} with n >= {sc['min_category_n']}; test items of other "
                          "categories are not run") if summary[fmt]["categories"] is not None else "all categories"}
else:
    rules = {
        "band_rule": f"within the categories below: layers (last excluded) where dev agree records (peers state the "
                     f"agent's answer, answer kept) have orig_bridge - ctrl_bridge >= {sc['dialogue_min']} under "
                     f"{lens} and at least {sc['latent_gap']} above the logit lens (hit@{k}, last token)",
        "category_rule": f"round-0 latent window = layers from the first to the last (last layer excluded) where "
                         f"correctly answered dev round-0 records have {lens} orig_bridge - ctrl_bridge >= "
                         f"{sc['latent_gap']} above the logit lens; categories whose window mean of that difference "
                         f"is >= {sc['latent_gap']} with n >= {sc['min_category_n']}; test items of other categories "
                         "are not run"}
supersedes = None
if sc.get("supersedes"):  # an earlier prereg of the same model that this one replaces (kept unchanged)
    old = Path(sc["supersedes"])
    supersedes = {"path": str(old), "sha256": hashlib.sha256(old.read_bytes()).hexdigest(),
                  "test_split_run": False, "reason": sc.get("supersedes_reason")}
prereg = {
    "study": "entity-answer silent dissent (two-hop facts; the bridge entity is never stated)",
    "model": cfg["model"]["name"], "lens": lens, "comparison_lens": "logit",
    "format": fmt, "band": summary[fmt]["band"], "k": k, "position": "last prompt token (primary)",
    **rules,
    "categories": summary[fmt]["categories"],
    "validation": {"F2_mode": f2_mode, **{key: sc[key] for key in ("min_n", "min_adoption_gap", "min_adoption_ratio",
                                                                   "min_persistence") if key in sc}},
    "dev_adjustments": sc.get("notes", []),
    **({"supersedes": supersedes} if supersedes else {}),
    "readout": "an entity's rank is the minimum full-vocabulary lens rank over the first tokens of its content words, "
               "tokens shared between the item's roles removed; hit@k = rank < k; a contrast is hit(role) - "
               "hit(control role) within a record; the statistic is its mean over the band, per record",
    "test_items": "items_test.jsonl: bridges disjoint from dev; run with scripts/entity_run.py --stage pressure "
                  "--split test",
    "conditions": {"styles": ec["styles"], "n_peers": ec["n_peers"], "rounds": ec["rounds"],
                   "baselines": ["agree", "mention", "mention_bridge"]},
    "primary_n_peers": ec["primary_n_peers"],
    "hypotheses": [
        {"id": "H1", "name": "latent premise retention",
         "records": f"{cell('answer')}; answer = the peers' (flipped)",
         "statistic": "orig_bridge - ctrl_bridge", "prediction": "> 0",
         "test": "95% bootstrap CI over records excludes 0"},
        {"id": "H2", "name": "implied premise beyond mention",
         "records": f"{cell('answer')}, flipped, vs condition mention (same n_peers, round 1), answer kept",
         "statistic": "peer_bridge - ctrl_bridge, difference of means", "prediction": "flipped > mention",
         "test": "95% bootstrap CI of the difference excludes 0"},
        {"id": "H3", "name": "locus of conformity",
         "records": f"{cell('hop2')}, flipped, vs {cell('hop1')}, flipped",
         "statistic": "orig_bridge - ctrl_bridge, difference of means", "prediction": "hop2 > hop1",
         "test": "95% bootstrap CI of the difference excludes 0"},
        {"id": "H4", "name": "the J-lens reads what the logit lens misses",
         "records": "the H1 records", "statistic": "H1 statistic under the J-lens minus under the logit lens, "
                                                   "paired over records", "prediction": "> 0",
         "test": "95% bootstrap CI of the paired difference excludes 0"},
    ],
    "exploratory": ["rounds 2-3"] * (ec["rounds"] > 1) + [
        "n_peers dose-response", "subject position", "per category and per source", "held (non-flipped) records",
        "E3 timeline", "E4 injection", "E5 free debate", "extra lenses"],
    "dev_checks": summary,
    "dev_run": str(entity_dir(cfg) / args.run),
    "created": datetime.datetime.now().isoformat(timespec="seconds"),
    "code_commit": commit,
}
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(prereg, indent=2) + "\n")
print(f"-> {path}  (review, then commit and push before running the test split)")
