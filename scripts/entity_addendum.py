"""Entity study, follow-ups: fix a pre-registration addendum from dev-split runs only.

Rules and thresholds are in entity.addenda of the config (declared before the dev runs); each
addendum is written once (prereg/<name>_entity_<part>.json, refused if it exists) and must be
committed and pushed before `entity_run.py --split test --addendum <part>`. The main entity
prereg is not touched; its hypotheses and results stand.

--part rounds   later rounds without the plain-transcript degeneration. Dev runs:
                pressure_dev_rounds_repeat (the pre-registered protocol, reference) and
                pressure_dev_rounds_<variant> per candidate (entity_run --later <variant>).
                A variant passes if, in every pressure / baseline cell (entity.addenda.rounds.n_peers
                peers), the share of "other" answers at rounds 2..round is <= max_other, and its
                round-1 answers agree with the reference run's in >= min_round1_agreement of the
                agents (round 1 is the same prompt in every variant). The passing variant with the
                lowest worst-cell "other" share is chosen (ties: config order).
                Hypotheses R1 / R2: H1 / H2 at the last round, agents that gave in every round.
--part bridges  non-place bridges (entity.addenda.bridges.bridge_types, e.g. person). Dev runs:
                e0_dev_bridges_<mode> per probe mode (entity_run --stage e0 --bridge-types ...
                --probe-mode <mode> --extended). For each probe mode x position (last, mention,
                span): band = layers where round-0 correct records have orig_bridge - ctrl_bridge
                >= band_threshold (hit@k, main lens, last layer excluded); categories with band mean
                >= category_threshold and n >= min_category_n; F1: band and categories non-empty,
                n >= min_n; F2 (adoption, as in the main prereg): hypothetical peer_bridge -
                mention_bridge peer_bridge >= min_adoption_gap and agree orig_bridge >= min_persistence
                x round-0 orig_bridge, each n >= min_n. Among passing rules the one covering the most
                dev records is chosen (ties: higher round-0 band mean).
                Hypotheses B1 / B2 / B4: H1 / H2 / H4 at round 1 under that readout, pooled over the
                addendum's n_peers, one value per item.
--part own      is the retained bridge recomputed from the agent's own round-0 answer, or kept
                without it? Variants (entity.addenda.own.variants): hidden (the agent's round-0
                statement replaced by a placeholder) and absent (no round-0 statement in context: the
                agent answers first, after the peers). The readout (lens, k, band, last token),
                format and categories are the main prereg's; nothing is selected. Dev runs:
                pressure_dev_own_<variant> (entity_run --own <variant>). A variant passes if the
                share of "other" answers is <= max_other in every cell of the pilot and the agree
                cell still reads the original bridge (n >= min_agree_n, CI lower bound > 0). The
                addendum registers the passing variants; without a passing hidden variant none is
                written (exit code 3). Hypotheses (round 1, n_peers peers, answer pressure, agents that gave in):
                O1 hidden: original bridge above control; O2 hidden minus absent > 0; O3 absent:
                original bridge above control; O4 hidden: J-lens minus logit lens > 0 (paired). A
                test cell with fewer than min_test_n records gives no verdict.

    python scripts/entity_addendum.py --config configs/qwen35_4b_entity.yaml --part rounds --dry-run
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
from silent_dissent.entity_io import addendum_path, entity_dir, load_run

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--part", choices=["rounds", "bridges", "own"], required=True)
ap.add_argument("--dry-run", action="store_true", help="print the checks, write nothing")
args = ap.parse_args()

cfg = load_config(args.config)
ec = cfg["entity"]
main_path = Path(cfg["entity_prereg"])
if not main_path.is_file():
    raise SystemExit(f"{main_path} missing: the addenda extend the main entity prereg")
main = json.loads(main_path.read_text())
ac = ec["addenda"][args.part]
root = entity_dir(cfg)
lens = cfg["lens"]["name"]


def rounds_part():
    n, last = ac["n_peers"], ac["round"]
    ref_recs, _, _ = load_run(root / "pressure_dev_rounds_repeat")

    def check(recs):
        cells: dict[tuple, list[str]] = {}
        for r in recs:
            if r.get("n_peers") == n and 2 <= r["round"] <= last and r["condition"] != "solo":
                cells.setdefault((r["condition"], r.get("style") or "-", r["round"]), []).append(r["outcome"])
        other = {f"{c}/{s}/round {t}": round(float(np.mean([o == "other" for o in v])), 3)
                 for (c, s, t), v in sorted(cells.items())}
        return other, (max(other.values()) if other else None)

    ref_r1 = {EM.state_key(r): r["answer"] for r in ref_recs if r["round"] == 1 and r.get("n_peers") == n}
    checks = {}
    ref_other, ref_worst = check(ref_recs)
    checks["repeat"] = {"worst_other": ref_worst, "other": ref_other, "reference": True}
    print(f"repeat (reference): worst cell 'other' share at rounds 2-{last}: {ref_worst}")
    passing = []
    for v in ac["variants"]:
        recs, _, info = load_run(root / f"pressure_dev_rounds_{v}")
        assert info.get("later") == v, f"pressure_dev_rounds_{v} was run with later={info.get('later')}"
        other, worst = check(recs)
        r1 = {EM.state_key(r): r["answer"] for r in recs if r["round"] == 1 and r.get("n_peers") == n}
        common = set(r1) & set(ref_r1)
        agree = float(np.mean([r1[s] == ref_r1[s] for s in common])) if common else None
        ok = (worst is not None and worst <= ac["max_other"] and agree is not None
              and agree >= ac["min_round1_agreement"])
        checks[v] = {"worst_other": worst, "round1_agreement": agree, "n_agents": len(common), "other": other,
                     "pass": ok}
        print(f"{v}: worst cell 'other' share {worst}, round-1 agreement with repeat {agree} "
              f"({len(common)} agents) -> {'pass' if ok else 'fail'}")
        for cell, share in other.items():
            if share > ac["max_other"]:
                print(f"    {cell}: {share}")
        if ok:
            passing.append((worst, ac["variants"].index(v), v))
    if not passing:
        raise SystemExit("no later-round variant passes on the dev split: no addendum written (use round 1 only)")
    chosen = min(passing)[2]
    print(f"CHOSEN: {chosen}")
    return {
        "study": "entity-answer silent dissent, addendum: later rounds",
        "later": chosen, "format": main["format"], "styles": ec["styles"], "baselines": ["agree", "mention", "mention_bridge"],
        "n_peers": n, "round": last, "categories": main["categories"],
        "readout": "as in the main prereg (lens, k, band, last token)", "lens": main["lens"], "k": main["k"],
        "band": main["band"],
        "rule": f"candidates {ac['variants']} (repeat = reference); pass: 'other' share <= {ac['max_other']} in every "
                f"pressure / baseline cell at rounds 2-{last} with {n} peers, and round-1 answers equal to the "
                f"reference run's for >= {ac['min_round1_agreement']} of agents; choose the lowest worst-cell share",
        "hypotheses": [
            {"id": "R1", "name": "latent premise retention persists",
             "records": f"pressure, style answer, {n} peers, round {last}; agents that gave the peers' answer in "
                        f"every round 1-{last}", "statistic": "orig_bridge - ctrl_bridge (band mean per record)",
             "prediction": "> 0", "test": "95% bootstrap CI over records excludes 0"},
            {"id": "R2", "name": "implied premise beyond mention persists",
             "records": f"the R1 records vs condition mention ({n} peers, round {last}), answer kept",
             "statistic": "peer_bridge - ctrl_bridge, difference of means", "prediction": "R1 records > mention",
             "test": "95% bootstrap CI of the difference excludes 0"},
        ],
        "exploratory": [f"change of the R1 statistic from round 1 to round {last} (same agents)",
                        "late flips (rounds 2-3)", "held agents", "hop1 / hop2 styles in later rounds"],
        "test_run": "scripts/entity_run.py --stage pressure --split test --addendum rounds "
                    "(-> pressure_test_rounds); round 1 there must reproduce pressure_test",
        "dev_checks": checks,
        "dev_runs": [str(root / f"pressure_dev_rounds_{v}") for v in ["repeat"] + ac["variants"]],
    }


def bridges_part():
    thr, k = ac["band_threshold"], main["k"]
    any_outcome = not ec.get("require_correct_round0", True)  # debug configs only: a random model is never right
    candidates, summary = [], {}
    for mode in ac["probe_modes"]:
        recs, arrs, info = load_run(root / f"e0_dev_bridges_{mode}")
        assert info.get("probe_mode") == mode and info.get("extended"), f"e0_dev_bridges_{mode}: wrong run settings"
        ranks = arrs[lens]
        n_layers = ranks.shape[2]
        rows = lambda **kw: [i for i, r in enumerate(recs) if r["format"] == main["format"]
                             and all(r.get(a) == b for a, b in kw.items() if not (any_outcome and a == "outcome"))]
        right = rows(condition="solo", outcome="original")
        for pos in ac["positions"]:
            curve = EM.curve(ranks, right, "orig_bridge", pos, k)
            band = [l for l in EM.select_band(curve, thr) if l < n_layers - 1]
            bm = lambda idx, c: EM.band_summary(ranks, idx, c, pos, k, band) if band else {"n": 0, "mean": np.nan}
            cats = []
            for cat in sorted({recs[i]["category"] for i in right}):
                s = bm([i for i in right if recs[i]["category"] == cat], "orig_bridge")
                if s["n"] >= ac["min_category_n"] and s["mean"] >= ac["category_threshold"]:
                    cats.append(cat)
            R = lambda **kw: [i for i in rows(**kw) if recs[i]["category"] in cats]
            r0 = bm(R(condition="solo", outcome="original"), "orig_bridge")
            hyp = bm(R(condition="hypothetical", outcome="peer"), "peer_bridge")
            men = bm(R(condition="mention_bridge", outcome="original"), "peer_bridge")
            agr = bm(R(condition="agree", outcome="original"), "orig_bridge")
            ok_n = lambda s: s["n"] >= ac["min_n"] and not np.isnan(s["mean"])
            f1 = bool(band) and bool(cats) and ok_n(r0)
            gap, keep = ac["min_adoption_gap"], ac["min_persistence"]  # None skips a check (debug configs only)
            f2 = (f1 and all(ok_n(s) for s in (hyp, men, agr))
                  and (gap is None or hyp["mean"] - men["mean"] >= gap)
                  and (keep is None or agr["mean"] >= keep * r0["mean"]))
            fmt_s = lambda s: {"n": s["n"], "mean": None if np.isnan(s["mean"]) else round(float(s["mean"]), 3)}
            key = f"{mode}/{pos}"
            summary[key] = {"band": band, "curve": [round(float(x), 3) for x in curve], "categories": cats,
                            "round0_orig_bridge": fmt_s(r0), "hypothetical_peer_bridge": fmt_s(hyp),
                            "mention_bridge_peer_bridge": fmt_s(men), "agree_orig_bridge": fmt_s(agr),
                            "F1": f1, "F2": f2}
            print(f"{key:18s} band {band}, categories {cats}, round 0 {fmt_s(r0)}, adoption {fmt_s(hyp)} vs "
                  f"{fmt_s(men)}, agree {fmt_s(agr)} -> F1 {'pass' if f1 else 'fail'}, F2 {'pass' if f2 else 'fail'}")
            if f1 and f2:
                candidates.append((-r0["n"], -r0["mean"], mode, pos))
    if not candidates:
        raise SystemExit("no readout reads the non-place bridges on the dev split: no addendum written")
    _, _, mode, pos = min(candidates)
    chosen = summary[f"{mode}/{pos}"]
    print(f"CHOSEN: probe mode {mode}, position {pos}, band {chosen['band']}, categories {chosen['categories']}")
    return {
        "study": "entity-answer silent dissent, addendum: non-place bridges",
        "bridge_types": ac["bridge_types"], "categories": chosen["categories"], "format": main["format"],
        "lens": lens, "comparison_lens": "logit", "k": k, "probe_mode": mode, "position": pos, "band": chosen["band"],
        "styles": ac["styles"], "baselines": ac["baselines"], "n_peers": ac["n_peers"], "round": 1,
        "rule": f"candidates: probe modes {ac['probe_modes']} x positions {ac['positions']}; band = layers where dev "
                f"round-0 correct records have orig_bridge - ctrl_bridge >= {thr} (hit@{k}, {lens}, last layer "
                f"excluded); categories with band mean >= {ac['category_threshold']} and n >= {ac['min_category_n']}; "
                f"F1 and F2 (adoption gap >= {ac['min_adoption_gap']}, persistence >= {ac['min_persistence']}, "
                f"n >= {ac['min_n']}); choose the rule covering the most dev records",
        "positions": {"last": "the answer position", "mention": "last token of the bridge's description in the "
                      "final statement", "span": "minimum rank over the mention's last token .. the last token"},
        "hypotheses": [
            {"id": "B1", "name": "latent premise retention (non-place bridges)",
             "records": f"pressure, style answer, round 1, n_peers in {ac['n_peers']}, flipped; one value per item "
                        "(mean over its records)", "statistic": "orig_bridge - ctrl_bridge (band mean)",
             "prediction": "> 0", "test": "95% bootstrap CI over items excludes 0"},
            {"id": "B2", "name": "implied premise beyond mention (non-place bridges)",
             "records": "the B1 items vs condition mention (round 1, same n_peers, answer kept), per item",
             "statistic": "peer_bridge - ctrl_bridge, difference of means", "prediction": "flipped > mention",
             "test": "95% bootstrap CI of the difference excludes 0"},
            {"id": "B4", "name": "the J-lens reads what the logit lens misses (non-place bridges)",
             "records": "the B1 items", "statistic": "B1 statistic under the J-lens minus the logit lens, paired "
                                                     "over items", "prediction": "> 0",
             "test": "95% bootstrap CI of the paired difference excludes 0"},
        ],
        "exploratory": ["per category", "hop1 style", "held agents", "other positions and probe modes",
                        "bridgeprobe_dev diagnosis"],
        "test_run": "scripts/entity_run.py --stage pressure --split test --addendum bridges (-> pressure_test_bridges)",
        "dev_checks": summary,
        "dev_runs": [str(root / f"e0_dev_bridges_{m}") for m in ac["probe_modes"]],
    }


def own_part():
    n, lens_, k, band, fmt = ac["n_peers"], main["lens"], main["k"], main["band"], main["format"]
    checks, passing = {}, []
    for v in ac["variants"]:
        recs, arrs, info = load_run(root / f"pressure_dev_own_{v}")
        assert info.get("own") == v and info.get("n_peers") == [n] and info.get("rounds") == 1, \
            f"pressure_dev_own_{v}: run with own={info.get('own')}, n_peers={info.get('n_peers')}, rounds={info.get('rounds')}"
        r1 = [i for i, r in enumerate(recs) if r["round"] == 1 and r["format"] == fmt and r.get("n_peers") == n]
        cells: dict[str, list[str]] = {}
        for i in r1:
            r = recs[i]
            cells.setdefault(f"{r['condition']}/{r.get('style') or '-'}", []).append(r["outcome"])
        other = {c: round(float(np.mean([o == "other" for o in v_])), 3) for c, v_ in sorted(cells.items())}
        gave = {c: {"peer": round(float(np.mean([o == "peer" for o in v_])), 3), "n": len(v_)}
                for c, v_ in sorted(cells.items()) if c.startswith("pressure/")}
        agree = EM.band_summary(arrs[lens_], [i for i in r1 if recs[i]["condition"] == "agree"
                                              and recs[i]["outcome"] == "original"], "orig_bridge", "last", k, band)
        worst = max(other.values()) if other else None
        min_n = ac["min_agree_n"]  # None skips the agree check (debug configs only)
        ok_agree = min_n is None or (agree["n"] >= min_n and agree["ci_lo"] > 0)
        ok = worst is not None and worst <= ac["max_other"] and ok_agree
        r3 = lambda x: None if x is None or np.isnan(x) else round(float(x), 3)
        checks[v] = {"other": other, "worst_other": worst, "gave_in": gave,
                     "agree_orig_bridge": {"n": agree["n"], **{x: r3(agree[x]) for x in ("mean", "ci_lo", "ci_hi")}},
                     "items": info.get("n_items"), "pass": ok}
        print(f"{v}: worst cell 'other' share {worst}, agree original bridge {checks[v]['agree_orig_bridge']}, "
              f"gave in {gave} -> {'pass' if ok else 'fail'}")
        if ok:
            passing.append(v)
    if "hidden" not in passing:  # exit code 3: a registered outcome, not a failure (scripts/entity_own.sh)
        print("NO ADDENDUM: the hidden variant fails its dev checks, no addendum written")
        raise SystemExit(3)
    records = f"condition pressure, style answer, {n} peers, round 1, outcome peer (gave in)"
    hyps = [{"id": "O1", "name": "retention without the agent's own round-0 answer in context",
             "records": f"variant hidden; {records}", "statistic": "orig_bridge - ctrl_bridge (band mean per record)",
             "prediction": "> 0", "test": "95% bootstrap CI over records excludes 0"}]
    if "absent" in passing:
        hyps += [{"id": "O2", "name": "commitment: retention beyond recomputation without an earlier answer",
                  "records": f"the O1 records vs variant absent; {records}",
                  "statistic": "orig_bridge - ctrl_bridge, difference of means (hidden - absent)", "prediction": "> 0",
                  "test": "95% bootstrap CI of the difference excludes 0"},
                 {"id": "O3", "name": "the original bridge without any earlier answer (recomputation baseline)",
                  "records": f"variant absent; {records}", "statistic": "orig_bridge - ctrl_bridge (band mean per record)",
                  "prediction": "> 0", "test": "95% bootstrap CI over records excludes 0"}]
    hyps += [{"id": "O4", "name": "the J-lens reads what the logit lens misses (hidden)", "records": "the O1 records",
              "statistic": "O1 statistic under the J-lens minus under the logit lens, paired over records",
              "prediction": "> 0", "test": "95% bootstrap CI of the paired difference excludes 0"}]
    print(f"REGISTERED VARIANTS: {passing}")
    return {
        "study": "entity-answer silent dissent, addendum: the agent's own round-0 answer hidden or absent",
        "variants": passing, "variants_declared": ac["variants"], "styles": ac["styles"], "baselines": ac["baselines"],
        "n_peers": n, "round": 1, "format": fmt, "categories": main["categories"],
        "readout": "as in the main prereg (lens, k, band, last token)", "lens": lens_, "comparison_lens": "logit",
        "k": k, "band": band, "min_test_n": ac["min_test_n"],
        "variant_definitions": {
            "hidden": "the agent's round-0 statement is replaced by '(answer submitted)'; peers and moderator as in "
                      "the main run",
            "absent": "the agent's round-0 statement is left out; the peers speak and the moderator asks it to "
                      "'complete the statement' (not 'again'), so round 1 is its first answer in context; round 0 "
                      "is still run alone to select the items answered correctly"},
        "rule": f"variants {ac['variants']}; pass: 'other' share <= {ac['max_other']} in every pilot cell and the "
                f"agree cell's original-bridge readout with n >= {ac['min_agree_n']} and CI lower bound > 0; "
                "variants that pass are registered, and no addendum without the hidden variant",
        "hypotheses": hyps,
        "no_verdict": f"a hypothesis whose cell has fewer than {ac['min_test_n']} records gets no verdict",
        "exploratory": ["give-in rates per variant vs the main run (visible answer)",
                        "paired per item: hidden vs visible (pressure_test) and hidden vs absent, items that gave in "
                        "under both", "peer bridge vs the mention baseline within each variant (H2 analogue)",
                        "hop1 style", "held agents", "extra lenses"],
        "test_run": "scripts/entity_run.py --stage pressure --split test --addendum own (-> pressure_test_own)",
        "dev_checks": checks,
        "dev_runs": [str(root / f"pressure_dev_own_{v}") for v in ac["variants"]],
    }


add = {"rounds": rounds_part, "bridges": bridges_part, "own": own_part}[args.part]()
if args.dry_run:
    raise SystemExit
path = addendum_path(cfg, args.part)
if path.exists():
    raise SystemExit(f"{path} already exists; an addendum is fixed once")
try:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
except Exception:
    commit = None
add.update({"model": cfg["model"]["name"], "main_prereg": str(main_path),
            "main_prereg_sha256": hashlib.sha256(main_path.read_bytes()).hexdigest(),
            "created": datetime.datetime.now().isoformat(timespec="seconds"), "code_commit": commit})
path.write_text(json.dumps(add, indent=2) + "\n")
print(f"-> {path}  (review, then commit and push before its test run)")
