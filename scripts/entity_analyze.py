"""Entity study: tables and figures from every run under <out_dir>/entity (no model needed).

  e0_* / pressure_*   outcome shares, band-mean contrasts with CIs (last token and subject token,
                      split by outcome), per-layer curves, a figure per lens; on pressure_test with
                      an entity prereg, the verdicts on H1-H4
  timeline_*          contrast per segment of the last round (own statement, peers, moderator,
                      final), split by style and by whether the agent then gave in; figure
  inject_*            answer shift per direction kind, layer and alpha
  debate_*            group accuracy per round: stated vs latent vs round-0 plurality
  addenda             with prereg/<name>_entity_<part>.json and pressure_test_<part>: R1-R2 (rounds;
                      plus how often round 1 reproduces pressure_test), B1-B4 (bridges) or O1-O4
                      (own; plus own_summary.json: rates, cells, paired comparisons with pressure_test)

The band is the prereg's; before a prereg exists it is selected on e0_dev with the same rule.
Outputs go to <out_dir>/entity/analysis/. With --addendum PART only pressure_test_<PART> is read and
only its files are written (<run>_* tables and figures, hypotheses_<PART>.json, own_summary.json):
the analysis of every other run is left as it is.

    python scripts/entity_analyze.py --config configs/qwen35_4b_entity.yaml
    python scripts/entity_analyze.py --config configs/qwen35_4b_entity.yaml --addendum own
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
import pandas as pd

from silent_dissent import entity_metrics as EM
from silent_dissent.config import load_config
from silent_dissent.entity_io import ADDENDA, entity_dir, load_addendum, load_run

COLORS = {"answer": "#2a78d6", "hop1": "#eb6834", "hop2": "#1baf7a", "mention": "#888888",
          "mention_bridge": "#b0b0b0", "agree": "#cccccc"}

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--addendum", choices=ADDENDA, help="only this addendum's test run (see above)")
args = ap.parse_args()

cfg = load_config(args.config)
ec = cfg["entity"]
root = entity_dir(cfg)
out = root / "analysis"
out.mkdir(parents=True, exist_ok=True)
pd.set_option("display.width", 220)
prereg_path = Path(cfg["entity_prereg"])
prereg = json.loads(prereg_path.read_text()) if prereg_path.exists() else None
k = prereg["k"] if prereg else ec["select"]["k"]
if prereg:
    band, band_src = prereg["band"], "prereg"
elif (root / "e0_dev" / "ranks.npz").exists():
    recs, arrs, info = load_run(root / "e0_dev")
    right = [i for i, r in enumerate(recs) if r["condition"] == "solo" and r["outcome"] == "original"
             and r["format"] == info["formats"][0]]
    band = EM.select_band(EM.curve(arrs[cfg["lens"]["name"]], right, "orig_bridge", "last", k),
                          ec["select"]["band_threshold"])
    band_src = "e0_dev (not yet pre-registered)"
else:
    band, band_src = None, "none"
print(f"band {band} ({band_src}), k {k}")


def pressure_figure(records, arrays, name, path, n_peers, flag=None):
    """flag: only the records with this own-addendum flag set (None: all records)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    if flag:
        records = [r if r.get(flag) else {**r, "round": None} for r in records]  # keeps the rows aligned
    for ax, contrast in zip(axes, ("orig_bridge", "peer_bridge")):
        for style in ec["styles"]:
            for outcome, ls in (("peer", "-"), ("original", ":")):
                idx = [i for i, r in enumerate(records) if r["round"] == 1 and r["condition"] == "pressure"
                       and r["style"] == style and r["n_peers"] == n_peers and r["outcome"] == outcome]
                if idx:
                    ax.plot(EM.curve(arrays, idx, contrast, "last", k), ls, color=COLORS[style], lw=1.8,
                            label=f"{style}, {'gave in' if outcome == 'peer' else 'held'} (n={len(idx)})")
        for cond in ("mention", "mention_bridge"):
            idx = [i for i, r in enumerate(records) if r["round"] == 1 and r["condition"] == cond
                   and r["outcome"] == "original" and r["n_peers"] == n_peers]
            if idx:
                ax.plot(EM.curve(arrays, idx, contrast, "last", k), "--", color=COLORS[cond], lw=1.2,
                        label=f"{cond} baseline (n={len(idx)})")
        if band:
            ax.axvspan(min(band) - 0.5, max(band) + 0.5, color="#f0f0f0", zorder=0)
        ax.axhline(0, color="#999999", lw=0.6)
        ax.set_title(f"{contrast} - control (hit@{k}, last token)", fontsize=10)
        ax.set_xlabel("layer")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("share of records")
    axes[1].legend(frameon=False, fontsize=7, loc="upper left")
    fig.suptitle(f"Round 1, {n_peers} peers, lens {name}" + (f", {flag}" if flag else ""), fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


runs = sorted(p for p in root.iterdir() if p.is_dir() and p.name != "analysis")
if args.addendum:
    runs = [p for p in runs if p.name == f"pressure_test_{args.addendum}"]
for run in runs:
    tag = run.name
    if tag.startswith(("e0_", "pressure_")) and (run / "ranks.npz").exists():
        records, arrays, info = load_run(run)
        ot = EM.outcome_table(records)
        ot.to_csv(out / f"{tag}_outcomes.csv", index=False)
        print(f"\n===== {tag}: outcomes\n{ot.to_string(index=False)}")
        if band is None:
            continue
        for name, ranks in arrays.items():
            for pos in [p for p, j in EM.POS.items() if j < ranks.shape[1]]:
                lt = EM.latent_table(records, ranks, k, band, pos)
                lt.to_csv(out / f"{tag}_{name}_{pos}_latent.csv", index=False)
            EM.layer_curves(records, ranks, k, "last", EM.group_keys(records, ["format", "condition", "style", "n_peers",
                                                                               "round", "outcome"])
                            ).to_csv(out / f"{tag}_{name}_curves.csv", index=False)
            lt = pd.read_csv(out / f"{tag}_{name}_last_latent.csv")
            show = lt[lt["round"].astype(str).isin(["0", "1"])]
            print(f"\n--- {tag} / {name}: band-mean contrasts at the last token (rounds 0-1)")
            print(show[EM.group_keys(records, ["format", "condition", "style", "n_peers", "round"])
                       + ["outcome", "n", "orig_bridge", "orig_bridge_lo", "orig_bridge_hi", "peer_bridge",
                          "peer_bridge_lo", "peer_bridge_hi"]].round(3).to_string(index=False))
            if tag.startswith("pressure_"):
                flags = [f for f in EM.OWN_FLAG.values() if any(r.get(f) for r in records)]
                for n in info.get("n_peers", []):
                    if not flags:
                        pressure_figure(records, ranks, name, out / f"{tag}_{name}_{n}peers.png", n)
                    for f in flags:
                        pressure_figure(records, ranks, name, out / f"{tag}_{name}_{n}peers_{f}.png", n, f)
        if prereg and tag == "pressure_test":
            n = prereg.get("primary_n_peers") or ec["primary_n_peers"]
            verdicts = EM.evaluate_hypotheses(records, arrays, prereg, n)
            (out / "hypotheses.json").write_text(json.dumps(verdicts, indent=1, default=float) + "\n")
            print("\n===== pre-registered hypotheses (test split)")
            for v in verdicts:
                print(f"{v['id']}: estimate {v['estimate']:.3f} [{v['ci_lo']:.3f}, {v['ci_hi']:.3f}] n={v['n']} -> "
                      f"{'supported' if v['supported'] else 'not supported'}")
    elif tag.startswith("timeline_") and (run / "timeline.npz").exists():
        with open(run / "records.jsonl", encoding="utf-8") as f:
            recs = [json.loads(l) for l in f]
        z = np.load(run / "timeline.npz")
        layers = [int(x) for x in z["layers"]]
        for name in [n for n in z.files if n not in ("offsets", "layers")]:
            arrs = [z[name][r["offset"] : r["offset"] + r["window"]] for r in recs]
            tt = EM.timeline_table(recs, arrs, [r["labels"] for r in recs], k, layers)
            tt.to_csv(out / f"{tag}_{name}.csv", index=False)
            seg_order = ["own"] + sorted({s for s in tt.segment if s.startswith("peer")}) + ["moderator", "final"]
            avg = tt[tt.layer.isin(band or layers)].groupby(["condition", "style", "outcome", "segment", "contrast"]
                                                            )["value"].mean().reset_index()
            fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
            for ax, contrast in zip(axes, ("orig_bridge", "peer_bridge")):
                for (cond, style, outcome), g in avg[avg.contrast == contrast].groupby(["condition", "style", "outcome"]):
                    if outcome not in ("peer", "original"):
                        continue
                    y = [g[g.segment == s]["value"].mean() if (g.segment == s).any() else np.nan for s in seg_order]
                    key = style if cond == "pressure" else cond
                    ax.plot(range(len(seg_order)), y, "-" if outcome == "peer" else ":", marker="o", ms=3,
                            color=COLORS.get(key, "#444444"), label=f"{key}, {'gave in' if outcome == 'peer' else 'held'}")
                ax.set_xticks(range(len(seg_order)), seg_order, rotation=30, fontsize=8)
                ax.axhline(0, color="#999999", lw=0.6)
                ax.set_title(f"{contrast} - control over the last round", fontsize=10)
                for s in ("top", "right"):
                    ax.spines[s].set_visible(False)
            axes[1].legend(frameon=False, fontsize=7)
            fig.suptitle(f"{tag}, lens {name}, mean over layers {band or layers}", fontsize=11)
            fig.tight_layout()
            fig.savefig(out / f"{tag}_{name}.png", dpi=180)
            plt.close(fig)
            print(f"\n===== {tag} / {name}: segment means written")
    elif tag.startswith("inject_") and (run / "records.jsonl").exists():
        with open(run / "records.jsonl", encoding="utf-8") as f:
            recs = [json.loads(l) for l in f]
        if recs:
            it = EM.injection_table(recs)
            it.to_csv(out / f"{tag}.csv", index=False)
            print(f"\n===== {tag}\n{it.round(3).to_string(index=False)}")
    elif tag.startswith("debate_") and (run / "records.jsonl").exists():
        with open(run / "records.jsonl", encoding="utf-8") as f:
            recs = [json.loads(l) for l in f]
        for name in recs[0]["candidate_ranks"] if recs and "candidate_ranks" in recs[0] else []:  # not *_bridges
            dt = EM.debate_table(recs, name, band or list(range(len(recs[0]["candidate_ranks"][name][0]))))
            dt.to_csv(out / f"{tag}_{name}.csv", index=False)
            print(f"\n===== {tag} / {name}\n{dt.round(3).to_string(index=False)}")

for part in [args.addendum] if args.addendum else ADDENDA:
    add, run = load_addendum(cfg, part), root / f"pressure_test_{part}"
    if not (prereg and add and (run / "ranks.npz").exists()):
        if args.addendum:
            raise SystemExit(f"--addendum {part}: needs the main prereg, the addendum and {run}/ranks.npz")
        continue
    records, arrays, _ = load_run(run)
    if part == "rounds":
        verdicts = EM.evaluate_rounds(records, arrays, prereg, add)
        if (root / "pressure_test" / "records.jsonl").exists():  # round 1 is the same prompt as in pressure_test
            ref = {EM.state_key(r): r["answer"] for r in load_run(root / "pressure_test")[0] if r["round"] == 1}
            same = [ref[EM.state_key(r)] == r["answer"] for r in records if r["round"] == 1 and EM.state_key(r) in ref]
            print(f"\nround 1 of pressure_test_rounds reproduces pressure_test for {np.mean(same):.3f} of {len(same)} agents")
    elif part == "own":
        verdicts = EM.evaluate_own(records, arrays, add)
        ref = load_run(root / "pressure_test")[:2] if (root / "pressure_test" / "ranks.npz").exists() else None
        summary = EM.own_summary(records, arrays, add, ref)
        (out / "own_summary.json").write_text(json.dumps(summary, indent=1, default=float) + "\n")
        print("\n===== addendum own: give-in rates (share giving the peers' answer, n)")
        for key, r in summary["rates"].items():
            print(f"  {key:22s} {r['peer']} (n={r['n']})")
        print("paired per item (original bridge, items that gave in under both): "
              + "; ".join(f"{key} {p['diff']} {p['ci']} ({p['n_items']} items)" for key, p in summary["paired"].items()))
    else:
        verdicts = EM.evaluate_bridges(records, arrays, add)
    (out / f"hypotheses_{part}.json").write_text(json.dumps(verdicts, indent=1, default=float) + "\n")
    print(f"\n===== addendum {part}: pre-registered hypotheses (test split)")
    for v in verdicts:
        verdict = (v.get("note") or "exploratory") if v["supported"] is None else \
            "supported" if v["supported"] else "not supported"
        print(f"{v['id']}: estimate {v['estimate']:.3f} [{v['ci_lo']:.3f}, {v['ci_hi']:.3f}] n={v['n']} -> {verdict}")
print(f"\n-> {out}")
