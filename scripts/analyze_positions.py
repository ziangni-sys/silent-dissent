"""Summarise scripts/probe_positions.py output (no model needed). Exploratory, not pre-registered.

Criteria, fixed before the data were looked at (sensitivity counts complied records only,
i.e. stated == the letter the agent was asked for; chance is ~0.25):

  usable readout   reliability >= 0.9 on round-0 prompts (the prereg tau) and sensitivity
                   >= 0.5 in both variants and both original_correct strata of a family
  visible at all   (shifted family) some (position, layer) has sensitivity >= 0.5 in both
                   variants and strata, whatever its reliability: the lens can show an
                   internal answer that differs from the output

    python scripts/analyze_positions.py --config configs/qwen35_4b.yaml
"""
import argparse
import json

import numpy as np

from silent_dissent import metrics as M
from silent_dissent.config import load_config, load_prereg
from silent_dissent.experiments import read_jsonl

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--layers", type=int, nargs=2, default=(15, None), help="layer range shown in the grids")
args = ap.parse_args()

cfg = load_config(args.config)
d = cfg["out_dir"] / "positions"
meta = read_jsonl(d / "meta.jsonl")
arrays = np.load(d / "lens_letter_logits.npz")
tail = json.loads((d / "tail_tokens.json").read_text())
print("tail tokens:", " | ".join(f"{p - tail['n_last']}:{t!r}" for p, t in enumerate(tail["tokens"])),
      "| identical across prompts:", tail["same_tail_for_all_prompts"])
variants = sorted({m["variant"] for m in meta} - {"solo"})
for v in variants:
    rs = [m for m in meta if m["variant"] == v]
    print(f"compliance {v:18s} {np.mean([m['stated'] == m['target'] for m in rs]):.3f}  (n={len(rs)})")

# Sanity: position -1 of the configured lens must reproduce the prereg agreement curve.
prereg = load_prereg(cfg)
if prereg["lens"] in arrays.files:
    df = M.position_summary(meta, arrays[prereg["lens"]].astype(np.float32))
    rel = df[df.position == -1].sort_values("layer")["reliability"].to_numpy()
    if len(rel) == len(prereg["agreement_curve"]):
        print(f"position -1 vs prereg agreement curve: max |diff| {np.abs(rel - prereg['agreement_curve']).max():.3f}")

# Sanity: the instructed prompts must reproduce the positive-control run's round-1 answers.
pc = cfg["out_dir"] / "pressure_positive_control.jsonl"
if pc.exists():
    ref = {r["item_id"]: r["stated"] for r in read_jsonl(pc) if r["round"] == 1}
    ins = [m for m in meta if m["variant"] == "instructed" and m["item_id"] in ref]
    if ins:
        print(f"instructed answers equal to the positive-control run: {np.mean([ref[m['item_id']] == m['stated'] for m in ins]):.3f}")


def show_grid(df, col, layers):
    grid = df.pivot(index="position", columns="layer", values=col)[layers]
    print(f"\n{col} (rows: position, cols: layer)")
    print("pos  " + " ".join(f"{l:>4d}" for l in layers))
    for p, row in grid.iterrows():
        print(f"{p:>3d}  " + " ".join(f"{x:4.2f}" for x in row))


lo, hi = args.layers
for name in arrays.files:
    df = M.position_summary(meta, arrays[name].astype(np.float32))
    df.to_csv(d / f"summary_{name}.csv", index=False)
    layers = [l for l in sorted(df.layer.unique()) if l >= lo and (hi is None or l <= hi)]
    print(f"\n===== lens {name} | complied records per column: {df.attrs['counts']}")
    show_grid(df, "reliability", layers)
    for fam, vs in M.POSITION_FAMILIES.items():
        if f"sens_min_{fam}" not in df:
            continue
        print(f"\n----- family {fam} ({', '.join(vs)})")
        show_grid(df, f"sens_min_{fam}", layers)
        show_grid(df, f"target_{vs[0]}", layers)
        sens = [f"sens_{v}_{s}" for v in vs for s in ("ok", "wrong")]
        best = df.sort_values(f"sens_min_{fam}", ascending=False).head(6)
        print(f"\ntop cells by sens_min_{fam}:")
        print(best[["position", "layer", "reliability", *sens, f"target_{vs[0]}"]].round(3).to_string(index=False))
        top = best.iloc[0]
        print(f"USABLE READOUT ({name}, {fam}): {int(df[f'passes_{fam}'].sum())} cells",
              df[df[f"passes_{fam}"]][["position", "layer"]].values.tolist())
        print(f"VISIBLE AT ALL ({name}, {fam}): {'yes' if top[f'sens_min_{fam}'] >= 0.5 else 'no'} "
              f"(best cell position {int(top.position)}, layer {int(top.layer)}: sens_min {top[f'sens_min_{fam}']:.3f})")
