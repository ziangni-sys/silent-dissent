"""Tables (CSV) and the layer-curve figure from saved records."""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from silent_dissent import metrics as M
from silent_dissent.config import load_config, load_prereg
from silent_dissent.experiments import read_jsonl

# Categorical slots 1-4 of the reference palette, in fixed order.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
args = ap.parse_args()

cfg = load_config(args.config)
out = cfg["out_dir"]
prereg = load_prereg(cfg)
layer, delta = prereg["layer"], prereg["delta"]
(out / "tables").mkdir(exist_ok=True)
(out / "figures").mkdir(exist_ok=True)

pressure = []
for name in cfg["pressure"]["grids"]:
    path = out / f"pressure_{name}.jsonl"
    if path.exists():
        pressure += read_jsonl(path)

if pressure:
    M.flip_table(pressure).to_csv(out / "tables/flip_rates.csv", index=False)
    M.silent_dissent_table(pressure, layer, delta).to_csv(out / "tables/silent_dissent.csv", index=False)
    mc = M.mention_control_table(pressure, layer)
    if len(mc):
        mc.to_csv(out / "tables/mention_control.csv", index=False)

    # Figure: where along depth does the original answer lose?
    # Pressure series use the intervention source cell (answer-only peers barely flip the
    # model); the controls exist only at 3 answer-only peers.
    last = cfg["pressure"]["rounds"]
    src = cfg["intervention"]["source"]
    sel = lambda cond, n, style: [r for r in pressure if r["condition"] == cond and r["n_peers"] == n
                                  and r["peer_style"] == style and r["round"] == last]
    strong = sel(src["condition"], src["n_peers"], src["peer_style"])
    series = [
        ("Pressure, flipped", [r for r in strong if M.is_flip(r)]),
        ("Pressure, held", [r for r in strong if not M.is_flip(r)]),
        ("Original removed, flipped", [r for r in sel("remove_original", 3, "answer_only") if M.is_flip(r)]),
        ("Peers agree", sel("agree", 3, "answer_only")),
    ]
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for (label, recs), c in zip(series, COLORS):
        if recs:
            ax.plot(M.original_top1_curve(recs), color=c, lw=2, label=f"{label} (n={len(recs)})")
    ax.axvline(layer, color="#888888", lw=1, ls="--")
    ax.text(layer, 1.0, " prereg layer", color="#555555", fontsize=8, ha="left", va="bottom")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Layer")
    ax.set_ylabel("Original answer is lens top-1 (fraction)")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25, lw=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    style = src["peer_style"].replace("_", " ")
    ax.set_title(f"Original answer across depth, round {last} (pressure: {src['n_peers']} peers, {style})",
                 fontsize=10, pad=14)
    fig.tight_layout()
    fig.savefig(out / "figures/original_top1_by_layer.png", dpi=200)

for path in sorted(out.glob("intervention*.jsonl")):  # incl. tagged runs, e.g. intervention_r1.jsonl
    M.intervention_table(read_jsonl(path)).to_csv(out / f"tables/{path.stem}.csv", index=False)

path = out / "debate.jsonl"
if path.exists():
    M.aggregation_table(read_jsonl(path), layer).to_csv(out / "tables/aggregation.csv", index=False)

print(f"tables -> {out / 'tables'}, figures -> {out / 'figures'}")
