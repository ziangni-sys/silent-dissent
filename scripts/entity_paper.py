"""Paper tables and figures for the entity study across models, from the cross-model summary
(scripts/entity_crossmodel.py, run where the raw results are; its output is kept in the repo as
docs/results/crossmodel_summary.json). Needs no model, GPU or raw results.

  fig1_layers     original-bridge readout per layer: agents that gave in vs agents whose peers agreed,
                  J-lens vs logit lens, one panel per model, the pre-registered band shaded
  fig2_h1         H1 cell per model: agents that gave in, J-lens vs logit lens (band mean, 95% CI), with
                  the agree baseline for reference
  fig_conditions  original-bridge readout per condition (round 0, peers agreed, held, gave in), J-lens vs
                  logit lens, one panel per model (band mean, 95% CI)
  fig3_conformity flip rate against retention (gave-in readout / agree readout), k = 100 and k = 500
  fig4_k          the same cells as a function of k: 1 to 10000 from the robustness summary
                  (docs/results/robust_summary.json, k_curve) where it exists, else 10 / 50 / 100 / 500
  fig5_inject     E4: share of agents that gave in answering the original after injection, per kind, with
                  95% Wilson bands counting each agent once (a cell pools its injected layers)
  tables.md / tables.tex   setup, pre-registered hypotheses, cells, robustness, E4, E5, later rounds

    python scripts/entity_paper.py [--summary docs/results/crossmodel_summary.json] [--out docs/results/paper]
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

NAMES = {"qwen35_4b_entity": "Qwen3.5-4B", "qwen36_27b_entity": "Qwen3.6-27B",
         "gemma4_e4b_it_entity": "Gemma-4-E4B-it", "llama31_8b_it_entity": "Llama-3.1-8B-Instruct"}
CATS = {"landmark-cntry-capital": "landmark→capital", "person-birthcity-cntry": "person→birth city→country",
        "univ-hqcity-cntry": "university→city→country", "person-uguniv-hqcity": "person→university→city",
        "person-uguniv-hqcntry": "person→university→country"}
# Categorical slots 1-3 of the reference palette (validated: scripts/validate_palette.js, light mode)
J, LOGIT, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, BAND = "#0b0b0b", "#52514e", "#e4e3df", "#f0efec"

ap = argparse.ArgumentParser()
ap.add_argument("--summary", default="docs/results/crossmodel_summary.json")
ap.add_argument("--robust", default="docs/results/robust_summary.json")
ap.add_argument("--out", default="docs/results/paper")
args = ap.parse_args()
data = json.loads(Path(args.summary).read_text())
models = [m for m in NAMES if m in data["models"]]
M = data["models"]
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.titlesize": 7.5, "axes.titleweight": "bold", "legend.frameon": False,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight"})


def arr(x):
    return np.array([np.nan if v is None else v for v in x], dtype=float)


def style(ax):
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", dpi=220)
    plt.close(fig)


def band_runs(layers):
    """[32, 34, 35, 36] -> [(32, 32), (34, 36)]."""
    out, start = [], layers[0]
    for a, b in zip(layers, layers[1:] + [None]):
        if b != a + 1:
            out.append((start, a))
            start = b
    return out


def flip(m, style="answer", n=3):
    return M[m]["flips"].get(f"{style}/{n}", {}).get("peer")


def cell(m, c, lens="jlens", contrast="orig_bridge"):
    return M[m]["cells"].get(f"{c}/{lens}/{contrast}", {})


# ------------------------------------------------------------------ fig 1: layers
fig, axes = plt.subplots(1, len(models), figsize=(1.75 * len(models), 1.85), sharey=True)
for ax, m in zip(np.atleast_1d(axes), models):
    meta = M[m]["meta"]
    for lo, hi in band_runs(meta["band"]):
        ax.axvspan(lo - 0.5, hi + 0.5, color=BAND, zorder=0, lw=0)
    for lens, color in (("jlens", J), ("logit", LOGIT)):
        for c, ls in (("flipped", "-"), ("agree", "--")):
            y = arr(M[m]["curves"][f"{c}/{lens}/orig_bridge"])
            ax.plot(np.arange(len(y)), y, ls, color=color, lw=1.8 if ls == "-" else 1.3,
                    label=f"{'J-lens' if lens == 'jlens' else 'logit lens'}, {'gave in' if c == 'flipped' else 'peers agreed'}")
    ax.set_title(f"{NAMES[m]}\n(gave in: {flip(m):.0%})", loc="left")
    ax.set_xlabel("layer")
    ax.set_xlim(0, meta["n_layers"] - 1)
    ax.axhline(0, color=INK2, lw=0.6)
    style(ax)
np.atleast_1d(axes)[0].set_ylabel("original bridge readout\n(hit@100 − control)")
h, l = np.atleast_1d(axes)[0].get_legend_handles_labels()
fig.legend(h, l, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.1))
fig.tight_layout()
save(fig, "fig1_layers")

# ------------------------------------------------------------------ fig 2: H1 per model
fig, ax = plt.subplots(figsize=(5.2, 0.55 * len(models) + 0.9))
for i, m in enumerate(models[::-1]):
    for lens, color, dy, lab in (("jlens", J, 0.13, "J-lens, gave in"), ("logit", LOGIT, -0.13, "logit lens, gave in")):
        c = cell(m, "flipped", lens)
        if c.get("mean") is not None:
            ax.errorbar(c["mean"], i + dy, xerr=[[c["mean"] - c["ci"][0]], [c["ci"][1] - c["mean"]]], fmt="o",
                        color=color, ms=6, lw=1.6, capsize=0, label=lab if i == 0 else None)
    a = cell(m, "agree", "jlens")
    ax.plot(a["mean"], i + 0.13, "o", mfc="white", mec=J, mew=1.6, ms=7, label="J-lens, peers agreed" if i == 0 else None,
            zorder=3)
ax.set_yticks(range(len(models)), [f"{NAMES[m]}  (n={cell(m, 'flipped')['n']})" for m in models[::-1]])
ax.axvline(0, color=INK2, lw=0.6)
ax.set_xlim(-0.05, 1.0)
ax.set_xlabel("original bridge readout in the pre-registered band (95% CI)")
ax.grid(axis="x", color=GRID, lw=0.6)
ax.set_axisbelow(True)
ax.legend(loc="lower right", fontsize=8)
save(fig, "fig2_h1")

# ------------------------------------------------------------------ fig conditions: the dissociation
CONDS = [("round0", "alone"), ("agree", "agree"), ("held", "held"), ("flipped", "gave in")]
fig, axes = plt.subplots(1, len(models), figsize=(1.75 * len(models), 1.85), sharey=True)
for ax, m in zip(np.atleast_1d(axes), models):
    ax.axvspan(len(CONDS) - 1.5, len(CONDS) - 0.5, color=BAND, zorder=0, lw=0)
    for lens, color, dx, lab in (("jlens", J, -0.08, "J-lens"), ("logit", LOGIT, 0.08, "logit lens")):
        cs = [cell(m, c, lens) for c, _ in CONDS]
        x = np.arange(len(CONDS)) + dx
        y = arr([c.get("mean") for c in cs])
        lo = arr([c["ci"][0] if c.get("mean") is not None else None for c in cs])
        hi = arr([c["ci"][1] if c.get("mean") is not None else None for c in cs])
        ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt="o", color=color, ms=3.5, lw=1.2, capsize=0, label=lab)
    ns = [cell(m, c, "jlens").get("n") for c, _ in CONDS]
    ax.set_xticks(range(len(CONDS)), [f"{lab}\n{n}" for (_, lab), n in zip(CONDS, ns)])
    ax.set_title(NAMES[m], loc="left")
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_ylim(-0.05, 1.0)
    style(ax)
np.atleast_1d(axes)[0].set_ylabel("original bridge readout\n(hit@100 − control)")
h, l = np.atleast_1d(axes)[0].get_legend_handles_labels()
fig.legend(h, l, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.1))
fig.tight_layout()
save(fig, "fig_conditions")

# ------------------------------------------------------------------ fig 3: conformity vs retention
fig, ax = plt.subplots(figsize=(3.1, 2.3))
for m in models:
    x = flip(m)
    pts = []
    for kk, filled in ((100, True), (500, False)):
        f, a = M[m]["k_cells"][f"{kk}/flipped/jlens"], M[m]["k_cells"][f"{kk}/agree/jlens"]
        if f["mean"] is not None and a["mean"]:
            pts.append(f["mean"] / a["mean"])
            # a ring (k = 500) drawn larger and below the dot (k = 100), so coinciding points both show
            ax.plot(x, pts[-1], "o", ms=5 if filled else 8, color=J, mfc=J if filled else "white", mew=1.6,
                    zorder=4 if filled else 3)
    if len(pts) == 2:
        ax.plot([x, x], pts, color=J, lw=0.8, alpha=0.5)
    dy = {"gemma4_e4b_it_entity": -7, "qwen36_27b_entity": 0}.get(m, 0)
    ax.annotate(NAMES[m], (x, pts[0]), xytext=(8, dy), textcoords="offset points", va="center", fontsize=7, color=INK)
ax.plot([], [], "o", color=J, label="k = 100 (pre-registered k)")
ax.plot([], [], "o", ms=8, mfc="white", mec=J, mew=1.6, label="k = 500")
ax.set_xlabel("share of agents that gave in (answer pressure, 3 peers)")
ax.set_ylabel("retention (J-lens):\ngave in ÷ peers agreed")
ax.set_xlim(0, 1)
ax.set_ylim(bottom=0)
ax.axhline(1, color=INK2, lw=0.7, ls="--")
style(ax)
ax.legend(fontsize=7, loc="center right", bbox_to_anchor=(1.0, 0.72))
save(fig, "fig3_conformity")

# ------------------------------------------------------------------ fig 4: k
R = json.loads(Path(args.robust).read_text()) if Path(args.robust).exists() else None
ks = R["ks"] if R else [10, 50, 100, 500]
fig, axes = plt.subplots(1, len(models), figsize=(1.75 * len(models), 1.9), sharey=True)
for ax, m in zip(np.atleast_1d(axes), models):
    for lens, color in (("jlens", J), ("logit", LOGIT)):
        for c, ls in (("flipped", "-"), ("agree", "--")):
            y = (R["models"][m]["k_curve"][f"{c}/{lens}/orig_bridge"] if R else
                 [M[m]["k_cells"][f"{kk}/{c}/{lens}"]["mean"] for kk in ks])
            ax.plot(ks, arr(y), ls, marker="o", ms=4, color=color, lw=1.8 if ls == "-" else 1.3,
                    label=f"{'J-lens' if lens == 'jlens' else 'logit lens'}, {'gave in' if c == 'flipped' else 'peers agreed'}")
    ax.set_xscale("log")
    ax.set_xticks(ks[::2] if len(ks) > 5 else ks, [str(k) for k in (ks[::2] if len(ks) > 5 else ks)])
    ax.minorticks_off()
    ax.set_title(NAMES[m], loc="left")
    ax.set_xlabel("k (hit@k)")
    ax.axhline(0, color=INK2, lw=0.6)
    style(ax)
np.atleast_1d(axes)[0].set_ylabel("original bridge readout\n(band mean)")
h, l = np.atleast_1d(axes)[0].get_legend_handles_labels()
fig.legend(h, l, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.12))
fig.tight_layout()
save(fig, "fig4_k")

# ------------------------------------------------------------------ fig 5: E4
def wilson(p, n, z=1.96):
    """95% Wilson interval of a share p out of n (NaN if missing)."""
    if p is None or not n:
        return np.nan, np.nan
    c, w = (p + z * z / (2 * n)) / (1 + z * z / n), z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, c - w), min(1.0, c + w)


alphas = [0.0, 0.1, 0.2, 0.4, 0.8]
kinds = [("orig_bridge", J, "-", "original bridge"), ("orig_answer", AQUA, "-", "original answer"),
         ("ctrl_bridge", INK2, "--", "control bridge"), ("random", INK2, ":", "random direction")]
fig, axes = plt.subplots(1, len(models), figsize=(1.75 * len(models), 1.9), sharey=True)
for ax, m in zip(np.atleast_1d(axes), models):
    g = M[m].get("inject", {}).get("gave_in", {})
    n_layers = max(len(M[m].get("inject", {}).get("layers", [])), 1)  # cells pool the injected layers
    for kind, color, ls, lab in kinds:
        cells = [g.get(f"{kind}/{a}", {}) for a in alphas]
        ax.plot(alphas, arr([c.get("share") for c in cells]), ls, marker="o", ms=4, color=color, lw=1.6, label=lab)
        band = np.array([wilson(c.get("share"), c.get("n", 0) / n_layers) for c in cells])
        ax.fill_between(alphas, band[:, 0], band[:, 1], color=color, alpha=0.13, lw=0)
    n = max((v["n"] for v in g.values()), default=0) // n_layers
    ax.set_title(f"{NAMES[m]}\n({n} agents)", loc="left")
    ax.set_xlabel("injection strength α")
    style(ax)
np.atleast_1d(axes)[0].set_ylabel("top next token is the\noriginal answer (share)")
h, l = np.atleast_1d(axes)[0].get_legend_handles_labels()
fig.legend(h, l, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.12))
fig.tight_layout()
save(fig, "fig5_inject")

# ------------------------------------------------------------------ tables
f3 = lambda x: "–" if x is None else f"{x:.3f}"
f2 = lambda x: "–" if x is None else f"{x:.2f}"
ci = lambda h: f"{f3(h.get('estimate'))} [{f3(h.get('ci_lo'))}, {f3(h.get('ci_hi'))}]"
cm = lambda c: "–" if c.get("mean") is None else f"{c['mean']:.2f} [{c['ci'][0]:.2f}, {c['ci'][1]:.2f}]"
nn = lambda n: "×".join(str(x) for x in n) if isinstance(n, list) else str(n)


def runs(layers):
    """[32, 34, 35, 36] -> '32, 34–36'."""
    out, start = [], layers[0]
    for a, b in zip(layers, layers[1:] + [None]):
        if b != a + 1:
            out.append(f"{start}–{a}" if a != start else str(a))
            start = b
    return ", ".join(out)


def hcell(h):
    if h.get("estimate") is None:
        return f"untestable (n={nn(h['n'])})"
    return f"{ci(h)}{' *' if h['supported'] else ''} (n={nn(h['n'])})"


tables = []

rows = []
for m in models:
    meta = M[m]["meta"]
    rows.append([NAMES[m], meta["rule"], meta["format"], f"{runs(meta['band'])} / {meta['n_layers']}", str(meta["k"]),
                 "; ".join(CATS.get(c, c) for c in meta["categories"] or []), str(meta["n_items"]),
                 f"{meta['round0_accuracy']:.2f}", " / ".join(f"{flip(m, s):.0%}" for s in ("answer", "hop1", "hop2"))])
tables.append(("Setup (test split)", ["model", "rule", "format", "band / layers", "k", "categories", "items",
                                        "round-0 acc.", "gave in: answer / hop1 / hop2 (3 peers)"], rows))

rows = []
for m in models:
    h = M[m]["hypotheses"]
    rows.append([NAMES[m]] + [hcell(h[i]) for i in ("H1", "H2", "H3", "H4")])
tables.append(("Pre-registered hypotheses (* = 95% CI excludes 0)", ["model", "H1 retention", "H2 implied premise",
                                                                    "H3 hop2 vs hop1", "H4 J-lens − logit"], rows))

rows = []
for m in models:
    for lens in [x for x in M[m]["meta"]["lenses"]]:
        rows.append([NAMES[m], lens] + [cm(cell(m, c, lens)) for c in ("round0", "agree", "held", "flipped")]
                    + [cm(cell(m, "flipped", lens, "peer_bridge")), cm(cell(m, "mention_bridge", lens, "peer_bridge"))])
tables.append(("Band-mean readouts (3 peers, round 1; round 0 alone)",
               ["model", "lens", "orig: round 0", "orig: peers agreed", "orig: held", "orig: gave in",
                "peer: gave in", "peer: mention_bridge"], rows))

rows = []
for m in models:
    for r in M[m]["robustness"]:
        rows.append([NAMES[m], r["what"] + (f" ({runs(r['band'])})" if "band" in r else ""),
                     f"{ci(r['H1'])} (n={nn(r['H1']['n'])})", ci(r["H2"]), ci(r.get("H4", {}))])
tables.append(("Robustness on the test split (exploratory)", ["model", "variant", "H1", "H2", "H4"], rows))

rows = []
for m in models:
    g = M[m].get("inject", {}).get("gave_in", {})
    best = lambda kind: max((g.get(f"{kind}/{a}", {}).get("share") or 0) for a in alphas[1:])
    rows.append([NAMES[m], f2(g.get("orig_bridge/0.0", {}).get("share"))] +
                [f2(best(k)) for k in ("orig_bridge", "ctrl_bridge", "random", "orig_answer")] +
                [", ".join(map(str, M[m].get("inject", {}).get("layers", [])))])
tables.append(("E4: agents that gave in, share answering the original (max over α > 0)",
               ["model", "α = 0", "original bridge", "control bridge", "random", "original answer", "layers"], rows))

rows = []
for m in models:
    for r in M[m].get("debate", []):
        rows.append([NAMES[m], str(r["round"]), f2(r["acc_stated"]), f2(r["acc_latent"]), f2(r["acc_initial"])])
tables.append(("E5 free debate: group accuracy (plurality)", ["model", "round", "stated", "latent (J-lens)", "round 0"], rows))

rows = [[NAMES[m], r["id"], ci(r), nn(r["n"])] for m in models for r in M[m].get("rounds", [])]
if rows:
    tables.append(("Later rounds (Qwen3.5-4B addendum, summary protocol, round 3)", ["model", "hypothesis", "estimate [CI]", "n"], rows))

md, tex = [], []
for title, head, rows in tables:
    md += [f"### {title}", "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    md += ["| " + " | ".join(r) + " |" for r in rows] + [""]
    esc = lambda s: s.replace("_", r"\_").replace("%", r"\%").replace("→", r"$\to$").replace("−", "$-$").replace("×", r"$\times$").replace("α", r"$\alpha$")
    tex += [f"% {title}", r"\begin{tabular}{" + "l" * len(head) + "}", r"\toprule",
            " & ".join(esc(h) for h in head) + r" \\", r"\midrule"]
    tex += [" & ".join(esc(c) for c in r) + r" \\" for r in rows] + [r"\bottomrule", r"\end{tabular}", ""]
(out / "tables.md").write_text("\n".join(md), encoding="utf-8")
(out / "tables.tex").write_text("\n".join(tex), encoding="utf-8")
print(f"-> {out}: " + ", ".join(sorted(p.name for p in out.iterdir())))
