"""LaTeX tables for the supplementary material, from the cross-model summary and the prereg files
(no model, GPU or raw results). One file per table in docs/results/paper/supp/.

  flips.tex      share of agents giving the peers' answer (and other answers), style x peers
  cells_<m>.tex  band-mean readouts per condition and lens, original and peer bridge, with 95% CIs
  robust.tex     robustness rows (band rule, k, peers, extra lenses): H1-H4
  inject.tex     E4 shares per injected direction and alpha, agents that gave in and agents that held
  debate.tex     E5 group accuracy per round (models whose E5 run is reported)
  devchecks.tex  development-split validation checks from each model's registration
  own_dev.tex    own addendum: development pilots (format and agree checks) from the addendum files
  own_rates.tex  own addendum: give-in rates per variant and style, paired and H2 comparisons
  own_cells.tex  own addendum: original-bridge readouts per variant and condition
  rob_tests.tex  H1-H4 with record and bridge-clustered intervals, Holm-adjusted p (robust summary)
  rob_tf.tex     threshold-free readouts (logrank, win) for H1, H2, H4
  rob_items.tex  same-item comparisons and the selection check
  rob_cats.tex   H1 and H2 per category
  deb_kinds.tex  free debate: answer classes per round and the registered B07 check (docs/results/debate)
  deb_explore.tex  free debate: the post hoc abstention variants for Qwen3.6-27B (check_explore_summary.json; not registered)
  deb_bridges.tex  free debate: original-bridge readout per round-1 transition (key `debate_bridges`)
The own_* tables need the key `own` of the summary, the rob_* tables docs/results/robust_summary.json.

    python scripts/entity_supp_tables.py [--summary docs/results/crossmodel_summary.json] [--out docs/results/paper/supp]
"""
import argparse
import json
import re
from pathlib import Path

NAMES = {"qwen35_4b_entity": "Qwen3.5-4B", "qwen36_27b_entity": "Qwen3.6-27B",
         "gemma4_e4b_it_entity": "Gemma-4-E4B-it", "llama31_8b_it_entity": "Llama-3.1-8B-Instruct"}
PREREG = {"qwen35_4b_entity": "prereg/qwen35_4b_entity.json", "qwen36_27b_entity": "prereg/qwen36_27b_entity_v2.json",
          "gemma4_e4b_it_entity": "prereg/gemma4_e4b_it_entity.json", "llama31_8b_it_entity": "prereg/llama31_8b_it_entity.json"}
LENS = {"jlens": "J-lens", "logit": "logit", "jlens_base": "J-lens (base)", "jlens_wiki100": "J-lens (refit)"}
COND = [("round0", "round 0 alone"), ("agree", "peers agree"), ("mention", "mention"),
        ("mention_bridge", "mention-bridge"), ("held", "held (answer)"), ("flipped", "gave in (answer)")]
# E5 rows withheld from the supplement (none since 2026-09-30: Qwen3.6-27B's rounds 2-3 are reported with the caveat
# that their drop in stated accuracy is unexplained; docs/RESULTS.md, section 3)
DEBATE_WITHHELD = set()

ap = argparse.ArgumentParser()
ap.add_argument("--summary", default="docs/results/crossmodel_summary.json")
ap.add_argument("--robust", default="docs/results/robust_summary.json")
ap.add_argument("--debate-dir", default="docs/results/debate")
ap.add_argument("--out", default="docs/results/paper/supp")
args = ap.parse_args()
M = json.loads(Path(args.summary).read_text())["models"]
models = [m for m in NAMES if m in M]
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)


def ci(c):
    if not c or c.get("mean") is None:
        return "--"
    f = lambda x: f"{x:.2f}".replace("-0.00", "0.00")
    return f"{f(c['mean'])} [{f(c['ci'][0])}, {f(c['ci'][1])}]"


def hyp(h):
    if not h or h.get("estimate") is None:
        return "--"
    return f"{h['estimate']:.3f} [{h['ci_lo']:.3f}, {h['ci_hi']:.3f}]"


def write(name, lines):
    text = "\n".join(lines) + "\n"
    if name.startswith(("own_", "rob_", "deb_")):  # typeset minus signs of numbers (not in names such as Qwen3.5-4B)
        text = re.sub(r"(?<![$\w.])-(\d)", r"$-$\1", text)
    (out / f"{name}.tex").write_text(text, encoding="utf-8")


# flips
L = [r"\begin{tabular}{lrrrrrrrrrr}", r"\toprule",
     r" & & \multicolumn{3}{c}{answer} & \multicolumn{3}{c}{hop 1} & \multicolumn{3}{c}{hop 2} \\",
     r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}",
     r"Model & $n$ & 1 & 3 & 5 & 1 & 3 & 5 & 1 & 3 & 5 \\", r"\midrule"]
for m in models:
    f = M[m]["flips"]
    n = f["answer/3"]["n"]
    cells = [f"{100 * f[f'{s}/{k}']['peer']:.0f}" for s in ("answer", "hop1", "hop2") for k in (1, 3, 5)]
    L.append(f"{NAMES[m]} & {n} & " + " & ".join(cells) + r" \\")
L.append(r"\midrule")
for m in models:
    f = M[m]["flips"]
    cells = [f"{100 * f[f'{s}/{k}']['other']:.1f}" for s in ("answer", "hop1", "hop2") for k in (1, 3, 5)]
    L.append(f"{NAMES[m]} (other) & & " + " & ".join(cells) + r" \\")
L += [r"\bottomrule", r"\end{tabular}"]
write("flips", L)

# cells per model
for m in models:
    lenses = M[m]["meta"]["lenses"]
    L = [r"\begin{tabular}{l" + "l" * (2 * len(lenses)) + "}", r"\toprule",
         r" & \multicolumn{%d}{c}{original bridge} & \multicolumn{%d}{c}{peer bridge} \\" % (len(lenses), len(lenses)),
         r"\cmidrule(lr){2-%d}\cmidrule(lr){%d-%d}" % (1 + len(lenses), 2 + len(lenses), 1 + 2 * len(lenses)),
         "Condition & " + " & ".join(LENS.get(l, l) for l in lenses) + " & " + " & ".join(LENS.get(l, l) for l in lenses)
         + r" \\", r"\midrule"]
    for c, lab in COND:
        row = [ci(M[m]["cells"].get(f"{c}/{l}/orig_bridge")) for l in lenses]
        row += [ci(M[m]["cells"].get(f"{c}/{l}/peer_bridge")) for l in lenses]
        n = M[m]["cells"].get(f"{c}/jlens/orig_bridge", {}).get("n")
        L.append(f"{lab} ($n{{=}}{n}$) & " + " & ".join(row) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write(f"cells_{m}", L)

# robustness
L = [r"\begin{tabular}{llllll}", r"\toprule", r"Model & Variant & H1 & H2 & H3 & H4 \\", r"\midrule"]
for m in models:
    for r in M[m]["robustness"]:
        n = r.get("H1", {}).get("n")
        L.append(f"{NAMES[m]} & {r['what'].replace('_', chr(92) + '_')} ($n{{=}}{n}$) & {hyp(r.get('H1'))} & "
                 f"{hyp(r.get('H2'))} & {hyp(r.get('H3'))} & {hyp(r.get('H4'))} " + r"\\")
    L.append(r"\midrule" if m != models[-1] else r"\bottomrule")
L.append(r"\end{tabular}")
write("robust", L)

# injection
alphas = ["0.0", "0.1", "0.2", "0.4", "0.8"]
L = [r"\begin{tabular}{lll" + "r" * len(alphas) + "}", r"\toprule",
     r"Model (layers) & Agents & Direction & " + " & ".join(rf"$\alpha{{=}}{a}$" for a in alphas) + r" \\", r"\midrule"]
for m in models:
    inj = M[m]["inject"]
    lay = ", ".join(str(x) for x in inj.get("layers", []))
    for grp, lab in (("gave_in", "gave in"), ("held", "held")):
        g = inj.get(grp, {})
        kinds = sorted({k.split("/")[0] for k in g})
        for i, kind in enumerate(kinds):
            vals = [g.get(f"{kind}/{a}", {}).get("share") for a in alphas]
            head = f"{NAMES[m]} ({lay})" if grp == "gave_in" and i == 0 else ""
            L.append(f"{head} & {lab if i == 0 else ''} & {kind.replace('_', ' ')} & "
                     + " & ".join("--" if v is None else f"{v:.2f}" for v in vals) + r" \\")
    L.append(r"\midrule" if m != models[-1] else r"\bottomrule")
L.append(r"\end{tabular}")
write("inject", L)

# free debate
L = [r"\begin{tabular}{lrrrr}", r"\toprule", r"Model & Round & Stated & Latent (J-lens) & Round-0 plurality \\", r"\midrule"]
for m in models:
    if m in DEBATE_WITHHELD:
        continue
    for r in M[m]["debate"]:
        L.append(f"{NAMES[m] if r['round'] == 0 else ''} & {r['round']} & {r['acc_stated']:.2f} & {r['acc_latent']:.2f} & "
                 f"{r['acc_initial']:.2f} " + r"\\")
L += [r"\bottomrule", r"\end{tabular}"]
write("debate", L)

# free debate, stage 6: answer classes and the registered attribution check (B07), the post hoc variants, and the
# original-bridge readout per transition
check = Path(args.debate_dir) / "check_summary.json"
if check.exists():
    C = json.loads(check.read_text())["models"]
    L = [r"\begin{tabular}{lrrrrrrr}", r"\toprule",
         r"Model & Round & Correct & Other cand. & Degenerate & Other & Stated (a) & Without degenerate (b) \\",
         r"\midrule"]
    for m in models:
        if m not in C:
            continue
        for i, t in enumerate(range(4)):
            k, a = C[m]["kinds"][f"r{t}/all"], C[m]["accuracy"][f"r{t}/all"]
            gone = a["b_no_valid_answer"]
            L.append(f"{NAMES[m] if i == 0 else ''} & {t} & " + " & ".join(f"{k[x]:.2f}" for x in
                     ("correct", "other_candidate", "degenerate", "other")) + f" & {a['a_stated']['mean']:.2f} & "
                     f"{ci(a['b_excluded'])}{f' ({gone})' if gone else ''} " + r"\\")
        L.append(r"\midrule" if m != models[-1] else r"\bottomrule")
    L.append(r"\end{tabular}")
    write("deb_kinds", L)

explore = Path(args.debate_dir) / "check_explore_summary.json"
if explore.exists():
    E = json.loads(explore.read_text())["models"]
    L = [r"\begin{tabular}{lrrrr}", r"\toprule",
         r"Abstentions (attributed?) & Round & Share & (b) & No answer left \\", r"\midrule"]
    for m in ["qwen36_27b_entity"]:  # the other models are not attributed under any variant (text)
        if m not in E:
            continue
        for j, (v, lab) in enumerate((("registered", "as fixed in advance"), ("agent_words", "+ agent words"),
                                      ("candidates", "all non-candidates"))):
            mv = E[m][v]
            for i, t in enumerate(range(1, 4)):
                verdict = "yes" if mv["attribution"]["attributed"] else "no"
                L.append(f"{f'{lab} ({verdict})' if i == 0 else ''} & {t} & "
                         f"{mv['share'][f'r{t}']:.2f} & {ci(mv['b_excluded'][f'r{t}'])} & "
                         f"{mv['b_no_valid_answer'][f'r{t}']} " + r"\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("deb_explore", L)

if any("debate_bridges" in M[m] for m in models):
    def cell(c):
        if not c or not c.get("n"):
            return "--"
        return (ci(c) if "ci" in c else f"{c['mean']:.2f}") + f" ({c['n']})"
    L = [r"\begin{tabular}{llrrrrrr}", r"\toprule",
         r"Model & Facts & Round 0 & Gave in & J-lens $-$ logit & Held & Agreed & Gave in (logit) \\", r"\midrule"]
    for m in models:
        if "debate_bridges" not in M[m]:
            continue
        d = M[m]["debate_bridges"]
        c, x = d["cells"], d["contrasts"]
        for i, s in enumerate(("known", "all")):
            L.append(f"{NAMES[m] if i == 0 else ''} & {s} & {cell(c.get(f'{s}/round0_correct/jlens'))} & "
                     f"{cell(c.get(f'{s}/r1/gave_in/jlens'))} & {cell(x.get(f'{s}/r1/gave_in/jlens-logit'))} & "
                     f"{cell(c.get(f'{s}/r1/held/jlens'))} & {cell(c.get(f'{s}/r1/agreed/jlens'))} & "
                     f"{cell(c.get(f'{s}/r1/gave_in/logit'))} " + r"\\")
        n = d["counts"]
        L.append(r" & \multicolumn{7}{l}{\footnotesize round-1 labels, all facts: " + ", ".join(
            f"{lab.replace('_', ' ')} {n[f'r1/{lab}']}" for lab in ("gave_in", "switched", "held", "held_degenerate",
                                                                   "agreed", "was_wrong") if f"r1/{lab}" in n) + r"} \\")
        L.append(r"\midrule" if m != models[-1] else r"\bottomrule")
    L.append(r"\end{tabular}")
    write("deb_bridges", L)

# development-split checks from the registrations
KEYS = [("round0_orig_bridge", "original bridge, round 0"), ("agree_orig_bridge", "original bridge, peers agree"),
        ("instructed_orig_bridge", "original bridge, instructed"), ("hypothetical_peer_bridge", "peer bridge, hypothetical"),
        ("mention_bridge_peer_bridge", "peer bridge, mention-bridge"), ("mention_peer_bridge", "peer bridge, mention")]
L = [r"\begin{tabular}{l" + "l" * len(models) + "}", r"\toprule", "Check & " + " & ".join(NAMES[m] for m in models) + r" \\",
     r"\midrule"]
pre = {m: json.loads(Path(PREREG[m]).read_text()) for m in models}
dc = {m: pre[m]["dev_checks"][pre[m]["format"]] for m in models}
L.append("format & " + " & ".join(pre[m]["format"] for m in models) + r" \\")
for k, lab in KEYS:
    L.append(f"{lab} & " + " & ".join(ci(dc[m].get(k)) for m in models) + r" \\")
L.append("compliance, instructed & " + " & ".join(f"{dc[m]['compliance']['instructed']:.3f}" for m in models) + r" \\")
L.append("compliance, hypothetical & " + " & ".join(f"{dc[m]['compliance']['hypothetical']:.3f}" for m in models) + r" \\")
L += [r"\bottomrule", r"\end{tabular}"]
write("devchecks", L)

# ---------------------------------------------------------------- own addendum
OWN_PREREG = {m: Path(PREREG[m]).with_name(Path(PREREG[m]).stem + "_own.json") for m in PREREG}
own = [m for m in models if "own" in M[m] and OWN_PREREG[m].exists()]
f2 = lambda x: "--" if x is None else f"{x:.2f}"
cic = lambda d: "--" if not d or d.get("mean") is None else \
    f"{f2(d['mean'])} [{f2(d['ci'][0] if 'ci' in d else d['ci_lo'])}, {f2(d['ci'][1] if 'ci' in d else d['ci_hi'])}]"
if own:
    L = [r"\begin{tabular}{lllrlll}", r"\toprule",
         r"Model & Variant & Pass & Items & Worst `other' & Agree: original bridge & Gave in (answer / hop 1) \\",
         r"\midrule"]
    for m in own:
        add = json.loads(OWN_PREREG[m].read_text())
        for i, v in enumerate(add["variants_declared"]):
            d = add["dev_checks"][v]
            g = d["gave_in"]
            L.append(f"{NAMES[m] if i == 0 else ''} & {v} & {'yes' if d['pass'] else 'no'} & {d['items']} & "
                     f"{d['worst_other']:.3f} & {cic(d['agree_orig_bridge'])} ($n{{=}}{d['agree_orig_bridge']['n']}$) & "
                     f"{100 * g['pressure/answer']['peer']:.0f}\\% / {100 * g['pressure/hop1']['peer']:.0f}\\% " + r"\\")
        L.append(r"\midrule" if m != own[-1] else r"\bottomrule")
    L.append(r"\end{tabular}")
    write("own_dev", L)

    L = [r"\begin{tabular}{lrrrrrrll}", r"\toprule",
         r" & \multicolumn{3}{c}{Gave in, answer (\%)} & \multicolumn{3}{c}{Gave in, hop 1 (\%)} & "
         r"\multicolumn{2}{c}{Same items, original bridge} \\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-9}",
         r"Model & vis. & hid. & abs. & vis. & hid. & abs. & hidden $-$ visible & hidden $-$ absent \\", r"\midrule"]
    for m in own:
        sm = M[m]["own"]["summary"]
        r_ = sm["rates"]
        rate = lambda k: f"{100 * r_[k]['peer']:.0f}" if k in r_ else "--"
        pr = lambda k: (f"{sm['paired'][k]['diff']:.2f} [{sm['paired'][k]['ci'][0]:.2f}, {sm['paired'][k]['ci'][1]:.2f}], "
                        f"{sm['paired'][k]['n_items']} items") if k in sm["paired"] else "--"
        L.append(f"{NAMES[m]} & " + " & ".join(rate(f"{v}/{st}") for st in ("answer", "hop1")
                                                for v in ("visible", "hidden", "absent"))
                 + f" & {pr('hidden-visible')} & {pr('hidden-absent')} " + r"\\")
    L += [r"\midrule", r" & \multicolumn{8}{l}{H2 analogue: peer bridge, gave in minus mention (visible / hidden / absent)} \\"]
    for m in own:
        h2 = M[m]["own"]["summary"]["h2"]
        L.append(f"{NAMES[m]} & \\multicolumn{{8}}{{l}}{{" + " / ".join(
            f"{h2[v]['diff']:.2f} [{h2[v]['ci'][0]:.2f}, {h2[v]['ci'][1]:.2f}]" for v in ("visible", "hidden", "absent")
            if v in h2) + r"} \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("own_rates", L)

    L = [r"\begin{tabular}{llllll}", r"\toprule",
         r"Model & Variant & Peers agree & Mention & Held & Gave in \\", r"\midrule"]
    for m in own:
        c = M[m]["own"]["summary"]["cells"]
        for i, v in enumerate(("hidden", "absent")):
            row = []
            for g in ("agree", "mention", "held", "flipped"):
                j, lg = c.get(f"{v}/{g}/jlens/orig_bridge"), c.get(f"{v}/{g}/logit/orig_bridge")
                row.append(f"{cic(j)}; {f2(lg.get('mean') if lg else None)}")
            L.append(f"{NAMES[m] if i == 0 else ''} & {v} & " + " & ".join(row) + r" \\")
        L.append(r"\midrule" if m != own[-1] else r"\bottomrule")
    L.append(r"\end{tabular}")
    write("own_cells", L)

# ---------------------------------------------------------------- exploratory robustness
if Path(args.robust).exists():
    R = json.loads(Path(args.robust).read_text())["models"]
    rob = [m for m in models if m in R]
    fp = lambda p: "--" if p is None else ("$<$0.001" if p < 0.001 else f"{p:.3f}")
    cl = lambda t, sh=0.0: ("--" if not t or t.get("estimate") is None else
                            f"{t['estimate'] + sh:.2f} [{t['cluster']['ci'][0] + sh:.2f}, {t['cluster']['ci'][1] + sh:.2f}]"
                            if "cluster" in t else f"{t['estimate'] + sh:.2f}")
    nn = lambda n: "$\\times$".join(map(str, n)) if isinstance(n, list) else str(n)

    L = [r"\begin{tabular}{lllrlllll}", r"\toprule",
         r"Model & Test & Estimate & $n$ & Records: 95\% CI & Bridges & Bridges: 95\% CI & $p$ (Holm) & Supported \\",
         r"\midrule"]
    for m in rob:
        for i, (h, t) in enumerate(R[m]["tests"].items()):
            if t.get("estimate") is None:
                L.append(f"{NAMES[m] if i == 0 else ''} & {h} & \\multicolumn{{7}}{{l}}{{no estimate ({nn(t['n'])} records)}} \\\\")
                continue
            c, r_ = t["cluster"], t["record"]
            L.append(f"{NAMES[m] if i == 0 else ''} & {h} & {t['estimate']:.3f} & {nn(t['n'])} & "
                     f"[{r_['ci'][0]:.3f}, {r_['ci'][1]:.3f}] & {t['n_clusters']} & [{c['ci'][0]:.3f}, {c['ci'][1]:.3f}] & "
                     f"{fp(c.get('p_holm'))} & {'yes' if c.get('holm_supported') else 'no'} " + r"\\")
        L.append(r"\midrule" if m != rob[-1] else r"\bottomrule")
    L.append(r"\end{tabular}")
    write("rob_tests", L)

    L = [r"\begin{tabular}{llllll}", r"\toprule",
         r"Model & H1 logrank & H1 win & H2 logrank & H4 logrank & H4 win \\", r"\midrule"]
    for m in rob:
        t = R[m]["threshold_free"]["tests"]
        L.append(f"{NAMES[m]} & {cl(t['H1/logrank'])} & {cl(t['H1/win'], 0.5)} & {cl(t['H2/logrank'])} & "
                 f"{cl(t['H4/logrank'])} & {cl(t['H4/win'])} " + r"\\")
    L += [r"\midrule", r" & \multicolumn{5}{l}{Gave in, logrank of the original bridge: \jlens{} / logit lens} \\"]
    for m in rob:
        c = R[m]["threshold_free"]["cells"]
        L.append(f"{NAMES[m]} & \\multicolumn{{5}}{{l}}{{{cl(c['flipped/jlens/orig_bridge']['logrank'])} / "
                 f"{cl(c['flipped/logit/orig_bridge']['logrank'])}}} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("rob_tf", L)

    PAIRS = [("flipped-agree/orig_bridge", "gave in $-$ agree"), ("flipped-round0/orig_bridge", "gave in $-$ round 0"),
             ("held-agree/orig_bridge", "held $-$ agree"), ("flipped-mention/peer_bridge", "gave in $-$ mention (peer b.)")]
    L = [r"\begin{tabular}{l" + "l" * len(rob) + "}", r"\toprule", "Comparison & " + " & ".join(NAMES[m] for m in rob) + r" \\",
         r"\midrule"]
    for key, lab in PAIRS:
        for kind in ("hit", "logrank"):
            cells = []
            for m in rob:
                t = R[m]["within_item"].get(f"{key}/{kind}", {})
                ratio = t.get("ratio", {}).get("estimate")
                cells.append(cl(t) + (f"; {ratio:.2f}" if ratio is not None and kind == "hit" else ""))
            L.append(f"{lab} ({kind}) & " + " & ".join(cells) + r" \\")
    L.append(r"\midrule")
    for c_ in ("round0", "agree"):
        for kind in ("hit", "logrank"):
            L.append(f"selection, {c_.replace('round0', 'round 0')} ({kind}) & "
                     + " & ".join(cl(R[m]["within_item"].get(f"selection/{c_}/{kind}")) for m in rob) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("rob_items", L)

    CATN = {"landmark-cntry-capital": "L", "person-birthcity-cntry": "B", "univ-hqcity-cntry": "U",
            "person-uguniv-hqcity": "P$_c$", "person-uguniv-hqcntry": "P$_n$"}
    # intervals only where every sample has >= 10 readable records (the rule recorded before the run;
    # entity_robust.py counted agents, including those whose bridge is unreadable)
    big = lambda t: t.get("estimate") is not None and min(t["n"] if isinstance(t["n"], list) else [t["n"]]) >= 10
    cc = lambda t: cl(t) if big(t) else ("--" if t.get("estimate") is None else f"{t['estimate']:.2f}")
    L = [r"\begin{tabular}{llrrrlll}", r"\toprule",
         r"Model & Cat. & Pressed & Gave in & Held & H1 & H1 logrank & H2 \\", r"\midrule"]
    for m in rob:
        for i, (cat, t) in enumerate(R[m]["categories"].items()):
            h1n = t["H1"].get("n")
            L.append(f"{NAMES[m] if i == 0 else ''} & {CATN.get(cat, cat)} & {t['n_pressed']} & {t['n_flipped']} & "
                     f"{t['n_held']} & {cc(t['H1'])}" + (f" ($n{{=}}{h1n}$)" if t["H1"].get("estimate") is not None else "")
                     + f" & {cc(t['H1/logrank'])} & {cc(t['H2'])} " + r"\\")
        L.append(r"\midrule" if m != rob[-1] else r"\bottomrule")
    L.append(r"\end{tabular}")
    write("rob_cats", L)
print(f"-> {out}: " + ", ".join(sorted(p.name for p in out.iterdir())))
