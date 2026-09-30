"""Entity study, follow-up diagnosis (exploratory, dev split): where can the lens read the bridge?

The pre-registered readout (last token) reads place bridges (cities, countries) but not person
bridges ("The author of the novel X was born in the city of" -> the author). This script asks, per
bridge type and every category of the split:
  statement  the round-0 prompt: ranks at every token from the subject's / mention's last token to
             the answer position (the mention is the bridge's description, e.g. "The author of the
             novel X"; there the first hop should be resolved)
  hop1       positive control: the first hop as the statement ("Agent 1: The author of the novel X
             is"), where the bridge is the next token: can the lens show this entity's name at all?
for every probe mode (entity_facts.PROBE_MODES: any content word, or the last word = surname) and
lens. Round-0 and hop-1 answers are generated; the tables use items answered correctly.
Writes <out_dir>/entity/bridgeprobe_<split>/{records.jsonl, window.npz, hop1.npz, info.json}.

    python scripts/entity_bridge_probe.py --config configs/qwen35_4b_entity.yaml --split dev
"""
import argparse
import dataclasses
import json
import subprocess
import warnings
from pathlib import Path

import numpy as np

from silent_dissent import entity_metrics as EM
from silent_dissent.config import load_config, setup
from silent_dissent.entity_experiments import classify, entity_readout, generate, readout_inputs, timeline_readout
from silent_dissent.entity_facts import PROBE_MODES, ROLES, matches, role_probes
from silent_dissent.entity_io import entity_dir, lens_set, load_items
from silent_dissent.entity_prompts import EntityState

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--split", choices=["dev", "test"], default="dev")
ap.add_argument("--format", help="default: the entity prereg's format, else plain")
ap.add_argument("--limit", type=int)
ap.add_argument("--k", type=int, help="hit@k for the printed tables (default entity.select.k)")
ap.add_argument("--no-extra-lenses", action="store_true", help="only the configured lens and the logit lens")
args = ap.parse_args()
if args.split == "test":
    raise SystemExit("the diagnosis is for choosing a readout: dev split only")

cfg = load_config(args.config)
ec = cfg["entity"]
prereg = Path(cfg.get("entity_prereg") or "")
fmt = args.format or (json.loads(prereg.read_text())["format"] if prereg.is_file() else "plain")
k = args.k or ec["select"]["k"]
lm, lens = setup(cfg)
lenses = lens_set(cfg, lm.model, lens, not args.no_extra_lenses)
items = load_items(cfg, args.split, args.limit)
gen_bs = ec.get("gen_batch_size", cfg["batch_size"])
print(f"{len(items)} items ({args.split}, every category), format {fmt}, lenses {list(lenses)}")

solo = [EntityState(f, p, c, "solo") for f, p, c, _ in items]
prompts, offs, moffs = readout_inputs(lm.tok, solo, fmt, extended=True)
window = [max(-o, -m) for o, m in zip(offs, moffs)]
hop = [EntityState(dataclasses.replace(f, composed=f.r1_prompt), p, c, "solo") for f, p, c, _ in items]
hop_prompts = [st.render(fmt, lm.tok) for st in hop]
answers = generate(lm, prompts, ec.get("gen_tokens", 12), gen_bs, desc="round 0")
hop_answers = generate(lm, hop_prompts, ec.get("gen_tokens", 12), gen_bs, desc="hop 1")

layers = list(range(lm.n_layers))
win, h1 = {}, {}
for mode in PROBE_MODES:
    roles = [role_probes(lm.tok, f, p, c, mode) for f, p, c, _ in items]
    tl = timeline_readout(lm, lenses, prompts, window, roles, layers, cfg["batch_size"])
    for name in lenses:
        win[f"{mode}__{name}"] = np.concatenate(tl[name]) if items else np.zeros((0, len(layers), len(ROLES)))
    hr = entity_readout(lm, lenses, hop_prompts, [-1] * len(items), roles, cfg["batch_size"])
    for name in lenses:
        h1[f"{mode}__{name}"] = hr[name][:, 1]

records, offset = [], 0
for i, ((f, p, c, _), a, ha) in enumerate(zip(items, answers, hop_answers)):
    records.append({"uid": f.uid, "category": f.category, "e2": f.e2, "e2_type": f.e2_type, "e3_type": f.e3_type,
                    "answer": a, "outcome": classify(a, f, p), "hop1_answer": ha, "hop1_ok": matches(ha, f.e2_aliases),
                    "offset": offset, "window": window[i], "subject_index": window[i] + offs[i],
                    "mention_index": window[i] + moffs[i]})
    offset += window[i]
out = entity_dir(cfg) / f"bridgeprobe_{args.split}"
out.mkdir(parents=True, exist_ok=True)
with open(out / "records.jsonl", "w", encoding="utf-8") as fh:
    for r in records:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
np.savez_compressed(out / "window.npz", **win)
np.savez_compressed(out / "hop1.npz", **h1)
try:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
except Exception:
    commit = None
(out / "info.json").write_text(json.dumps({"split": args.split, "format": fmt, "lenses": list(lenses),
                                           "probe_modes": list(PROBE_MODES), "n_items": len(items),
                                           "config": args.config, "code_commit": commit}, indent=1) + "\n")


def positions(arr: np.ndarray, r: dict) -> dict[str, np.ndarray]:
    """[L, R] ranks at each position rule for one record from its window [T, L, R]."""
    w = arr[r["offset"] : r["offset"] + r["window"]]
    return {"subject": w[r["subject_index"]], "mention": w[r["mention_index"]], "last": w[-1],
            "span": w[r["mention_index"]:].min(0)}


def contrast_curve(ranks: np.ndarray) -> np.ndarray:
    """ranks [n, L, R] -> mean orig_bridge - ctrl_bridge hit@k per layer."""
    if not len(ranks):
        return np.full(len(layers), np.nan)
    h = EM.hits(ranks, k)
    with warnings.catch_warnings():  # a layer with no readable record is NaN
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(h[..., ROLES.index("orig_bridge")] - h[..., ROLES.index("ctrl_bridge")], 0)


show = list(range(max(0, lm.n_layers - 24), lm.n_layers))
fmt_curve = lambda v: " ".join(f"{x:5.2f}" for x in v[show])
print(f"\norig_bridge - ctrl_bridge, hit@{k}, round-0 correct records; layers {show[0]}..{show[-1]}")
types = sorted({r["e2_type"] for r in records}, key=lambda t: -sum(r["e2_type"] == t for r in records))
for t in types:
    right = [r for r in records if r["e2_type"] == t and r["outcome"] == "original"]
    hop_ok = [i for i, r in enumerate(records) if r["e2_type"] == t and r["hop1_ok"]]
    print(f"\n=== bridge type {t}: {len(right)} correct at round 0 (of {sum(r['e2_type'] == t for r in records)}), "
          f"hop 1 correct {len(hop_ok)}")
    for mode in PROBE_MODES:
        for name in lenses:
            arr = win[f"{mode}__{name}"]
            per = [positions(arr, r) for r in right]
            for pos in ("subject", "mention", "last", "span"):
                v = contrast_curve(np.stack([x[pos] for x in per]) if per else np.zeros((0,)))
                print(f"  {mode:9s} {name:14s} {pos:8s} {fmt_curve(v)}")
            print(f"  {mode:9s} {name:14s} {'hop1':8s} {fmt_curve(contrast_curve(h1[f'{mode}__{name}'][hop_ok]))}"
                  "   <- positive control: the bridge is the next token")
print(f"\nper category (mode words, {cfg['lens']['name']}): max over layers {show[0]}.. of the contrast")
name = cfg["lens"]["name"]
for cat in sorted({r["category"] for r in records}):
    right = [r for r in records if r["category"] == cat and r["outcome"] == "original"]
    if not right:
        continue
    per = [positions(win[f"words__{name}"], r) for r in right]
    best = {pos: contrast_curve(np.stack([x[pos] for x in per]))[show] for pos in ("subject", "mention", "last", "span")}
    print(f"  {cat:28s} {right[0]['e2_type']:12s} n={len(right):4d}  " +
          "  ".join(f"{pos} {np.nanmax(v):.2f}@L{show[int(np.nanargmax(v))]}" if not np.isnan(v).all() else f"{pos} -"
                    for pos, v in best.items()))
print(f"-> {out}")
