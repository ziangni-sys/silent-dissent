"""Entity study E3: when does the original premise fade? Readout at every token of the last round.

Round 0, then round 1 of the pressure styles (entity.timeline.styles) and the mention baseline,
with entity.timeline.n_peers peers. For each round-1 prompt the lenses are read at every token
from the agent's own previous statement to the end (own statement, each peer turn, moderator,
the statement the agent is starting) at entity.timeline.layers (default: the prereg band), and
the round-1 answer is recorded, so curves can be split by whether the agent then gives in.

Writes <out_dir>/entity/timeline_<split>/{records.jsonl, timeline.npz, info.json}; timeline.npz
holds per lens the per-token arrays concatenated ([sum T, L', R]) and `offsets`.

    python scripts/entity_timeline.py --config configs/qwen35_4b_entity.yaml --split test
"""
import argparse
import json
from pathlib import Path

import numpy as np

from silent_dissent.config import load_config, setup
from silent_dissent.entity_experiments import classify, generate, timeline_readout, token_labels
from silent_dissent.entity_io import entity_dir, lens_set, load_items, prereg_categories, states_after_round0
from silent_dissent.entity_prompts import EntityState

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--split", choices=["dev", "test"], required=True)
ap.add_argument("--limit", type=int, default=None)
args = ap.parse_args()

cfg = load_config(args.config)
ec, tc = cfg["entity"], cfg["entity"]["timeline"]
prereg = Path(cfg["entity_prereg"])
pr = json.loads(prereg.read_text()) if prereg.exists() else {}
fmt = pr.get("format", ec["formats"][0])
lm, lens = setup(cfg)
gen_bs = ec.get("gen_batch_size", cfg["batch_size"])
layers = tc.get("layers") or pr.get("band") or list(range(lm.n_layers // 2, lm.n_layers))
lenses = lens_set(cfg, lm.model, lens)
items = load_items(cfg, args.split, args.limit or tc.get("limit"), prereg_categories(cfg))

solo = [EntityState(f, p, c, "solo").render(fmt, lm.tok) for f, p, c, _ in items]
answers = generate(lm, solo, ec.get("gen_tokens", 12), gen_bs, desc="round 0")
specs = [{"condition": "pressure", "style": s} for s in tc["styles"]] + [{"condition": "mention"}]
states, meta, roles = states_after_round0(lm.tok, items, answers, specs, [tc["n_peers"]],
                                          require_correct=ec.get("require_correct_round0", True))
for st in states:
    st.advance()
prompts, windows, labels = [], [], []
for st in states:
    text, spans = st.render_with_segments(fmt, lm.tok)
    n, lab = token_labels(lm.tok, text, spans, spans[0][1])
    prompts.append(text)
    windows.append(n)
    labels.append(lab)
print(f"{len(states)} round-1 prompts ({fmt}), layers {layers}, window {min(windows)}-{max(windows)} tokens")
tl = timeline_readout(lm, lenses, prompts, windows, roles, layers, cfg["batch_size"])
round1 = generate(lm, prompts, ec.get("gen_tokens", 12), gen_bs, desc="round 1")

out = entity_dir(cfg) / f"timeline_{args.split}"
out.mkdir(parents=True, exist_ok=True)
offsets = np.cumsum([0] + windows)
with open(out / "records.jsonl", "w", encoding="utf-8") as f:
    for i, (st, m) in enumerate(zip(states, meta)):
        f.write(json.dumps({**m, "split": args.split, "format": fmt, "round": 1, "answer": round1[i],
                            "outcome": classify(round1[i], st.fact, st.peer), "labels": labels[i],
                            "offset": int(offsets[i]), "window": windows[i], "roles": roles[i]},
                           ensure_ascii=False) + "\n")
np.savez_compressed(out / "timeline.npz", offsets=offsets, layers=np.array(layers),
                    **{name: np.concatenate(v) for name, v in tl.items()})
(out / "info.json").write_text(json.dumps({"format": fmt, "layers": layers, "styles": tc["styles"],
                                           "n_peers": tc["n_peers"], "lenses": list(lenses)}, indent=1) + "\n")
print(f"-> {out}")
