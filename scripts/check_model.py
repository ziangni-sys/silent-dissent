"""Pre-flight check on the real model: run this before any full experiment.

    python scripts/check_model.py --config configs/qwen35_4b.yaml [--limit 50]

Checks: chat template + forced prefix, single-token letters, lens file, left-padding
invariance, J-lens == model output at the last layer, letter-format rate and
accuracy, J-lens vs logit-lens agreement by layer, throughput. Writes only the item cache (out_dir/items.jsonl).
"""
import argparse
import time

import numpy as np

from silent_dissent.config import get_items, load_config, setup
from silent_dissent.experiments import run_baseline
from silent_dissent.lenses import LogitLens
from silent_dissent.metrics import agreement_curve
from silent_dissent.prompts import render, solo_messages

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--limit", type=int, default=50)
args = ap.parse_args()
cfg = load_config(args.config)
bs = cfg["batch_size"]
problems = []

lm, lens = setup(cfg)
print(f"model {cfg['model']['name']}: {lm.n_layers} layers | lens {type(lens).__name__}", end="")
if hasattr(lens, "maps"):
    print(f" with {len(lens.maps)} fitted layers (n_prompts={getattr(lens, 'n_prompts', '?')})")
    if len(lens.maps) != lm.n_layers - 1:
        problems.append(f"lens has {len(lens.maps)} layers, expected {lm.n_layers - 1}")
else:
    print()
print("letter token ids:", dict(zip(lm.letters, lm.letter_ids)))

items = get_items(cfg)[: args.limit]
example = render(lm.tok, solo_messages(items[0]))
print("\n--- rendered prompt (tail) ---\n" + example[-300:] + "\n------------------------------")
if example.rfind("<think>") > example.rfind("</think>"):
    problems.append("prompt ends inside an open <think> block: set model.chat_template_kwargs.enable_thinking: false")

# left-padding invariance: same prompts alone vs in one padded batch
texts = [render(lm.tok, solo_messages(it)) for it in items[:bs]]
alone = [lm.readout([t], lens).final_letter_logits[0] for t in texts]
together = lm.readout(texts, lens).final_letter_logits
diff = max(float((a - together[i]).abs().max()) for i, a in enumerate(alone))
flips = sum(int(a.argmax()) != int(together[i].argmax()) for i, a in enumerate(alone))
print(f"\npadding: max |alone - batched| letter logit = {diff:.3f}, argmax changes {flips}/{len(texts)}")
if flips > 1 or diff > 1.0:
    problems.append("batched readout differs from unbatched (left padding?)")

t0 = time.perf_counter()
base = run_baseline(lm, lens, items, bs)
dt = (time.perf_counter() - t0) / len(items)
last = max(abs(x - y) for r in base for x, y in zip(r["lens_logits"][-1], r["final_logits"]))
print(f"last-layer lens vs model output: max diff {last:.3f}")
if last > 0.5:
    problems.append("lens at the last layer does not reproduce the model output")

acc = np.mean([r["original_correct"] for r in base])
fmt = np.mean([r["top_is_letter"] for r in base])
print(f"accuracy {acc:.2f} | letter-format {fmt:.2f} (unrestricted top-1 is a letter) | n={len(base)}")
if fmt < 0.8:
    problems.append(f"only {fmt:.0%} of answers start with a letter: check the template / prefix")

logit_base = run_baseline(lm, LogitLens(lm.model), items, bs)
cj, cl = agreement_curve(base), agreement_curve(logit_base)
print("\nlens top-1 == stated answer, by layer")
print("layer  " + " ".join(f"{l:>4}" for l in range(len(cj))))
print("jlens  " + " ".join(f"{x:4.2f}" for x in cj))
print("logit  " + " ".join(f"{x:4.2f}" for x in cl))
tau = cfg["layer_selection"]["tau"]
first = lambda c: next((l for l, x in enumerate(c) if x >= tau), None)
print(f"first layer >= tau={tau}: jlens {first(cj)}, logit {first(cl)}")

print(f"\nthroughput: {dt * 1000:.0f} ms per readout at batch {bs} (solo prompts; multi-round prompts are longer)")
print(f"  -> ~{250_000 * dt * 1.5 / 3600:.1f} h for ~250k readouts at 1.5x this cost (the default full run)")

print("\n" + ("ALL CHECKS PASSED" if not problems else "PROBLEMS:\n  - " + "\n  - ".join(problems)))
raise SystemExit(1 if problems else 0)
