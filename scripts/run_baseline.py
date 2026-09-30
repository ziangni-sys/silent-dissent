"""Round 0: independent answers + lens readouts; then stratify by initial correctness."""
import argparse

from silent_dissent.config import get_items, load_config, setup
from silent_dissent.experiments import run_baseline, stratify, write_jsonl

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
args = ap.parse_args()

cfg = load_config(args.config)
lm, lens = setup(cfg)
items = get_items(cfg)
records = run_baseline(lm, lens, items, cfg["batch_size"])
write_jsonl(records, cfg["out_dir"] / "baseline_all.jsonl")

kept = stratify(records, cfg["data"]["n_per_stratum"], cfg["seed"], cfg["data"].get("require_letter_format", True))
if not kept:
    raise SystemExit("no items survived stratification (is the top token a letter? see require_letter_format)")
write_jsonl(kept, cfg["out_dir"] / "baseline.jsonl")
acc = sum(r["original_correct"] for r in records) / len(records)
fmt = sum(r["top_is_letter"] for r in records) / len(records)
n_ok = sum(r["original_correct"] for r in kept)
print(f"{len(records)} items | accuracy {acc:.3f} | letter-format {fmt:.3f} | kept {n_ok} correct + {len(kept) - n_ok} wrong")
