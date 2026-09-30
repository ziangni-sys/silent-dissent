"""Write a config on which scripts/check_model.py can run for an entity-study model.

check_model.py tests model-level properties (chat template, padding invariance, lens = model output at
the last layer, letter format) on the letter task, whose items and layer rule an entity config lacks.
This copies `data` and `layer_selection` from the letter-task config into the entity config and points
out_dir at results/preflight/<name>. --extra-lens NAME uses that entry of entity.extra_lenses as the lens,
so the check can run before a refit lens file exists.

    python scripts/preflight_config.py --config configs/gemma4_e4b_it_entity.yaml --extra-lens jlens_base
    python scripts/check_model.py --config results/preflight/gemma4_e4b_it_entity.yaml
"""
import argparse
from pathlib import Path

import yaml

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--letter-config", default="configs/qwen35_4b.yaml")
ap.add_argument("--extra-lens", help="name of an entity.extra_lenses entry to use as the lens")
ap.add_argument("--out-dir", default="results/preflight")
args = ap.parse_args()

cfg = yaml.safe_load(Path(args.config).read_text())
letter = yaml.safe_load(Path(args.letter_config).read_text())
name = Path(args.config).stem
cfg["data"], cfg["layer_selection"] = letter["data"], letter["layer_selection"]
cfg["out_dir"] = str(Path(args.out_dir) / name)
if args.extra_lens:
    extra = next(e for e in cfg["entity"]["extra_lenses"] if e["name"] == args.extra_lens)
    cfg["lens"] = {"name": "jlens", "kwargs": {k: v for k, v in extra.items() if k != "name"}}
out = Path(args.out_dir) / f"{name}.yaml"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(yaml.safe_dump(cfg, sort_keys=False))
print(out)
