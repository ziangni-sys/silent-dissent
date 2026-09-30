"""Fix the readout layer from round-0 data only, and write the pre-registration file.

Commit the resulting prereg/*.json BEFORE running the pressure experiments.
"""
import argparse
import datetime
import json
import subprocess
from pathlib import Path

from silent_dissent.config import load_config
from silent_dissent.experiments import read_jsonl
from silent_dissent.metrics import agreement_curve, select_layer

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--force", action="store_true", help="overwrite an existing prereg file")
args = ap.parse_args()

cfg = load_config(args.config)
path = Path(cfg["prereg"])
if path.exists() and not args.force:
    raise SystemExit(f"{path} already exists; pre-registration is fixed (use --force only if you mean it)")

baseline = read_jsonl(cfg["out_dir"] / "baseline.jsonl")
sel = cfg["layer_selection"]
curve = agreement_curve(baseline)
layer = select_layer(baseline, sel["tau"])
try:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
except Exception:
    commit = None

prereg = {
    "model": cfg["model"]["name"],
    "lens": cfg["lens"]["name"],
    "layer": layer,
    "n_layers": len(curve),
    "rule": f"earliest layer with lens-top1 == output on >= {sel['tau']} of round-0 items",
    "tau": sel["tau"],
    "delta": sel["delta"],
    "definition": {
        "strict": "flipped and original letter top-1 among letters at layer",
        "lenient": "flipped, original in top-2 among letters, logit(stated)-logit(original) < delta",
    },
    "position": "last prompt token (after the forced prefix 'The answer is')",
    "agreement_curve": [round(float(x), 4) for x in curve],
    "n_items": len(baseline),
    "created": datetime.datetime.now().isoformat(timespec="seconds"),
    "code_commit": commit,
}
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(prereg, indent=2) + "\n")
print(f"layer {layer}/{len(curve) - 1} -> {path}")
print("agreement by layer:", " ".join(f"{x:.2f}" for x in curve))
