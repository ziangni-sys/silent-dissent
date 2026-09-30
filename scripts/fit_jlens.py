"""Fit the Jacobian lens for the config's model and write it to lens.kwargs.path.

    python scripts/fit_jlens.py --config C                     # single GPU
    python scripts/fit_jlens.py --config C --shard 0/4         # GPU 0 of 4 (disjoint prompt slices)
    python scripts/fit_jlens.py --config C --merge             # combine the shard files

Interrupted runs resume from <path>.ckpt (or <shard file>.ckpt).
"""
import argparse
import glob
import logging

from silent_dissent import jlens
from silent_dissent.config import load_config
from silent_dissent.model import load_model

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--shard", default=None, help="i/n: fit prompts i, i+n, i+2n, ...")
ap.add_argument("--merge", action="store_true", help="merge all <path>.shard*of* files into <path>")
ap.add_argument("--n-prompts", type=int, default=None, help="override jlens.n_prompts")
args = ap.parse_args()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

cfg = load_config(args.config)
jc = cfg["jlens"]
out = cfg["lens"]["kwargs"]["path"]
n_prompts = args.n_prompts or jc["n_prompts"]

if args.merge:
    shards = sorted(p for p in glob.glob(f"{out}.shard*of*") if not p.endswith(".ckpt"))
    n_expected = {p.rsplit("of", 1)[1] for p in shards}
    if not shards or len(n_expected) != 1 or len(shards) != int(n_expected.pop()):
        raise SystemExit(f"incomplete or mixed shard set: {shards}")
    jlens.merge(shards, out)
    print(f"merged {len(shards)} shards -> {out}")
    raise SystemExit

prompts = jlens.load_corpus(jc["corpus"], n_prompts, cfg["seed"])
if args.shard:
    i, n_shards = map(int, args.shard.split("/"))
    prompts = prompts[i::n_shards]
    out = f"{out}.shard{i}of{n_shards}"

m = cfg["model"]
model, tok = load_model(m["name"], m.get("dtype", "bfloat16"), m.get("device_map", "auto"),
                        m.get("chat_template_kwargs"))
jlens.fit(model, tok, prompts, out, dim_batch=jc.get("dim_batch", 8), max_seq_len=jc.get("max_seq_len", 128),
          skip_first=jc.get("skip_first", jlens.SKIP_FIRST), target_layer=jc.get("target_layer"))
print(f"fitted on {len(prompts)} prompts -> {out}")
