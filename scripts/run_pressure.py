"""Scripted-peer pressure: one output file per grid in the config."""
import argparse
import json

from silent_dissent.config import get_items, load_config, setup
from silent_dissent.experiments import (build_states, expand_grid, generate_reasons, read_jsonl, run_pressure,
                                        write_jsonl)

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--grid", action="append", help="grid name(s) to run; default all")
ap.add_argument("--limit", type=int, help="debug: only the first N baseline items")
args = ap.parse_args()

cfg = load_config(args.config)
lm, lens = setup(cfg)
items = {it.item_id: it for it in get_items(cfg)}
baseline = read_jsonl(cfg["out_dir"] / "baseline.jsonl")[: args.limit]
grids = cfg["pressure"]["grids"]

for name in args.grid or list(grids):
    settings = expand_grid(grids[name])
    states, meta = build_states(items, baseline, settings, cfg["seed"])

    reasons = None
    if any(s["peer_style"] == "with_reason" for s in settings):
        path = cfg["out_dir"] / "reasons.json"
        reasons = json.loads(path.read_text()) if path.exists() else {}
        need: dict[str, set] = {}
        for st in states:
            if st.peer_style == "with_reason":
                for l in set(st.peer_answers()) - set(reasons.get(st.item.item_id, {})):
                    need.setdefault(st.item.item_id, set()).add(l)
        if need:
            new = generate_reasons(lm, [items[i] for i in need], need, cfg["batch_size"])
            for i, d in new.items():
                reasons.setdefault(i, {}).update(d)
            path.write_text(json.dumps(reasons, indent=1, ensure_ascii=False))
        states, meta = build_states(items, baseline, settings, cfg["seed"], reasons)

    print(f"[{name}] {len(states)} conversations x {cfg['pressure']['rounds']} rounds")
    records = run_pressure(lm, lens, states, meta, cfg["pressure"]["rounds"], cfg["batch_size"])
    write_jsonl(records, cfg["out_dir"] / f"pressure_{name}.jsonl")
