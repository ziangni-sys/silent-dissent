"""YAML config loading plus shared setup for the scripts."""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from .data import MCQItem, load_items, save_items
from .lenses import build_lens
from .model import LM, load_model
from .prompts import ANSWER_PREFIX


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("seed", 0)
    cfg.setdefault("batch_size", 8)
    cfg["out_dir"] = Path(cfg["out_dir"])
    return cfg


def setup(cfg: dict) -> tuple[LM, object]:
    m = cfg["model"]
    model, tok = load_model(m["name"], m.get("dtype", "bfloat16"), m.get("device_map", "auto"),
                            m.get("chat_template_kwargs"))
    letters = list("ABCDEFGH"[: cfg["data"].get("n_choices", 4)])
    lm = LM(model, tok, letters, ANSWER_PREFIX)
    lens = build_lens(model, cfg["lens"]["name"], **cfg["lens"].get("kwargs", {}))
    return lm, lens


def get_items(cfg: dict) -> list[MCQItem]:
    """Load (and cache) the item pool defined in the config."""
    path = cfg["out_dir"] / "items.jsonl"
    if path.exists():
        return list(load_items(str(path), n_choices=None))
    items = []
    for src in cfg["data"]["sources"]:
        items += load_items(src["name"], src.get("split", "test"), src.get("subset"),
                            cfg["data"].get("n_choices", 4), src.get("limit"), cfg["seed"])
    save_items(items, path)
    return items


def load_prereg(cfg: dict) -> dict:
    path = Path(cfg["prereg"])
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run scripts/select_layer.py and commit the result first")
    return json.loads(path.read_text())
