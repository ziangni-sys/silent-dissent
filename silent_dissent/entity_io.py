"""Shared plumbing for the entity-study scripts: item files, lens sets, run directories."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .entity_facts import Fact, role_probes
from .entity_prompts import EntityState
from .lenses import JLens, LogitLens


def entity_dir(cfg: dict) -> Path:
    return Path(cfg["out_dir"]) / "entity"


def prereg_categories(cfg: dict) -> list[str] | None:
    """Categories fixed in the entity prereg (None: no prereg, or all categories)."""
    path = Path(cfg.get("entity_prereg") or "")
    return json.loads(path.read_text()).get("categories") if path.is_file() else None


ADDENDA = ("rounds", "bridges", "own")


def addendum_path(cfg: dict, part: str) -> Path:
    """prereg/<name>_entity.json -> prereg/<name>_entity_<part>.json (written once by
    scripts/entity_addendum.py; committed and pushed before its test run)."""
    if part not in ADDENDA:
        raise ValueError(f"unknown addendum {part!r}")
    p = Path(cfg["entity_prereg"])
    return p.with_name(f"{p.stem}_{part}{p.suffix}")


def load_addendum(cfg: dict, part: str) -> dict | None:
    p = addendum_path(cfg, part)
    return json.loads(p.read_text()) if p.is_file() else None


def load_items(cfg: dict, split: str, limit: int | None = None, categories: list[str] | None = None,
               bridge_types: list[str] | None = None) -> list[tuple[Fact, Fact, Fact, Fact | None]]:
    """(fact, peer, control, sibling) written by scripts/entity_build.py, optionally only `categories`
    and only bridges of `bridge_types` (Fact.e2_type, e.g. person)."""
    out = []
    with open(entity_dir(cfg) / f"items_{split}.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if categories is not None and d["fact"]["category"] not in categories:
                continue
            if bridge_types is not None and d["fact"].get("e2_type") not in bridge_types:
                continue
            out.append((Fact.from_dict(d["fact"]), Fact.from_dict(d["peer"]), Fact.from_dict(d["control"]),
                        Fact.from_dict(d["sibling"]) if d.get("sibling") else None))
    return out[:limit]


def lens_set(cfg: dict, model, lens, extra_lenses: bool = True) -> dict:
    """The configured lens, the logit lens, and (extra_lenses) any extra fitted J-lenses
    (entity.extra_lenses: [{name, path}] or [{name, repo, filename, revision}] for a lens on the
    Hub), e.g. one fit on chat text for the corpus comparison."""
    lenses = {cfg["lens"]["name"]: lens}
    lenses.setdefault("logit", LogitLens(model))
    for extra in (cfg.get("entity", {}).get("extra_lenses", []) or []) if extra_lenses else []:
        if "repo" in extra:
            lenses[extra["name"]] = JLens(model, repo=extra["repo"], filename=extra["filename"],
                                          revision=extra.get("revision"))
        elif Path(extra["path"]).exists():
            lenses[extra["name"]] = JLens(model, path=extra["path"])
        else:
            print(f"extra lens {extra['name']} skipped: {extra['path']} not found")
    return lenses


def states_after_round0(tok, items, answers: list[str], specs: list[dict], n_peers: list[int],
                        require_correct: bool = True, probe_mode: str = "words", later: str = "repeat"):
    """For every item whose round-0 answer is correct, one EntityState per spec
    ({condition, style, hidden_own, absent_own}) and n_peers value, with the round-0 answer recorded.
    require_correct=False (debug configs only) keeps every item."""
    from .entity_experiments import classify

    states, meta, roles = [], [], []
    for (f, p, c, sib), a in zip(items, answers):
        if require_correct and classify(a, f, p) != "original":
            continue
        probes = role_probes(tok, f, p, c, probe_mode)
        for spec in specs:
            if spec["condition"] == "retention" and sib is None:
                continue
            peers = n_peers if spec["condition"] in ("pressure", "agree", "mention", "mention_bridge") else [0]
            for n in peers:
                st = EntityState(f, p, c, spec["condition"], spec.get("style"), n_peers=max(n, 1), sibling=sib,
                                 hidden_own=spec.get("hidden_own", False), absent_own=spec.get("absent_own", False),
                                 later=later)
                st.record(a)
                states.append(st)
                meta.append({"uid": f.uid, "category": f.category, "condition": spec["condition"],
                             "style": spec.get("style"), "hidden_own": spec.get("hidden_own", False),
                             "absent_own": spec.get("absent_own", False), "n_peers": n})
                roles.append(probes)
    return states, meta, roles


def save_run(path: Path, records: list[dict], arrays: dict[str, list[np.ndarray]], info: dict) -> None:
    """records.jsonl, ranks.npz (one array per lens, rows aligned with records), info.json."""
    path.mkdir(parents=True, exist_ok=True)
    with open(path / "records.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    np.savez_compressed(path / "ranks.npz", **{k: np.concatenate(v) for k, v in arrays.items() if v})
    (path / "info.json").write_text(json.dumps(info, indent=1, ensure_ascii=False) + "\n")


def load_run(path: Path) -> tuple[list[dict], dict[str, np.ndarray], dict]:
    with open(path / "records.jsonl", encoding="utf-8") as f:
        records = [json.loads(l) for l in f if l.strip()]
    arrays = dict(np.load(path / "ranks.npz"))
    info = json.loads((path / "info.json").read_text()) if (path / "info.json").exists() else {}
    return records, arrays, info
