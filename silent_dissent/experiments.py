"""Experiment loops: solo baseline, scripted-peer pressure, and injection.

All loops write one JSON record per (item, setting, round) so that analysis
never needs the model. Lens arrays are stored restricted to the answer letters:

    lens_logits[layer][k]   lens logit of letter k at `layer`
    lens_rank[layer][k]     rank of letter k in the full-vocabulary lens distribution
"""
from __future__ import annotations

import contextlib
import itertools
import json
import random
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
from tqdm import tqdm

from .data import MCQItem
from .intervene import inject_last_position, pick_direction
from .model import LM
from .prompts import CONDITIONS, DebateState, choose_control, choose_target, format_question, render, solo_messages


# --------------------------------------------------------------------------- utils


def write_jsonl(records: Iterable[dict], path: str | Path, mode: str = "w") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, mode) as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def item_rng(seed: int, *keys) -> random.Random:
    """Deterministic per-item RNG so targets are identical across conditions/runs."""
    return random.Random("-".join(map(str, (seed, *keys))))


def batched_readout(
    lm: LM,
    lens,
    texts: list[str],
    batch_size: int,
    ctx_fn: Callable[[list[int]], contextlib.AbstractContextManager] | None = None,
    desc: str = "",
) -> list[dict]:
    """Readout for every text; batches are length-sorted, results in input order.

    ctx_fn(indices) may return a context manager (e.g. an injection hook) that
    is active for the batch containing those input indices.
    """
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    results: list[dict | None] = [None] * len(texts)
    for start in tqdm(range(0, len(order), batch_size), desc=desc, leave=False):
        idx = order[start : start + batch_size]
        ctx = ctx_fn(idx) if ctx_fn else contextlib.nullcontext()
        with ctx:
            ro = lm.readout([texts[i] for i in idx], lens)
        for j, i in enumerate(idx):
            final = ro.final_letter_logits[j]
            results[i] = {
                "stated": lm.letters[int(final.argmax())],
                "final_logits": [round(x, 4) for x in final.tolist()],
                "top_is_letter": bool(ro.final_top_is_letter[j]),
                "lens_logits": [[round(x, 3) for x in row] for row in ro.lens_letter_logits[j].tolist()],
                "lens_rank": ro.lens_vocab_rank[j].tolist(),
            }
    return results  # type: ignore[return-value]


def batched_position_readout(
    lm: LM, lenses: dict, texts: list[str], batch_size: int, n_last: int, desc: str = ""
) -> tuple[list[dict], dict[str, np.ndarray], np.ndarray]:
    """`LM.readout_positions` over length-sorted batches, results in input order.

    Returns per-text {stated, final_logits, top_is_letter}, per-lens float16 arrays
    [N, n_last, n_layers, K] of letter logits, and the tail token ids [N, n_last].
    """
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    outs: list[dict | None] = [None] * len(texts)
    arrays: dict[str, np.ndarray] = {}
    tails = np.zeros((len(texts), n_last), dtype=np.int64)
    for start in tqdm(range(0, len(order), batch_size), desc=desc, leave=False):
        idx = order[start : start + batch_size]
        ro = lm.readout_positions([texts[i] for i in idx], lenses, n_last)
        for name, x in ro.lens_letter_logits.items():
            if name not in arrays:
                arrays[name] = np.zeros((len(texts), *x.shape[1:]), dtype=np.float16)
            arrays[name][idx] = x.numpy().astype(np.float16)
        tails[idx] = ro.tail_ids.numpy()
        for j, i in enumerate(idx):
            final = ro.final_letter_logits[j]
            outs[i] = {"stated": lm.letters[int(final.argmax())],
                       "final_logits": [round(x, 4) for x in final.tolist()],
                       "top_is_letter": bool(ro.final_top_is_letter[j])}
    return outs, arrays, tails  # type: ignore[return-value]


def _item_fields(it: MCQItem) -> dict:
    return {"item_id": it.item_id, "dataset": it.dataset, "subject": it.subject, "gold": it.gold, "letters": it.letters}


# ------------------------------------------------------------------------ baseline


def run_baseline(lm: LM, lens, items: list[MCQItem], batch_size: int) -> list[dict]:
    """Round 0: independent answer with no peers."""
    texts = [render(lm.tok, solo_messages(it)) for it in items]
    outs = batched_readout(lm, lens, texts, batch_size, desc="baseline")
    records = []
    for it, o in zip(items, outs):
        records.append({**_item_fields(it), "round": 0, "original": o["stated"], "original_correct": o["stated"] == it.gold, **o})
    return records


def stratify(baseline: list[dict], n_per_stratum: int, seed: int, require_letter: bool = True) -> list[dict]:
    """Up to n initially-correct and n initially-wrong items (deterministic)."""
    pool = [r for r in baseline if r["top_is_letter"] or not require_letter]
    rng = random.Random(seed)
    out = []
    for correct in (True, False):
        group = [r for r in pool if r["original_correct"] == correct]
        rng.shuffle(group)
        out += group[:n_per_stratum]
    return out


# ---------------------------------------------------------------- peer reasoning


def generate_reasons(lm: LM, items: list[MCQItem], letters_needed: dict[str, set[str]], batch_size: int) -> dict:
    """{item_id: {letter: one-sentence argument for that letter}} for with_reason peers."""
    jobs = [(it, l) for it in items for l in sorted(letters_needed.get(it.item_id, ()))]
    texts = []
    for it, l in jobs:
        msg = format_question(it).replace("Reply with a single letter.", "") + (
            f"Argue in one or two sentences that the correct answer is {l}. "
            "Do not mention any other option and do not restate the letter."
        )
        texts.append(render(lm.tok, [{"role": "user", "content": msg}], prefix=""))
    reasons: dict[str, dict[str, str]] = {}
    for start in tqdm(range(0, len(texts), batch_size), desc="reasons", leave=False):
        outs = lm.generate(texts[start : start + batch_size], max_new_tokens=64)
        for (it, l), text in zip(jobs[start : start + batch_size], outs):
            reasons.setdefault(it.item_id, {})[l] = " ".join(text.split())
    return reasons


# ------------------------------------------------------------------------ pressure


def expand_grid(grid: dict) -> list[dict]:
    """Cartesian product of a {key: list} config block."""
    keys = list(grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]


def build_states(items: dict[str, MCQItem], baseline: list[dict], settings: list[dict], seed: int, reasons=None):
    """One DebateState per (item, setting); skips undefined combinations."""
    states, meta, seen = [], [], set()
    for b in baseline:
        it = items[b["item_id"]]
        for s in settings:
            cond, mode = s["condition"], s["target_mode"]
            assert cond in CONDITIONS, cond
            target = choose_target(b["original"], it.gold, it.letters, mode, item_rng(seed, it.item_id, mode, "target"))
            if target is None:
                continue
            if cond == "split" and s["n_peers"] < 2:
                continue
            if cond == "instructed" and (s["n_peers"] != 0 or s["peer_style"] != "answer_only"):
                continue  # no peers, so only the n_peers = 0, answer_only cell exists
            control = choose_control(b["original"], target, it.gold, it.letters, item_rng(seed, it.item_id, mode, "control"))
            key = (it.item_id, cond, s["n_peers"], s["peer_style"], target if cond != "agree" else None)
            if key in seen:  # e.g. `agree` does not depend on the target
                continue
            seen.add(key)
            states.append(
                DebateState(
                    item=it,
                    condition=cond,
                    n_peers=s["n_peers"],
                    original=b["original"],
                    target=target,
                    peer_style=s["peer_style"],
                    control=control,
                    reasons=(reasons or {}).get(it.item_id, {}),
                )
            )
            meta.append({**_item_fields(it), "original": b["original"], "original_correct": b["original_correct"], **s,
                         "target": target, "control": control})
    return states, meta


def run_pressure(lm: LM, lens, states: list[DebateState], meta: list[dict], rounds: int, batch_size: int) -> list[dict]:
    records = []
    for r in range(1, rounds + 1):
        for st in states:
            st.advance()
        texts = [render(lm.tok, st.messages()) for st in states]
        outs = batched_readout(lm, lens, texts, batch_size, desc=f"pressure round {r}")
        for st, m, o in zip(states, meta, outs):
            st.record(o["stated"])
            records.append({**m, "round": r, "stated_history": list(st.stated), **o})
    return records


def replay_state(item: MCQItem, rec: dict, reasons=None) -> DebateState:
    """Rebuild the exact conversation that produced `rec` (up to its round)."""
    st = DebateState(
        item=item,
        condition=rec["condition"],
        n_peers=rec["n_peers"],
        original=rec["original"],
        target=rec["target"],
        peer_style=rec["peer_style"],
        control=rec["control"],
        reasons=(reasons or {}).get(item.item_id, {}),
    )
    hist = rec["stated_history"]
    for r in range(1, rec["round"] + 1):
        st.advance()
        if r < rec["round"]:
            st.record(hist[r])
    return st


# -------------------------------------------------------------------- intervention


def run_intervention(
    lm: LM,
    lens,
    items: dict[str, MCQItem],
    source: list[dict],
    layer: int,
    alphas: list[float],
    kinds: list[str],
    batch_size: int,
    seed: int,
    contrastive: bool = True,
    reasons=None,
) -> list[dict]:
    """Re-run the prompt behind each source record with an injection at `layer`.

    `source` are pressure records (rebuilt via replay) or baseline records
    (round 0, solo prompt; used to check that injection does not merely bias
    the output towards a letter on unpressured questions).
    """
    texts, jobs = [], []
    for rec in source:
        it = items[rec["item_id"]]
        if rec["round"] == 0:
            text = render(lm.tok, solo_messages(it))
        else:
            text = render(lm.tok, replay_state(it, rec, reasons).messages())
        texts.append(text)
        jobs.append(rec)

    d_model = lm.model.get_output_embeddings().weight.shape[1]
    records = []
    for kind, alpha in itertools.product(kinds, alphas):
        dirs, dir_letters = [], []
        for rec in jobs:
            rng = item_rng(seed, rec["item_id"], kind, "direction")
            majority = rec.get("target") or rec["original"]
            d, letter = pick_direction(kind, lens, lm.letters, lm.letter_ids, layer, rec["original"], majority, rng,
                                       d_model, contrastive)
            dirs.append(d.cpu())
            dir_letters.append(letter)
        D = torch.stack(dirs)

        def ctx_fn(idx, D=D, alpha=alpha):
            return inject_last_position(lm.model, layer, D[idx], alpha)

        outs = batched_readout(lm, lens, texts, batch_size, ctx_fn=ctx_fn, desc=f"inject {kind} a={alpha}")
        for rec, letter, o in zip(jobs, dir_letters, outs):
            base = {k: v for k, v in rec.items() if k not in ("stated", "final_logits", "top_is_letter", "lens_logits", "lens_rank")}
            records.append({**base, "stated_before": rec["stated"], "inject_kind": kind, "inject_letter": letter,
                            "alpha": alpha, "inject_layer": layer, **o})
    return records
