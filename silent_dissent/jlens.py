"""Fitting the Jacobian lens (Gurnee, Sofroniew et al. 2026, "Verbalizable
Representations Form a Global Workspace in Language Models").

    lens_l(h) = unembed( J_l @ h ),   J_l = E[ d h_final / d h_l ]

Port of the reference estimator in github.com/anthropics/jacobian-lens
(jlens/fitting.py, Apache-2.0) onto this package's model helpers. For each
output dimension a one-hot cotangent is placed at *every* valid target position
at once, so the gradient at source position p is sum_{p' >= p} dh_final[p'] /
dh_l[p]; J_l is the mean of that over valid source positions and prompts.

Files are written in the reference format ({"J": {layer: [d, d]}, "n_prompts",
"source_layers", "d_model"}), so lenses fitted here load with
`jlens.JacobianLens.load` and pre-fitted reference lenses load with
`silent_dissent.lenses.JLens`.

Cost per prompt: one forward on the prompt replicated `dim_batch` times plus
ceil(d_model / dim_batch) backward passes. Shard with `fit(..., prompts[i::n])`
and combine with `merge`.
"""
from __future__ import annotations

import logging
import math
import os
import time
from contextlib import contextmanager

import torch

from .model import get_layers, layer_output

log = logging.getLogger(__name__)

#: Leading positions act as attention sinks and are excluded (reference default).
SKIP_FIRST = 16


def valid_positions(seq_len: int, skip_first: int = SKIP_FIRST) -> torch.Tensor:
    """Positions skip_first .. seq_len-2 (the last position has no next-token target)."""
    if seq_len - 1 <= skip_first:
        raise ValueError(f"prompt too short: {seq_len} tokens, need > {skip_first + 1}")
    return torch.arange(skip_first, seq_len - 1)


@contextmanager
def _record(model, at: list[int], root: int):
    """Keep block outputs (with graph) at `at`; make block `root`'s output the graph leaf."""
    acts: dict[int, torch.Tensor] = {}
    handles = []
    for i, layer in enumerate(get_layers(model)):
        if i not in at:
            continue

        def hook(_mod, _inp, out, i=i):
            h = layer_output(out)
            if i == root:
                h.requires_grad_(True)
            acts[i] = h

        handles.append(layer.register_forward_hook(hook))
    try:
        yield acts
    finally:
        for h in handles:
            h.remove()


def _forward(model, input_ids):
    try:
        return model(input_ids=input_ids, use_cache=False, logits_to_keep=1)
    except TypeError:
        return model(input_ids=input_ids, use_cache=False)


def jacobian_for_prompt(
    model, input_ids: torch.Tensor, source_layers: list[int], target_layer: int,
    dim_batch: int = 8, skip_first: int = SKIP_FIRST,
) -> dict[int, torch.Tensor]:
    """Per-prompt estimate of J_l for every source layer. input_ids: [1, S]. Returns fp32 CPU [d, d]."""
    pos = valid_positions(input_ids.shape[1], skip_first)
    with _record(model, [*source_layers, target_layer], min(source_layers)) as acts, torch.enable_grad():
        _forward(model, input_ids.expand(dim_batch, -1))
        target = acts[target_layer]  # [dim_batch, S, d]
        sources = [acts[l] for l in source_layers]
        d = target.shape[-1]
        J = {l: torch.zeros(d, d) for l in source_layers}
        b = torch.arange(dim_batch, device=target.device)
        p = pos.to(target.device)
        cot = torch.zeros_like(target)
        n_pass = math.ceil(d / dim_batch)
        for k, start in enumerate(range(0, d, dim_batch)):
            n = min(dim_batch, d - start)
            cot.zero_()
            cot[b[:n, None], p[None, :], start + b[:n, None]] = 1.0
            grads = torch.autograd.grad(target, sources, grad_outputs=cot, retain_graph=k < n_pass - 1)
            for l, g in zip(source_layers, grads):
                J[l][start : start + n] = g[:n, p.to(g.device)].float().mean(1).cpu()
            del grads
    return J


def _save(obj, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_lens(J: dict[int, torch.Tensor], n_prompts: int, path: str, dtype=torch.float16) -> None:
    layers = sorted(J)
    _save({"J": {l: J[l].to(dtype) for l in layers}, "n_prompts": n_prompts,
           "source_layers": layers, "d_model": J[layers[0]].shape[0]}, path)


def load_lens(path: str) -> tuple[dict[int, torch.Tensor], int]:
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if "J" not in ck:
        raise ValueError(f"{path} is not a Jacobian-lens file (keys {sorted(ck)}); a fit checkpoint?")
    return {int(l): J.float() for l, J in ck["J"].items()}, int(ck["n_prompts"])


def merge(paths: list[str], out: str) -> None:
    """n_prompts-weighted mean of lenses fitted on disjoint prompt shards."""
    lenses = [load_lens(p) for p in paths]
    total = sum(n for _, n in lenses)
    layers = sorted(lenses[0][0])
    if any(sorted(J) != layers for J, _ in lenses):
        raise ValueError("shards disagree on source layers")
    save_lens({l: sum(J[l] * n for J, n in lenses) / total for l in layers}, total, out)


def fit(
    model, tokenizer, prompts: list[str], out: str, *, source_layers: list[int] | None = None,
    target_layer: int | None = None, dim_batch: int = 8, max_seq_len: int = 128,
    skip_first: int = SKIP_FIRST, checkpoint_every: int = 10,
) -> str:
    """Fit J_l over `prompts` and write the lens to `out`. Resumes from `out + '.ckpt'`."""
    n_layers = len(get_layers(model))
    target = n_layers - 1 if target_layer is None else target_layer % n_layers
    sources = list(range(target)) if source_layers is None else sorted(l % n_layers for l in source_layers)
    if not sources or sources[-1] >= target:
        raise ValueError(f"source layers {sources} must lie below target layer {target}")
    for prm in model.parameters():
        prm.requires_grad_(False)
    model.eval()
    device = next(model.parameters()).device

    ckpt = out + ".ckpt"
    meta = {"source_layers": sources, "target_layer": target, "skip_first": skip_first, "max_seq_len": max_seq_len}
    if os.path.exists(ckpt):
        st = torch.load(ckpt, map_location="cpu", weights_only=True)
        if {k: st[k] for k in meta} != meta:
            raise ValueError(f"{ckpt} was fitted with different settings; delete it to start over")
        J_sum, n_done, next_idx = st["J_sum"], st["n_done"], st["next_idx"]
        log.info("resuming at prompt %d (%d fitted)", next_idx, n_done)
    else:
        d = model.config.get_text_config().hidden_size
        J_sum, n_done, next_idx = {l: torch.zeros(d, d) for l in sources}, 0, 0

    def checkpoint():
        _save({"J_sum": J_sum, "n_done": n_done, "next_idx": next_idx, **meta}, ckpt)

    add_bos = getattr(tokenizer, "bos_token_id", None) is not None
    for i in range(next_idx, len(prompts)):
        t0 = time.perf_counter()
        ids = tokenizer(prompts[i], return_tensors="pt", truncation=True, max_length=max_seq_len - add_bos,
                        add_special_tokens=False).input_ids
        if add_bos and (ids.shape[1] == 0 or ids[0, 0] != tokenizer.bos_token_id):
            ids = torch.cat([torch.tensor([[tokenizer.bos_token_id]]), ids], 1)
        try:
            J = jacobian_for_prompt(model, ids.to(device), sources, target, dim_batch, skip_first)
        except ValueError as e:
            log.warning("skipping prompt %d: %s", i, e)
        else:
            for l in sources:
                J_sum[l] += J[l]
            n_done += 1
            log.info("prompt %d/%d  %d tokens  %.1fs", i + 1, len(prompts), ids.shape[1], time.perf_counter() - t0)
        next_idx = i + 1
        if checkpoint_every and next_idx % checkpoint_every == 0:
            checkpoint()
    if n_done == 0:
        raise ValueError("no prompt was long enough to fit on")
    save_lens({l: J_sum[l] / n_done for l in sources}, n_done, out)
    if os.path.exists(ckpt):
        os.remove(ckpt)
    return out


def load_corpus(name: str, n: int, seed: int = 0, min_chars: int = 600) -> list[str]:
    """Generic text for fitting. `wikitext` (the reference lenses' corpus), or a local .txt (one doc per line) / .jsonl ({"text"})."""
    import json
    import random

    if name == "wikitext":
        from datasets import load_dataset

        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
        docs = [t for t in ds["text"] if len(t) >= min_chars and not t.lstrip().startswith("=")]
    elif name.endswith(".jsonl"):
        with open(name) as f:
            docs = [json.loads(line)["text"] for line in f if line.strip()]
    else:
        with open(name) as f:
            docs = [line.rstrip("\n") for line in f if line.strip()]
    random.Random(seed).shuffle(docs)
    return docs[:n]
