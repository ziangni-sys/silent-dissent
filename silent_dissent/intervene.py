"""Activation injection for the causal test.

At decoder block `layer`, the residual at the answer position is shifted by

    h <- h + alpha * ||h|| * d_hat

where d_hat is a unit direction per batch row, so alpha is a fraction of the
residual norm and is comparable across layers and models.
"""
from __future__ import annotations

import random
from contextlib import contextmanager

import torch

from .model import get_layers, layer_output, replace_layer_output

DIRECTION_KINDS = ("original", "majority", "other", "random")


def letter_direction(lens, letter_ids: list[int], k: int, layer: int, contrastive: bool = True) -> torch.Tensor:
    """Direction for letter index k; contrastive = minus the mean of the other letters."""
    dirs = torch.stack([lens.direction(t, layer) for t in letter_ids])  # [K, d]
    d = dirs[k]
    if contrastive:
        others = torch.cat([dirs[:k], dirs[k + 1 :]])
        d = d - others.mean(0)
    return d / d.norm()


def pick_direction(
    kind: str,
    lens,
    letters: list[str],
    letter_ids: list[int],
    layer: int,
    original: str,
    majority: str,
    rng: random.Random,
    d_model: int,
    contrastive: bool = True,
) -> tuple[torch.Tensor, str | None]:
    """Return (unit direction, letter it encodes or None for random)."""
    if kind == "original":
        letter = original
    elif kind == "majority":
        letter = majority
    elif kind == "other":
        letter = rng.choice([l for l in letters if l not in (original, majority)])
    elif kind == "random":
        g = torch.Generator().manual_seed(rng.randrange(2**31))
        d = torch.randn(d_model, generator=g)
        return d / d.norm(), None
    else:
        raise ValueError(f"unknown direction kind {kind!r}")
    return letter_direction(lens, letter_ids, letters.index(letter), layer, contrastive), letter


@contextmanager
def inject_last_position(model, layer: int, directions: torch.Tensor, alpha: float):
    """Register a hook adding alpha*||h||*d to the last position of `layer`.

    directions: [B, d] unit vectors (one per batch row). Register this before
    any capture hooks so the captured residuals include the injection.
    """
    block = get_layers(model)[layer]

    def hook(_mod, _inp, out):
        h = layer_output(out)
        last = h[:, -1, :]
        d = directions.to(device=h.device, dtype=torch.float32)
        delta = alpha * last.float().norm(dim=-1, keepdim=True) * d
        h = h.clone()
        h[:, -1, :] = (last.float() + delta).to(h.dtype)
        return replace_layer_output(out, h)

    handle = block.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()
