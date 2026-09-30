"""Lenses: decode an intermediate residual into vocabulary logits.

Every lens implements two methods used by the rest of the pipeline:

    logits(h, layer)          h: [B, d] residual after decoder block `layer`
                              -> [B, V] vocabulary logits
    direction(token, layer)   -> [d] residual-space direction that increases the
                              lens logit of `token` at `layer` (used for
                              injection in the causal test)

`LogitLens` is fully implemented and serves as the baseline. `AffineLens`
covers any lens that is a learned/derived per-layer linear map into the final
residual space (tuned lens and similar). `JLens` is the Jacobian lens, the
special case where that map is the average Jacobian J_l (silent_dissent/jlens.py).
"""
from __future__ import annotations

import torch

from .model import get_final_norm


class Lens:
    name = "base"

    def logits(self, h: torch.Tensor, layer: int) -> torch.Tensor:
        raise NotImplementedError

    def direction(self, token_id: int, layer: int) -> torch.Tensor:
        raise NotImplementedError


def _norm_gain(norm: torch.nn.Module) -> torch.Tensor | None:
    """Elementwise gain of the final norm.

    For RMS-style norms this is read off empirically as norm(ones), which covers
    both w and (1 + w) parameterisations (Gemma, Qwen3-Next / Qwen3.5) without
    per-architecture special cases. LayerNorm (mean-subtracting) uses its weight.
    """
    w = getattr(norm, "weight", None)
    if w is None:
        return None
    if isinstance(norm, torch.nn.LayerNorm):
        return w
    with torch.no_grad():
        return norm(torch.ones(1, w.shape[-1], dtype=w.dtype, device=w.device))[0].float()


def _final_softcap(model) -> float | None:
    """Final-logit soft cap (Gemma 2 / 4: logits = c * tanh(logits / c)), or None."""
    cfg = getattr(model, "config", None)
    text = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
    return getattr(text, "final_logit_softcapping", None)


class LogitLens(Lens):
    """unembed(final_norm(h)), soft-capped like the model's own logits (a monotone map, so ranks
    are those of the uncapped logits). At the last layer this reproduces the model logits."""

    name = "logit"

    def __init__(self, model):
        self.norm = get_final_norm(model)
        self.unembed = model.get_output_embeddings()
        self.softcap = _final_softcap(model)

    def _decode(self, h: torch.Tensor) -> torch.Tensor:
        logits = self.unembed(self.norm(h))
        if self.softcap:  # in float32: a bf16 tanh would merge near-saturated logits into ties
            logits = self.softcap * torch.tanh(logits.float() / self.softcap)
        return logits

    def logits(self, h, layer):
        return self._decode(h)

    def direction(self, token_id, layer):
        w = self.unembed.weight[token_id].float()
        gain = _norm_gain(self.norm)
        return w * gain.float() if gain is not None else w


class AffineLens(LogitLens):
    """unembed(final_norm(A_l h + b_l)) with per-layer translators loaded from disk.

    File format (torch.save): {layer_index: {"weight": [d, d], "bias": [d]}}.
    Layers missing from the file fall back to the identity (plain logit lens).
    """

    name = "affine"

    def __init__(self, model, path: str):
        super().__init__(model)
        dev = self.unembed.weight.device
        raw = torch.load(path, map_location=dev)
        self.maps = {int(k): (v["weight"].to(dev), v.get("bias")) for k, v in raw.items()}

    def _translate(self, h, layer):
        if layer not in self.maps:
            return h
        W, b = self.maps[layer]
        out = h.to(W.dtype) @ W.T
        if b is not None:
            out = out + b.to(out.dtype)
        return out.to(h.dtype)

    def logits(self, h, layer):
        return self._decode(self._translate(h, layer))

    def direction(self, token_id, layer):
        d = super().direction(token_id, layer)
        if layer not in self.maps:
            return d
        return self.maps[layer][0].float().T @ d  # pull back through A_l


class JLens(AffineLens):
    """Jacobian lens: unembed(final_norm(J_l h)), J_l = E[d h_final / d h_l].

    Either `path` (a local lens file in the reference format, e.g. from
    scripts/fit_jlens.py) or `repo` + `filename` (+ `revision`) for a
    pre-fitted reference lens on the HuggingFace Hub.
    Layers without a J_l (the final layer, which is the fitting target) use the
    identity, so the last layer reproduces the model output exactly.
    `direction` is J_l^T times the logit-lens token direction.
    """

    name = "jlens"

    def __init__(self, model, path: str | None = None, repo: str | None = None, filename: str | None = None,
                 revision: str | None = None):
        from .jlens import load_lens

        if path is None:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(repo, filename, revision=revision)
        LogitLens.__init__(self, model)
        dev = self.unembed.weight.device
        J, self.n_prompts = load_lens(path)
        d = self.unembed.weight.shape[1]
        if next(iter(J.values())).shape != (d, d):
            raise ValueError(f"{path}: J is {tuple(next(iter(J.values())).shape)}, model d_model is {d}")
        self.maps = {l: (M.to(dev), None) for l, M in J.items()}


LENSES = {"logit": LogitLens, "affine": AffineLens, "jlens": JLens}


def build_lens(model, name: str = "logit", **kwargs) -> Lens:
    return LENSES[name](model, **kwargs)
