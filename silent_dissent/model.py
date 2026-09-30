"""Model loading, per-layer residual capture and single-letter readout."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import torch

_LAYER_PATHS = ("model.layers", "model.language_model.layers", "transformer.h", "gpt_neox.layers", "model.decoder.layers")
_NORM_PATHS = (
    "model.norm",
    "model.language_model.norm",
    "transformer.ln_f",
    "gpt_neox.final_layer_norm",
    "model.decoder.final_layer_norm",
)


def _resolve(obj, paths):
    for path in paths:
        cur = obj
        try:
            for part in path.split("."):
                cur = getattr(cur, part)
        except AttributeError:
            continue
        return cur
    raise AttributeError(f"none of {paths} found on {type(obj).__name__}")


def get_layers(model) -> torch.nn.ModuleList:
    return _resolve(model, _LAYER_PATHS)


def get_final_norm(model) -> torch.nn.Module:
    return _resolve(model, _NORM_PATHS)


def layer_output(out):
    """Decoder blocks return a tensor (newer transformers) or a tuple."""
    return out[0] if isinstance(out, tuple) else out


def replace_layer_output(out, new_h):
    return (new_h,) + tuple(out[1:]) if isinstance(out, tuple) else new_h


def load_model(name: str, dtype: str = "bfloat16", device_map: str | None = "auto",
               chat_template_kwargs: dict | None = None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    tok.sd_template_kwargs = dict(chat_template_kwargs or {})  # read by prompts.render
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    torch_dtype = getattr(torch, dtype)
    try:
        model = AutoModelForCausalLM.from_pretrained(name, dtype=torch_dtype, device_map=device_map)
    except TypeError:  # transformers < 4.56
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch_dtype, device_map=device_map)
    model.eval()
    return model, tok


def letter_token_ids(tokenizer, letters: list[str], prefix: str) -> list[int]:
    """Token id of each answer letter as it follows `prefix` (usually " A" etc.).

    Asserts that each letter is exactly one extra token, which is what makes
    the answer position well defined.
    """
    base = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    ids = []
    for l in letters:
        full = tokenizer(f"{prefix} {l}", add_special_tokens=False)["input_ids"]
        if full[: len(base)] != base or len(full) != len(base) + 1:
            raise ValueError(f"letter {l!r} is not a single token after {prefix!r}: {full} vs {base}")
        ids.append(full[-1])
    if len(set(ids)) != len(ids):
        raise ValueError(f"letter tokens collide: {ids}")
    return ids


@dataclass
class Readout:
    """Letter readout for a batch at the answer position.

    final_letter_logits: [B, K]    model output logits restricted to the K letters
    final_top_is_letter: [B]       whether the unrestricted argmax is one of the letters
    lens_letter_logits:  [B, L, K] lens logits per layer, restricted to letters
    lens_vocab_rank:     [B, L, K] rank of each letter in the full-vocabulary lens distribution (0 = top)
    """

    final_letter_logits: torch.Tensor
    final_top_is_letter: torch.Tensor
    lens_letter_logits: torch.Tensor
    lens_vocab_rank: torch.Tensor


@contextmanager
def capture_last_position(model):
    """Collect the residual stream after every decoder block at the last position."""
    store: list[torch.Tensor | None] = [None] * len(get_layers(model))
    handles = []
    for i, layer in enumerate(get_layers(model)):

        def hook(_mod, _inp, out, i=i):
            store[i] = layer_output(out)[:, -1, :].detach()

        handles.append(layer.register_forward_hook(hook))
    try:
        yield store
    finally:
        for h in handles:
            h.remove()


@contextmanager
def capture_last_positions(model, n: int):
    """Like `capture_last_position`, but for the last `n` positions: store[i] is [B, n, d]."""
    store: list[torch.Tensor | None] = [None] * len(get_layers(model))
    handles = []
    for i, layer in enumerate(get_layers(model)):

        def hook(_mod, _inp, out, i=i):
            store[i] = layer_output(out)[:, -n:, :].detach()

        handles.append(layer.register_forward_hook(hook))
    try:
        yield store
    finally:
        for h in handles:
            h.remove()


@dataclass
class PositionReadout:
    """Letter readout of several lenses at the last P positions (position -1 is the answer position).

    final_letter_logits: [B, K]           model output logits at the answer position, restricted to letters
    final_top_is_letter: [B]
    lens_letter_logits:  {name: [B, P, L, K]}
    tail_ids:            [B, P]           token ids of the last P positions
    """

    final_letter_logits: torch.Tensor
    final_top_is_letter: torch.Tensor
    lens_letter_logits: dict[str, torch.Tensor]
    tail_ids: torch.Tensor


class LM:
    """Thin wrapper bundling model, tokenizer and the answer-letter token ids."""

    def __init__(self, model, tokenizer, letters: list[str], prefix: str):
        self.model = model
        self.tok = tokenizer
        self.letters = letters
        self.letter_ids = letter_token_ids(tokenizer, letters, prefix)

    @property
    def device(self):
        return next(self.model.parameters()).device

    @property
    def n_layers(self) -> int:
        return len(get_layers(self.model))

    def _encode(self, texts: list[str]):
        enc = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        # Explicit positions so left padding does not shift absolute-position models.
        enc["position_ids"] = (enc["attention_mask"].cumsum(-1) - 1).clamp(min=0)
        return enc

    @torch.no_grad()
    def readout(self, texts: list[str], lens) -> Readout:
        """One forward pass; the next token after each text is the answer letter.

        Any intervention hooks must be registered by the caller *before* this is
        called so that the captured residuals include the intervention.
        """
        enc = self._encode(texts)
        ids = torch.tensor(self.letter_ids, device=self.device)
        with capture_last_position(self.model) as resid:
            out = self.model(**enc)
        final = out.logits[:, -1, :].float()
        top_is_letter = torch.isin(final.argmax(-1), ids)

        lens_letters, lens_rank = [], []
        for layer, h in enumerate(resid):
            logits = lens.logits(h, layer).float()  # [B, V]
            letter_logits = logits[:, ids]  # [B, K]
            rank = (logits.unsqueeze(1) > letter_logits.unsqueeze(-1)).sum(-1)  # [B, K]
            lens_letters.append(letter_logits)
            lens_rank.append(rank)
        return Readout(
            final_letter_logits=final[:, ids].cpu(),
            final_top_is_letter=top_is_letter.cpu(),
            lens_letter_logits=torch.stack(lens_letters, 1).cpu(),
            lens_vocab_rank=torch.stack(lens_rank, 1).cpu(),
        )

    @torch.no_grad()
    def readout_positions(self, texts: list[str], lenses: dict, n_last: int) -> PositionReadout:
        """Letter logits of each lens at every layer of the last `n_last` positions.

        Position -1 matches `readout`; earlier positions are the tail of the prompt
        before the agent's answer (template tokens, end of the last user turn).
        """
        enc = self._encode(texts)
        ids = torch.tensor(self.letter_ids, device=self.device)
        with capture_last_positions(self.model, n_last) as resid:
            out = self.model(**enc)
        final = out.logits[:, -1, :].float()
        per_lens = {}
        for name, lens in lenses.items():
            rows = []
            for layer, h in enumerate(resid):
                b, p, d = h.shape
                rows.append(lens.logits(h.reshape(b * p, d), layer).float()[:, ids].reshape(b, p, -1))
            per_lens[name] = torch.stack(rows, 2).cpu()  # [B, P, L, K]
        return PositionReadout(
            final_letter_logits=final[:, ids].cpu(),
            final_top_is_letter=torch.isin(final.argmax(-1), ids).cpu(),
            lens_letter_logits=per_lens,
            tail_ids=enc["input_ids"][:, -n_last:].cpu(),
        )

    @torch.no_grad()
    def generate(self, texts: list[str], max_new_tokens: int = 64) -> list[str]:
        enc = self._encode(texts)
        enc.pop("position_ids")
        out = self.model.generate(
            **enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=self.tok.pad_token_id
        )
        new = out[:, enc["input_ids"].shape[1] :]
        return [t.strip() for t in self.tok.batch_decode(new, skip_special_tokens=True)]
