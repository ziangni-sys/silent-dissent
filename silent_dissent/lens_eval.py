"""Lens quality outside the letter task (exploratory, not part of the pre-registered study).

text_fidelity   per layer on plain text: KL(model output || lens), top-1 agreement with the
                model's prediction, and cross-entropy of the actual next token
two_hop_ranks   per layer on two-hop factual prompts ("the capital of the country where the
                Eiffel Tower is located is"): full-vocabulary rank of the unstated bridge entity
                ("France"), of the answer and of an unrelated control entity, at the subject's
                last token and at the last prompt token

Residuals are captured with the same hooks the rest of the pipeline uses (block outputs,
before the final norm), and every lens is applied exactly as in the letter readout.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .model import LM, capture_last_positions


def token_ranks(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """Rank (0 = top) of ids[n, m] within logits[n, :], i.e. the number of strictly larger logits."""
    vals = logits.gather(1, ids)  # [N, M]
    asc = logits.sort(-1).values
    return logits.shape[-1] - torch.searchsorted(asc, vals.contiguous(), right=True)


@torch.no_grad()
def text_fidelity(model, lenses: dict, windows: torch.Tensor, batch_size: int, skip_first: int = 16) -> dict:
    """windows: [N, T] token ids (no padding). Returns {lens: {metric: [n_layers]}}.

    Positions skip_first .. T-2 are scored (each predicts the next token); the first
    positions are skipped as in the J-lens fit (attention sink).
    """
    dev = next(model.parameters()).device
    sums: dict[str, dict[str, np.ndarray]] = {}
    n = 0
    for start in tqdm(range(0, len(windows), batch_size), desc="text", leave=False):
        ids = windows[start : start + batch_size].to(dev)
        b, t = ids.shape
        with capture_last_positions(model, t) as resid:
            out = model(input_ids=ids)
        final_lp = F.log_softmax(out.logits[:, skip_first:-1].float(), -1).reshape(-1, out.logits.shape[-1])
        final_p, final_top = final_lp.exp(), final_lp.argmax(-1)
        target = ids[:, skip_first + 1 :].reshape(-1)
        n += len(target)
        for name, lens in lenses.items():
            s = sums.setdefault(name, {k: np.zeros(len(resid)) for k in ("kl", "top1_agree", "top5_contains", "ce")})
            for layer, h in enumerate(resid):
                lp = F.log_softmax(lens.logits(h[:, skip_first:-1].reshape(-1, h.shape[-1]), layer).float(), -1)
                s["kl"][layer] += (final_p * (final_lp - lp)).sum().item()
                s["top1_agree"][layer] += (lp.argmax(-1) == final_top).sum().item()
                s["top5_contains"][layer] += (lp.topk(5, -1).indices == final_top[:, None]).any(-1).sum().item()
                s["ce"][layer] += -lp.gather(1, target[:, None]).sum().item()
                del lp
    return {name: {k: v / n for k, v in s.items()} for name, s in sums.items()}


@dataclass
class TwoHopItem:
    template: str
    prompt: str
    subject_offset: int  # position of the subject's last token, counted from the end (-1 = last token)
    bridge: str          # the unstated intermediate entity, e.g. "France"
    answer: str
    control: str         # another entity of the same kind, unrelated to this prompt
    ids: tuple[int, int, int]  # first tokens of " bridge", " answer", " control"


def first_token(tok, word: str) -> int:
    return tok(" " + word, add_special_tokens=False)["input_ids"][0]


def make_two_hop_item(tok, template: str, prefix: str, subject: str, suffix: str, bridge: str, answer: str,
                      control: str) -> TwoHopItem | None:
    """prompt = prefix + subject + suffix. None if the subject boundary is not a token boundary
    or the three probe tokens are not distinct."""
    full = tok(prefix + subject + suffix, add_special_tokens=False)["input_ids"]
    head = tok(prefix + subject, add_special_tokens=False)["input_ids"]
    if full[: len(head)] != head:
        return None
    ids = (first_token(tok, bridge), first_token(tok, answer), first_token(tok, control))
    if len(set(ids)) < 3:
        return None
    return TwoHopItem(template, prefix + subject + suffix, len(head) - len(full) - 1, bridge, answer, control, ids)


@torch.no_grad()
def probe_ranks(lm: LM, lenses: dict, prompts: list[str], subject_offsets: list[int], probe_ids: list[int],
                batch_size: int, n_last: int = 32, max_rank: int = 30000):
    """Ranks of a fixed set of probe tokens at every layer, at each prompt's subject token and last token.

    Returns (model's top-1 next token id [N], {lens: int16 ranks [N, 2, L, T]}); axis 1 is
    (subject, last), ranks are clipped at max_rank (only the top of the list matters).
    """
    assert all(-o <= n_last for o in subject_offsets), "raise n_last"
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    top1 = np.zeros(len(prompts), dtype=np.int64)
    ranks: dict[str, np.ndarray] = {}
    probe = torch.tensor(probe_ids, device=lm.device)
    for start in tqdm(range(0, len(order), batch_size), desc="probe", leave=False):
        idx = order[start : start + batch_size]
        enc = lm._encode([prompts[i] for i in idx])
        with capture_last_positions(lm.model, n_last) as resid:
            out = lm.model(**enc)
        top1[idx] = out.logits[:, -1].argmax(-1).cpu().numpy()
        rows = torch.arange(len(idx), device=lm.device)
        pos = torch.tensor([n_last + subject_offsets[i] for i in idx], device=lm.device)
        ids = probe.expand(len(idx), -1)
        for name, lens in lenses.items():
            r = ranks.setdefault(name, np.zeros((len(prompts), 2, len(resid), len(probe_ids)), dtype=np.int16))
            for layer, h in enumerate(resid):
                for k, hh in enumerate((h[rows, pos], h[:, -1])):
                    rk = token_ranks(lens.logits(hh, layer).float(), ids).clamp(max=max_rank)
                    r[idx, k, layer] = rk.cpu().numpy().astype(np.int16)
    return top1, ranks


@torch.no_grad()
def two_hop_ranks(lm: LM, lenses: dict, items: list[TwoHopItem], batch_size: int, n_last: int = 16):
    """Returns (model's top-1 next token id [N], {lens: ranks [N, 2, L, 3]}), where axis 1 is
    (subject's last token, last prompt token) and axis 3 is (bridge, answer, control)."""
    assert all(-it.subject_offset <= n_last for it in items), "raise n_last"
    top1 = np.zeros(len(items), dtype=np.int64)
    ranks: dict[str, np.ndarray] = {}
    for start in tqdm(range(0, len(items), batch_size), desc="two-hop", leave=False):
        batch = items[start : start + batch_size]
        enc = lm._encode([it.prompt for it in batch])
        with capture_last_positions(lm.model, n_last) as resid:
            out = lm.model(**enc)
        top1[start : start + len(batch)] = out.logits[:, -1].argmax(-1).cpu().numpy()
        rows = torch.arange(len(batch), device=lm.device)
        pos = torch.tensor([n_last + it.subject_offset for it in batch], device=lm.device)
        probe = torch.tensor([it.ids for it in batch], device=lm.device)
        for name, lens in lenses.items():
            r = ranks.setdefault(name, np.zeros((len(items), 2, len(resid), 3), dtype=np.int64))
            for layer, h in enumerate(resid):
                for k, hh in enumerate((h[rows, pos], h[:, -1])):
                    r[start : start + len(batch), k, layer] = token_ranks(lens.logits(hh, layer).float(), probe).cpu().numpy()
    return top1, ranks
