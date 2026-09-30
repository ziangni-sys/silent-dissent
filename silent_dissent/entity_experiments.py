"""Experiment loops for the entity-answer study: generation, entity readout, injection, debate.

Readouts store, per record, layer and position, the rank (0 = top, full vocabulary) of each
role's entity (entity_facts.ROLES) under each lens: the minimum over the entity's probe
tokens, -1 where a role has no usable token for that item. Positions (entity_metrics.POS):
subject (last token of e1), last (the answer position); extended readouts add mention (last
token of the bridge's description, e.g. "The author of the novel X" or "The city where X was
born") and span (the minimum rank over every position from the mention to the last token).
"""
from __future__ import annotations

import random

import numpy as np
import torch
from tqdm import tqdm

from .entity_facts import ROLES, Fact, matches, norm, probe_tokens
from .entity_prompts import AGAIN, CHAT_SYSTEM, PLAIN_HEADER, PLAIN_SHOTS, EntityState, bos_text, mu, statement
from .intervene import inject_last_position
from .lens_eval import token_ranks
from .model import LM, capture_last_positions

MAX_RANK = 30000

# ------------------------------------------------------------------ positions


def subject_offset(tok, prompt: str, subject: str, ignore_case: bool = False) -> int | None:
    """Offset from the end (-1 = last token) of the token holding the last character of the
    last occurrence of `subject`; None if it does not occur."""
    end = prompt.lower().rfind(subject.lower()) if ignore_case else prompt.rfind(subject)
    if end < 0:
        return None
    end += len(subject) - 1
    enc = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
    for i, (a, b) in enumerate(enc["offset_mapping"]):
        if a <= end < b:
            return i - len(enc["input_ids"])
    return None


def token_labels(tok, prompt: str, spans: list[tuple[str, int, int]], start: int) -> tuple[int, list[str]]:
    """Number of tokens from the first token overlapping char `start` to the end, and a label per
    such token (the span it falls in, "" between spans)."""
    enc = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
    offs = enc["offset_mapping"]
    first = next(i for i, (a, b) in enumerate(offs) if b > start)
    labels = []
    for a, b in offs[first:]:
        labels.append(next((lab for lab, s, e in spans if a < e and b > s), ""))
    return len(offs) - first, labels


# ------------------------------------------------------------------ generation


def clean_answer(text: str, max_chars: int = 80) -> str:
    return text.split("\n")[0].strip().rstrip(".").strip()[:max_chars]


@torch.no_grad()
def generate(lm: LM, prompts: list[str], max_new_tokens: int, batch_size: int, temperature: float = 0.0,
             seed: int = 0, desc: str = "generate") -> list[str]:
    """Continuation of each prompt (first line only); greedy unless temperature > 0."""
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    out: list[str] = [""] * len(prompts)
    for start in tqdm(range(0, len(order), batch_size), desc=desc, leave=False):
        idx = order[start : start + batch_size]
        enc = lm._encode([prompts[i] for i in idx])
        enc.pop("position_ids")
        kw = dict(max_new_tokens=max_new_tokens, pad_token_id=lm.tok.pad_token_id)
        if temperature > 0:
            torch.manual_seed(seed + start)
            kw.update(do_sample=True, temperature=temperature, top_p=1.0, top_k=0)
        else:
            kw.update(do_sample=False)
        gen = lm.model.generate(**enc, **kw)[:, enc["input_ids"].shape[1] :]
        for i, text in zip(idx, lm.tok.batch_decode(gen, skip_special_tokens=True)):
            out[i] = clean_answer(text)
    return out


def classify(answer: str, fact: Fact, peer: Fact, sibling: Fact | None = None) -> str:
    if matches(answer, fact.e3_aliases):
        return "original"
    if matches(answer, peer.e3_aliases):
        return "peer"
    if sibling is not None and matches(answer, sibling.e3_aliases):
        return "sibling"
    return "other"


KNOWN_SHOTS = "The capital city of Japan is Tokyo.\nThe author of the play Hamlet is William Shakespeare.\n"


def known_check(lm: LM, facts: list[Fact], batch_size: int, max_new_tokens: int = 12) -> list[dict]:
    """Does the model complete hop 1 (-> e2), hop 2 (-> e3) and the composition (-> e3)?"""
    prompts = []
    for f in facts:
        head = bos_text(lm.tok) + KNOWN_SHOTS
        prompts += [head + f.r1_prompt, head + f.r2_prompt, head + f.composed]
    texts = generate(lm, prompts, max_new_tokens, batch_size, desc="known")
    out = []
    for i, f in enumerate(facts):
        h1, h2, c = texts[3 * i : 3 * i + 3]
        out.append({"uid": f.uid, "hop1": matches(h1, f.e2_aliases), "hop2": matches(h2, f.e3_aliases),
                    "composed": matches(c, f.e3_aliases), "texts": [h1, h2, c]})
    return out


# ------------------------------------------------------------------ entity readout


def _role_mask(roles_batch: list[dict[str, list[int]]], union: list[int], device) -> torch.Tensor:
    pos = {t: i for i, t in enumerate(union)}
    m = torch.zeros(len(roles_batch), len(ROLES), len(union), dtype=torch.bool)
    for b, roles in enumerate(roles_batch):
        for r, role in enumerate(ROLES):
            for t in roles.get(role, []):
                m[b, r, pos[t]] = True
    return m.to(device)


def _role_ranks(logits: torch.Tensor, union_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """logits [N, V], union_ids [U], mask [N, R, U] -> min rank per role [N, R] (-1 if no token)."""
    rk = token_ranks(logits, union_ids.expand(len(logits), -1)).clamp(max=MAX_RANK)  # [N, U]
    big = torch.full_like(rk, MAX_RANK + 1)
    out = torch.where(mask, rk[:, None, :], big[:, None, :]).min(-1).values
    return torch.where(mask.any(-1), out, torch.full_like(out, -1))


@torch.no_grad()
def entity_readout(lm: LM, lenses: dict, prompts: list[str], subject_offsets: list[int],
                   roles: list[dict[str, list[int]]], batch_size: int,
                   mention_offsets: list[int] | None = None) -> dict[str, np.ndarray]:
    """{lens: int16 [N, P, L, R]}: role ranks at (subject's last token, last token), P = 2; with
    `mention_offsets` also at the mention's last token and the minimum over the positions from
    the mention to the last token, P = 4."""
    n_pos = 2 if mention_offsets is None else 4
    if not prompts:
        return {name: np.zeros((0, n_pos, lm.n_layers, len(ROLES)), dtype=np.int16) for name in lenses}
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    n_last = max(-o for o in list(subject_offsets) + list(mention_offsets or [])) + 1
    out: dict[str, np.ndarray] = {}
    for start in tqdm(range(0, len(order), batch_size), desc="readout", leave=False):
        idx = order[start : start + batch_size]
        enc = lm._encode([prompts[i] for i in idx])
        with capture_last_positions(lm.model, n_last) as resid:
            lm.model(**enc)
        union = sorted({t for i in idx for ids in roles[i].values() for t in ids}) or [0]
        uid = torch.tensor(union, device=lm.device)
        mask = _role_mask([roles[i] for i in idx], union, lm.device)
        rows = torch.arange(len(idx), device=lm.device)
        pos = torch.tensor([n_last + subject_offsets[i] for i in idx], device=lm.device)
        if mention_offsets is not None:
            mpos = torch.tensor([n_last + mention_offsets[i] for i in idx], device=lm.device)
            w = max(-mention_offsets[i] for i in idx)  # window: the earliest mention in the batch to the end
            before = torch.arange(w, device=lm.device)[None, :] < (mpos - (n_last - w))[:, None]  # [B, w]
            span_mask = mask[:, None].expand(-1, w, -1, -1).reshape(-1, *mask.shape[1:])
        for name, lens in lenses.items():
            arr = out.setdefault(name, np.zeros((len(prompts), n_pos, len(resid), len(ROLES)), dtype=np.int16))
            for layer, h in enumerate(resid):
                slots = [h[rows, pos], h[:, -1]] + ([h[rows, mpos]] if mention_offsets is not None else [])
                for k, hh in enumerate(slots):
                    arr[idx, k, layer] = _role_ranks(lens.logits(hh, layer).float(), uid, mask).cpu().numpy()
                if mention_offsets is not None:  # minimum over the mention .. last token
                    hw = h[:, -w:].reshape(-1, h.shape[-1])
                    rk = _role_ranks(lens.logits(hw, layer).float(), uid, span_mask).view(len(idx), w, -1)
                    rk = torch.where(before[..., None] & (rk >= 0), torch.full_like(rk, MAX_RANK + 1), rk)
                    arr[idx, 3, layer] = rk.min(1).values.cpu().numpy()
    return out


@torch.no_grad()
def timeline_readout(lm: LM, lenses: dict, prompts: list[str], window: list[int], roles: list[dict[str, list[int]]],
                     layers: list[int], batch_size: int, chunk: int = 256) -> dict[str, list[np.ndarray]]:
    """{lens: [int16 [T_i, len(layers), R] per prompt]} over each prompt's last window[i] tokens."""
    out: dict[str, list] = {name: [None] * len(prompts) for name in lenses}
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    for start in tqdm(range(0, len(order), batch_size), desc="timeline", leave=False):
        idx = order[start : start + batch_size]
        n = max(window[i] for i in idx)
        enc = lm._encode([prompts[i] for i in idx])
        with capture_last_positions(lm.model, n) as resid:
            lm.model(**enc)
        union = sorted({t for i in idx for ids in roles[i].values() for t in ids}) or [0]
        uid = torch.tensor(union, device=lm.device)
        mask = _role_mask([roles[i] for i in idx], union, lm.device)  # [B, R, U]
        for name, lens in lenses.items():
            res = torch.zeros(len(idx), n, len(layers), len(ROLES), dtype=torch.int16)
            for j, layer in enumerate(layers):
                h = resid[layer]  # [B, n, d]
                flat = h.reshape(-1, h.shape[-1])
                m = mask[:, None].expand(-1, n, -1, -1).reshape(-1, len(ROLES), len(union))
                for c in range(0, len(flat), chunk):
                    rr = _role_ranks(lens.logits(flat[c : c + chunk], layer).float(), uid, m[c : c + chunk])
                    res.view(-1, len(layers), len(ROLES))[c : c + chunk, j] = rr.cpu().to(torch.int16)
            for b, i in enumerate(idx):
                out[name][i] = res[b, n - window[i] :].numpy()
    return out


# ------------------------------------------------------------------ conversation loop


def mention_offset(tok, prompt: str, final: str, fact: Fact) -> int | None:
    """Offset of the last token of the bridge's description (mu, e.g. "The author of the novel X")
    inside the final statement `final` (which ends the prompt); None if it does not contain it."""
    at = final.lower().rfind(mu(fact).lower())
    if at < 0 or not prompt.endswith(final):
        return None
    end = len(prompt) - len(final) + at + len(mu(fact)) - 1
    enc = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
    for i, (a, b) in enumerate(enc["offset_mapping"]):
        if a <= end < b:
            return i - len(enc["input_ids"])
    return None


def readout_inputs(tok, states: list[EntityState], fmt: str, extended: bool = False):
    """Prompts and subject offsets; with `extended`, also the mention offsets (entity_readout)."""
    prompts = [st.render(fmt, tok) for st in states]
    offs, moffs = [], []
    for p, st in zip(prompts, states):
        o = subject_offset(tok, p, st.fact.e1)
        offs.append(o if o is not None else -1)
        if extended:
            m = mention_offset(tok, p, st.final_prefix(), st.fact)
            moffs.append(m if m is not None else -1)
    return (prompts, offs, moffs) if extended else (prompts, offs)


def run_rounds(lm: LM, lenses: dict, states: list[EntityState], meta: list[dict], roles: list[dict[str, list[int]]],
               fmt: str, rounds: int, batch_size: int, gen_tokens: int, gen_batch_size: int | None = None,
               extended: bool = False) -> tuple[list[dict], dict[str, np.ndarray]]:
    """Run `rounds` more rounds on states whose round-0 answer is recorded. Each round: readout on
    the prompt, then the agent's answer (greedy) is generated, classified and recorded.
    `extended`: readouts with the mention and span positions too (entity_readout)."""
    records, parts = [], {name: [] for name in lenses}
    for r in range(1, rounds + 1):
        live = [i for i, st in enumerate(states) if not (st.condition in ("instructed", "hypothetical", "retention")
                                                         and st.rounds >= 1)]
        if not live:
            break
        for i in live:
            states[i].advance()
        sub = [states[i] for i in live]
        prompts, offs, *moffs = readout_inputs(lm.tok, sub, fmt, extended)
        arr = entity_readout(lm, lenses, prompts, offs, [roles[i] for i in live], batch_size,
                             moffs[0] if extended else None)
        answers = generate(lm, prompts, gen_tokens, gen_batch_size or batch_size, desc=f"round {r}")
        for i, a in zip(live, answers):
            st = states[i]
            st.record(a)
            records.append({**meta[i], "round": r, "answer": a,
                            "outcome": classify(a, st.fact, st.peer, st.sibling if st.condition == "retention" else None)})
        for name in lenses:
            parts[name].append(arr[name])
    return records, {name: np.concatenate(p) if p else np.zeros((0,)) for name, p in parts.items()}


# ------------------------------------------------------------------ injection (E4)


def entity_direction(lens, token_ids: list[int], layer: int) -> torch.Tensor:
    d = torch.stack([lens.direction(t, layer).float() for t in token_ids]).mean(0)
    return d / d.norm()


@torch.no_grad()
def run_injection(lm: LM, lens, states: list[EntityState], meta: list[dict], roles: list[dict[str, list[int]]],
                  fmt: str, layers: list[int], alphas: list[float], kinds: list[str], batch_size: int,
                  seed: int) -> list[dict]:
    """Re-run each state's current prompt with a direction added at the last position of `layer`.
    kinds: roles (e.g. orig_bridge) or "random". Outcome = the answer's first token."""
    prompts = [st.render(fmt, lm.tok) for st in states]
    first = lambda s: lm.tok(" " + s.strip(), add_special_tokens=False)["input_ids"][0]
    targets = [(first(st.fact.e3), first(st.peer.e3), first(st.control.e3)) for st in states]
    d_model = lm.model.get_output_embeddings().weight.shape[1]
    # answers whose first tokens collide cannot be told apart by the first token
    order = sorted((i for i in range(len(prompts)) if len(set(targets[i])) == 3), key=lambda i: len(prompts[i]))
    records = []
    for layer in layers:
        for kind in kinds:
            dirs = []
            for i, st in enumerate(states):
                if kind == "random":
                    g = torch.Generator().manual_seed(random.Random(f"{seed}-{meta[i]['uid']}-{layer}").randrange(2**31))
                    d = torch.randn(d_model, generator=g)
                    dirs.append(d / d.norm())
                else:
                    ids = roles[i].get(kind, [])
                    dirs.append(entity_direction(lens, ids, layer).cpu() if ids else torch.full((d_model,), float("nan")))
            D = torch.stack(dirs)
            for alpha in alphas:
                for start in range(0, len(order), batch_size):
                    idx = [i for i in order[start : start + batch_size] if not torch.isnan(D[i]).any()]
                    if not idx:
                        continue
                    enc = lm._encode([prompts[i] for i in idx])
                    with inject_last_position(lm.model, layer, D[idx], alpha):
                        logits = lm.model(**enc).logits[:, -1].float()
                    t = torch.tensor([targets[i] for i in idx], device=logits.device)
                    rk = token_ranks(logits, t).cpu().numpy()
                    top = logits.argmax(-1).cpu().numpy()
                    for j, i in enumerate(idx):
                        o, p, c = targets[i]
                        records.append({**meta[i], "inject_kind": kind, "inject_layer": layer, "alpha": alpha,
                                        "top1_is": "original" if top[j] == o else "peer" if top[j] == p else
                                        "control" if top[j] == c else "other",
                                        "rank_original": int(rk[j, 0]), "rank_peer": int(rk[j, 1]),
                                        "rank_control": int(rk[j, 2])})
    return records


# ------------------------------------------------------------------ real debate (E5)


def debate_prompt(fmt: str, tok, fact: Fact, own: list[str], others: list[list[str]]) -> str:
    """Agent's view: its own statements own[0..r-1] and the other agents' others[0..r-1]."""
    question = f"Complete the statement: {fact.composed} ..."
    if fmt == "plain":
        text = bos_text(tok) + PLAIN_HEADER + PLAIN_SHOTS + f"Question: {question}\n"
        for r in range(len(own)):
            text += f"Agent 1: {statement(fact.composed, own[r])}\n"
            text += "".join(f"Agent {j + 2}: {statement(fact.composed, a)}\n" for j, a in enumerate(others[r]))
            text += f"Moderator: {AGAIN}\n"
        return text + f"Agent 1: {fact.composed}"
    from .prompts import render

    msgs = [{"role": "system", "content": CHAT_SYSTEM}, {"role": "user", "content": question}]
    for r in range(len(own)):
        msgs.append({"role": "assistant", "content": statement(fact.composed, own[r])})
        body = "\n".join(f"Agent {j + 2}: {statement(fact.composed, a)}" for j, a in enumerate(others[r]))
        msgs.append({"role": "user", "content": f"{body}\nModerator: {AGAIN}"})
    return render(tok, msgs, prefix=fact.composed)


@torch.no_grad()
def run_debate(lm: LM, lenses: dict, facts: list[Fact], fmt: str, n_agents: int, rounds: int, temperature: float,
               batch_size: int, gen_tokens: int, seed: int) -> list[dict]:
    """Free debate among n copies of the model. Round 0 answers are sampled (temperature), later
    rounds greedy. For every agent and round: its answer, and per candidate answer (the distinct
    round-0 answers plus the truth) the lens rank of the candidate at the last token, per layer."""
    solo = [debate_prompt(fmt, lm.tok, f, [], []) for f in facts]
    hist = [[[] for _ in range(n_agents)] for _ in facts]  # [fact][agent][round]
    first = generate(lm, [p for p in solo for _ in range(n_agents)], gen_tokens, batch_size, temperature, seed,
                     desc="debate r0")
    cands, cand_roles = [], []
    for i, f in enumerate(facts):
        answers = first[i * n_agents : (i + 1) * n_agents]
        for a in range(n_agents):
            hist[i][a].append(answers[a])
        c = [f.e3] + [x for x in dict.fromkeys(answers) if x and not matches(x, f.e3_aliases)]
        seen, uniq = set(), []
        for x in c:  # merge candidates that are the same answer
            if norm(x) not in seen:
                seen.add(norm(x))
                uniq.append(x)
        cands.append(uniq)
        ids = [probe_tokens(lm.tok, x) for x in uniq]
        count: dict[int, int] = {}
        for s in ids:
            for t in set(s):
                count[t] = count.get(t, 0) + 1
        cand_roles.append([[t for t in s if count[t] == 1] for s in ids])
    records = []
    for r in range(rounds + 1):
        prompts, index = [], []
        for i, f in enumerate(facts):
            for a in range(n_agents):
                own = hist[i][a][:r]
                others = [[hist[i][b][t] for b in range(n_agents) if b != a] for t in range(r)]
                prompts.append(debate_prompt(fmt, lm.tok, f, own, others))
                index.append((i, a))
        ranks = _candidate_ranks(lm, lenses, prompts, [cand_roles[i] for i, _ in index], batch_size)
        if r > 0:
            answers = generate(lm, prompts, gen_tokens, batch_size, desc=f"debate r{r}")
            for (i, a), ans in zip(index, answers):
                hist[i][a].append(ans)
        for k, (i, a) in enumerate(index):
            f = facts[i]
            ans = hist[i][a][r]
            records.append({"uid": f.uid, "category": f.category, "agent": a, "round": r, "answer": ans,
                            "correct": matches(ans, f.e3_aliases), "candidates": cands[i],
                            "candidate_ranks": {name: ranks[name][k] for name in lenses}})
    return records


def _candidate_ranks(lm: LM, lenses: dict, prompts: list[str], cand_ids: list[list[list[int]]],
                     batch_size: int) -> dict[str, list[list[list[int]]]]:
    """{lens: per prompt, per candidate, per layer: min rank of the candidate's tokens (-1 if none)}."""
    out = {name: [None] * len(prompts) for name in lenses}
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        enc = lm._encode([prompts[i] for i in idx])
        with capture_last_positions(lm.model, 1) as resid:
            lm.model(**enc)
        union = sorted({t for i in idx for c in cand_ids[i] for t in c}) or [0]
        uid = torch.tensor(union, device=lm.device)
        for name, lens in lenses.items():
            per_layer = [token_ranks(lens.logits(h[:, -1], l).float(), uid.expand(len(idx), -1)).cpu().numpy()
                         for l, h in enumerate(resid)]  # L x [B, U]
            for b, i in enumerate(idx):
                out[name][i] = [[int(min(per_layer[l][b, union.index(t)] for t in c)) if c else -1
                                 for l in range(len(resid))] for c in cand_ids[i]]
    return out
