"""Real multi-agent debate: N copies of the same model exchange answers.

Every agent sees itself as "Agent 1" and the others as "Agent 2..N", which is
exactly the framing of the scripted-peer paradigm, so the two settings share
one prompt distribution. Round-0 answers are sampled from the letter
distribution (temperature T) so that agents can start out disagreeing; later
rounds are greedy, matching the scripted setting.
"""
from __future__ import annotations

import torch

from .data import MCQItem
from .experiments import batched_readout, item_rng
from .model import LM
from .prompts import SYSTEM_PROMPT, answer_turn, format_peer_message, format_question, render, solo_messages


def agent_messages(item: MCQItem, own: list[str], others: list[list[str]]) -> list[dict]:
    """own[r] is the agent's answer at round r; others[r] the other agents' answers."""
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": format_question(item)}]
    for r, peers in enumerate(others):
        msgs.append({"role": "assistant", "content": answer_turn(own[r])})
        msgs.append({"role": "user", "content": format_peer_message([answer_turn(a) for a in peers])})
    return msgs


def run_debate(
    lm: LM, lens, items: list[MCQItem], n_agents: int, rounds: int, temperature: float, batch_size: int, seed: int
) -> list[dict]:
    records = []
    solo = batched_readout(lm, lens, [render(lm.tok, solo_messages(it)) for it in items], batch_size, desc="debate r0")
    answers: list[list[list[str]]] = []  # [item][round][agent]
    for it, o in zip(items, solo):
        probs = torch.softmax(torch.tensor(o["final_logits"]) / temperature, -1)
        g = torch.Generator().manual_seed(item_rng(seed, it.item_id, "debate").randrange(2**31))
        draws = torch.multinomial(probs, n_agents, replacement=True, generator=g).tolist()
        r0 = [lm.letters[k] for k in draws]
        answers.append([r0])
        for a in range(n_agents):
            records.append({"item_id": it.item_id, "dataset": it.dataset, "gold": it.gold, "letters": it.letters,
                            "agent": a, "round": 0, **o, "greedy": o["stated"], "stated": r0[a]})

    for r in range(1, rounds + 1):
        texts, index = [], []
        for i, it in enumerate(items):
            hist = answers[i]
            for a in range(n_agents):
                own = [hist[t][a] for t in range(r)]
                others = [[hist[t][b] for b in range(n_agents) if b != a] for t in range(r)]
                texts.append(render(lm.tok, agent_messages(it, own, others)))
                index.append((i, a))
        outs = batched_readout(lm, lens, texts, batch_size, desc=f"debate r{r}")
        new = [[None] * n_agents for _ in items]
        for (i, a), o in zip(index, outs):
            new[i][a] = o["stated"]
            it = items[i]
            records.append({"item_id": it.item_id, "dataset": it.dataset, "gold": it.gold, "letters": it.letters,
                            "agent": a, "round": r, **o})
        for i in range(len(items)):
            answers[i].append(new[i])
    return records
