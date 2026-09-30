import numpy as np
import pytest

from silent_dissent import entity_metrics as EM
from silent_dissent.entity_facts import (CAPITALS, HANDMADE, PROBE_MODES, ROLES, Fact, assign_partners,
                                         handmade_facts, load_twohopfact, matches, probe_tokens, role_probes, siblings,
                                         split_of)
from silent_dissent.entity_prompts import AGAIN, FIRST, LATER, REPHRASE, STYLES, WITHHELD, EntityState, statement
from tests.conftest import FIXTURES

SAMPLE = str(FIXTURES / "twohopfact_sample.csv")


@pytest.fixture(scope="module")
def facts():
    return load_twohopfact(SAMPLE)


@pytest.fixture(scope="module")
def triples(facts):
    return assign_partners(facts, facts, seed=0)


# ---------------------------------------------------------------------------- facts


def test_twohopfact_loader(facts):
    assert len(facts) == 12
    orwell = next(f for f in facts if f.e1 == "Nineteen Eighty-Four" and f.e3 == "Motihari")
    assert "Eric Blair" in orwell.e2_aliases and orwell.e2_aliases[0] == "George Orwell"
    assert orwell.r2_prompt == "George Orwell was born in the city of"
    assert orwell.composed == "The author of the novel Nineteen Eighty-Four was born in the city of"
    assert all(f.e1 in f.composed and f.e2 not in f.composed for f in facts)
    assert Fact.from_dict(orwell.to_dict()) == orwell


def test_handmade_facts_are_consistent():
    hand = handmade_facts()
    assert len(hand) == sum(len(rows) for rows, _ in HANDMADE.values())
    for f in hand:
        assert f.e3 in CAPITALS.values() and f.e1 in f.composed
        assert not any(a in f.composed for a in f.e2_aliases) and f.e3 not in f.composed  # nothing leaks
        assert f.r2_prompt.startswith("The capital city of ")


def test_matching():
    assert matches("Motihari, a town in Bihar", ["Motihari"])
    assert matches("Brasilia", ["Brasília"]) and matches("the Czech Republic", ["Czech Republic", "the Czech Republic"])
    assert not matches("Parisian suburbs", ["Paris"]) and not matches("", ["Paris"])


def test_split_by_bridge(facts):
    s = {f.uid: split_of(f, 0.5, 0) for f in facts}
    assert set(s.values()) <= {"dev", "test"}
    for a in facts:  # the same bridge never lands in both splits
        for b in facts:
            if a.e2 == b.e2:
                assert s[a.uid] == s[b.uid]
    assert s == {f.uid: split_of(f, 0.5, 0) for f in facts}


def test_partners_and_siblings(facts, triples):
    uids = {f.uid for f, _, _ in triples}
    putin = next(f for f in facts if f.e1 == "Vladimir Putin")
    assert putin.uid not in uids  # the only other answers are Russia / the USA twice: no distinct control
    for f, p, c in triples:
        assert f.category == p.category == c.category
        assert len({f.e2, p.e2, c.e2}) == 3 and len({f.e3, p.e3, c.e3}) == 3
    sib = siblings(facts, facts)
    orwell = next(f for f in facts if f.e3 == "Motihari")
    assert sib[orwell.uid].e3 == "India" and sib[orwell.uid].e1 == orwell.e1


# -------------------------------------------------------------------------- prompts


def _state(triples, facts, condition, style=None, **kw):
    f, p, c = next(t for t in triples if t[0].e3 == "Motihari")
    sib = siblings(facts, facts).get(f.uid)
    st = EntityState(f, p, c, condition, style, sibling=sib, **kw)
    st.record("Motihari")
    if condition != "solo":
        st.advance()
    return st


def test_peer_lines_by_condition(triples, facts):
    f, p, _ = next(t for t in triples if t[0].e3 == "Motihari")
    lines = {s: _state(triples, facts, "pressure", s).peer_lines() for s in STYLES}
    assert lines["answer"] == [statement(f.composed, p.e3)] * 3
    assert p.e2 in lines["hop1"][0] and p.e3 in lines["hop1"][0] and f.e2 not in lines["hop1"][0]
    assert f.e2 in lines["hop2"][0] and p.e3 in lines["hop2"][0] and p.e2 not in lines["hop2"][0]
    mention = _state(triples, facts, "mention").peer_lines()
    assert p.e3 in mention[0] and f.e3 in mention[0] and mention[1:] == [statement(f.composed, f.e3)] * 2
    assert p.e2 in _state(triples, facts, "mention_bridge").peer_lines()[0]
    assert _state(triples, facts, "instructed").peer_lines() == []
    ret = _state(triples, facts, "retention")
    assert ret.final_prefix() == ret.sibling.composed != f.composed
    with pytest.raises(ValueError):
        ret.advance()  # single round
    with pytest.raises(ValueError):
        EntityState(f, p, p, "pressure")  # style missing
    with pytest.raises(ValueError):
        EntityState(f, p, p, "agree", "answer")


@pytest.fixture(scope="module")
def tok(facts, triples):
    pytest.importorskip("transformers")
    from silent_dissent.data import load_items
    from silent_dissent.debug import make_tokenizer

    texts = []
    sib = siblings(facts, facts)
    for f, p, c in triples:
        for cond, style in [("pressure", s) for s in STYLES] + [(x, None) for x in
                                                                 ("agree", "mention", "mention_bridge", "instructed",
                                                                  "hypothetical", "retention")]:
            if cond == "retention" and f.uid not in sib:
                continue
            st = EntityState(f, p, c, cond, style, sibling=sib.get(f.uid))
            st.record(f.e3)
            st.advance()
            texts += [st.render("plain"), f.e2, p.e2, c.e2, f.e3, p.e3, c.e3] + f.e2_aliases + f.e3_aliases
    return make_tokenizer(load_items(str(FIXTURES / "items.jsonl")), extra_texts=texts)


@pytest.mark.parametrize("fmt", ["plain", "chat"])
def test_render_and_segments(triples, facts, tok, fmt):
    st = _state(triples, facts, "pressure", "hop1")
    text, spans = st.render_with_segments(fmt, tok)
    labels = [s[0] for s in spans]
    assert labels == ["own", "peer1", "peer2", "peer3", "moderator", "final"]
    for label, a, b in spans:
        seg = text[a:b]
        if label == "own":
            assert seg == statement(st.fact.composed, "Motihari")
        elif label.startswith("peer"):
            assert seg == st.peer_lines()[0]
        elif label == "final":
            assert seg == st.fact.composed and b == len(text)
    hidden = _state(triples, facts, "pressure", "answer", hidden_own=True)
    assert WITHHELD in hidden.render(fmt, tok) and "Motihari" not in hidden.render(fmt, tok)
    # absent: no statement of the agent at all, the peers first, "complete the statement" (not "again")
    absent = _state(triples, facts, "pressure", "hop1", absent_own=True)
    text, spans = absent.render_with_segments(fmt, tok)
    assert [s[0] for s in spans] == ["peer1", "peer2", "peer3", "moderator", "final"]
    assert "Motihari" not in text and WITHHELD not in text and AGAIN not in text and FIRST in text
    for label, a, b in spans:
        if label.startswith("peer"):
            assert text[a:b] == absent.peer_lines()[0]
        elif label == "moderator":
            assert text[a:b] == FIRST
    if fmt == "plain":  # the question is followed directly by the peers
        assert f"...\nAgent 2: {absent.peer_lines()[0]}" in text
    with pytest.raises(ValueError):
        _state(triples, facts, "pressure", "answer", hidden_own=True, absent_own=True)


def test_role_probes_are_distinct(triples, tok):
    for f, p, c in triples:
        for mode in PROBE_MODES:
            roles = role_probes(tok, f, p, c, mode)
            assert set(roles) == set(ROLES)
            flat = [t for ids in roles.values() for t in ids]
            assert len(flat) == len(set(flat))


def test_probe_modes(tok):
    words, last = probe_tokens(tok, "George Orwell"), probe_tokens(tok, "George Orwell", "last_word")
    assert last == probe_tokens(tok, "Orwell") and set(last) < set(words)
    assert probe_tokens(tok, "the Czech Republic", "last_word") == probe_tokens(tok, "Czech")  # stopwords skipped
    with pytest.raises(ValueError):
        probe_tokens(tok, "Orwell", "surname")


def _conversation(triples, condition, style, later, rounds, fmt, tok, answers=("Motihari", "Paris", "Rome")):
    f, p, c = next(t for t in triples if t[0].e3 == "Motihari")
    st = EntityState(f, p, c, condition, style, later=later)
    st.record(answers[0])
    for r in range(1, rounds + 1):
        st.advance()
        if r < rounds:
            st.record(answers[r])
    return st, st.render_with_segments(fmt, tok)


@pytest.mark.parametrize("fmt", ["plain", "chat"])
def test_later_round_variants(triples, tok, fmt):
    for cond, style in [("pressure", s) for s in STYLES] + [("agree", None), ("mention", None), ("mention_bridge", None)]:
        ref1 = _conversation(triples, cond, style, "repeat", 1, fmt, tok)[1]
        for later in LATER:  # round 1 is the same prompt in every variant
            assert _conversation(triples, cond, style, later, 1, fmt, tok)[1] == ref1
    rep, (rep_text, _) = _conversation(triples, "pressure", "answer", "repeat", 3, fmt, tok)
    for later in ("rephrase", "summary"):
        st, (text, spans) = _conversation(triples, "pressure", "answer", later, 3, fmt, tok)
        assert text != rep_text and text.endswith(st.fact.composed)
        assert [s[0] for s in spans] == ["own", "peer1", "peer2", "peer3", "moderator", "final"]
        own = next(text[a:b] for lab, a, b in spans if lab == "own")
        assert own == statement(st.fact.composed, "Rome")  # the last round's own statement = the round-2 answer
        for lab, a, b in spans:
            if lab.startswith("peer"):
                assert st.peer.e3 in text[a:b]
    summ, (text, _) = _conversation(triples, "pressure", "answer", "summary", 3, fmt, tok)
    assert "Round 1: Agent 1 answered Motihari; Agents 2 to 4 answered " + summ.peer.e3 in text
    assert "Round 2: Agent 1 answered Paris" in text and len(text) < len(rep_text)
    assert text.count(statement(summ.fact.composed, summ.peer.e3)) == 3  # only the last round in full
    reph, (text, _) = _conversation(triples, "pressure", "answer", "rephrase", 3, fmt, tok)
    assert "Round 3. " + AGAIN in text and "Round 2. " + AGAIN in text and any(t.format("") in text for t in REPHRASE)
    with pytest.raises(ValueError):
        EntityState(summ.fact, summ.peer, summ.control, "agree", later="shorter")


# ---------------------------------------------------------------------- model runs


@pytest.fixture(scope="module")
def lm(tok):
    from silent_dissent.debug import make_model
    from silent_dissent.model import LM
    from silent_dissent.prompts import ANSWER_PREFIX

    return LM(make_model(tok), tok, list("ABCD"), ANSWER_PREFIX)


def test_readout_matches_model_output(lm, triples):
    import torch

    from silent_dissent.entity_experiments import entity_readout, readout_inputs, subject_offset
    from silent_dissent.lens_eval import token_ranks
    from silent_dissent.lenses import LogitLens

    states = [EntityState(f, p, c, "solo") for f, p, c in triples]
    roles = [role_probes(lm.tok, f, p, c) for f, p, c in triples]
    prompts, offs = readout_inputs(lm.tok, states, "plain")
    assert all(o < -1 for o in offs)  # the subject precedes the final words of the statement
    assert offs[0] == subject_offset(lm.tok, prompts[0], states[0].fact.e1)
    arr = entity_readout(lm, {"logit": LogitLens(lm.model)}, prompts, offs, roles, batch_size=2)["logit"]
    assert arr.shape == (len(states), 2, lm.n_layers, len(ROLES))
    for i, p in enumerate(prompts):  # last layer, last token == ranks in the model output
        enc = lm._encode([p])
        with torch.no_grad():
            logits = lm.model(**enc).logits[:, -1].float()
        for r, role in enumerate(ROLES):
            ids = roles[i][role]
            want = int(token_ranks(logits, torch.tensor([ids])).min()) if ids else -1
            assert arr[i, 1, -1, r] == min(want, 30000)


def test_extended_readout(lm, triples):
    from silent_dissent.entity_experiments import entity_readout, readout_inputs, timeline_readout
    from silent_dissent.entity_prompts import mu
    from silent_dissent.lenses import LogitLens

    lenses = {"logit": LogitLens(lm.model)}
    states = [EntityState(f, p, c, "solo") for f, p, c in triples]
    roles = [role_probes(lm.tok, f, p, c) for f, p, c in triples]
    prompts, offs, moffs = readout_inputs(lm.tok, states, "plain", extended=True)
    assert all(m < -1 for m in moffs)  # the description ends before the answer position
    for p, m, st in zip(prompts, moffs, states):  # the mention token closes the bridge's description
        ids = lm.tok(p, add_special_tokens=False)["input_ids"]
        assert lm.tok.decode(ids[: len(ids) + m + 1]).replace(" ", "").endswith(mu(st.fact).replace(" ", ""))
    base = entity_readout(lm, lenses, prompts, offs, roles, batch_size=2)["logit"]
    ext = entity_readout(lm, lenses, prompts, offs, roles, batch_size=2, mention_offsets=moffs)["logit"]
    assert ext.shape == (len(states), 4, lm.n_layers, len(ROLES)) and np.array_equal(ext[:, :2], base)
    tl = timeline_readout(lm, lenses, prompts, [-m for m in moffs], roles, list(range(lm.n_layers)), batch_size=2)
    for i, w in enumerate(tl["logit"]):
        assert np.array_equal(ext[i, 2], w[0]) and np.array_equal(ext[i, 3], w.min(0))
    for pos in ("mention", "span"):
        assert EM.curve(ext, list(range(len(states))), "orig_bridge", pos, 10).shape == (lm.n_layers,)


def test_rounds_timeline_injection_debate(lm, triples, facts):
    from silent_dissent.entity_experiments import (classify, generate, known_check, run_debate, run_injection,
                                                   run_rounds, timeline_readout, token_labels)
    from silent_dissent.lenses import LogitLens

    lenses = {"logit": LogitLens(lm.model)}
    sib = siblings(facts, facts)
    solo = [EntityState(f, p, c, "solo") for f, p, c in triples]
    first = generate(lm, [st.render("plain") for st in solo], 4, batch_size=3)
    assert len(first) == len(triples) and all(isinstance(a, str) for a in first)
    states, meta, roles = [], [], []
    for (f, p, c), a in zip(triples, first):
        for cond, style in (("pressure", "hop2"), ("agree", None), ("retention", None)):
            if cond == "retention" and f.uid not in sib:
                continue
            st = EntityState(f, p, c, cond, style, sibling=sib.get(f.uid))
            st.record(a)
            states.append(st)
            meta.append({"uid": f.uid, "split": "dev", "format": "plain", "condition": cond, "style": style, "n_peers": 3})
            roles.append(role_probes(lm.tok, f, p, c))
    records, arrays = run_rounds(lm, lenses, states, meta, roles, "plain", rounds=2, batch_size=4, gen_tokens=4)
    single = sum(m["condition"] == "retention" for m in meta)
    assert len(records) == 2 * len(states) - single == len(arrays["logit"])
    assert {r["outcome"] for r in records} <= {"original", "peer", "sibling", "other"}
    assert classify(states[0].fact.e3, states[0].fact, states[0].peer) == "original"

    # timeline: the last window token must agree with the last-token readout at the same layers
    prompts, spans = zip(*[st.render_with_segments("plain", lm.tok) for st in states[:3]])
    windows, labels = zip(*[token_labels(lm.tok, p, s, s[0][1]) for p, s in zip(prompts, spans)])
    tl = timeline_readout(lm, lenses, list(prompts), list(windows), roles[:3], layers=[0, lm.n_layers - 1],
                          batch_size=2, chunk=5)["logit"]
    from silent_dissent.entity_experiments import entity_readout, readout_inputs

    _, offs = readout_inputs(lm.tok, states[:3], "plain")
    ref = entity_readout(lm, lenses, list(prompts), offs, roles[:3], batch_size=3)["logit"]
    for i in range(3):
        assert tl[i].shape == (windows[i], 2, len(ROLES)) and len(labels[i]) == windows[i]
        assert np.array_equal(tl[i][-1, 1], ref[i, 1, -1]) and labels[i][-1] == "final"
        assert "peer1" in labels[i] or states[i].condition == "retention"

    inj = run_injection(lm, lenses["logit"], states[:4], meta[:4], roles[:4], "plain", layers=[1], alphas=[0.0, 2.0],
                        kinds=["orig_bridge", "random"], batch_size=2, seed=0)
    assert inj and {r["top1_is"] for r in inj} <= {"original", "peer", "control", "other"}
    EM.injection_table(inj)

    deb = run_debate(lm, lenses, [f for f, _, _ in triples[:2]], "plain", n_agents=3, rounds=1, temperature=1.0,
                     batch_size=4, gen_tokens=3, seed=0)
    assert len(deb) == 2 * 3 * 2 and all(len(r["candidate_ranks"]["logit"][0]) == lm.n_layers for r in deb)
    EM.debate_table(deb, "logit", band=[0, 1])
    kc = known_check(lm, facts[:3], batch_size=3, max_new_tokens=3)
    assert len(kc) == 3 and set(kc[0]) == {"uid", "hop1", "hop2", "composed", "texts"}

    # metrics on the round records
    EM.outcome_table(records)
    t = EM.latent_table(records, arrays["logit"], k=10, band=[0, 1])
    assert {"orig_bridge", "orig_bridge_lo", "peer_bridge", "n"} <= set(t.columns)
    EM.layer_curves(records, arrays["logit"], 10, "last", ["condition"])
    tt = EM.timeline_table([{**meta[i], "outcome": "peer"} for i in range(3)], list(tl), list(labels), 10, [0, 1])
    assert not tt.empty


def test_debate_latent_vote_only_over_proposed_answers():
    # nobody proposes the truth (Paris); the readout ranks it best, then Lyon
    def rec(agent, rnd, answer):
        return {"uid": "u", "agent": agent, "round": rnd, "answer": answer, "correct": answer == "Paris",
                "candidates": ["Paris", "Lyon", "Nice"], "candidate_ranks": {"j": [[0, 0], [5, 5], [50, 50]]}}

    recs = [rec(a, r, ans) for r in (0, 1) for a, ans in enumerate(["Lyon", "Nice", "Lyon"])]
    t = EM.debate_table(recs, "j", band=[0, 1]).set_index("round")
    assert t.loc[0, "acc_latent"] == 0.0  # Lyon, the best-ranked proposed answer
    t_all = EM.debate_table(recs, "j", band=[0, 1], stated_only=False).set_index("round")
    assert t_all.loc[0, "acc_latent"] == 1.0  # the unrestricted rule would pick the unproposed truth


def _synthetic(rows):
    """records + a [N, 4, 2, R] rank array: orig / peer bridge ranks at position `slot`, 500 elsewhere."""
    recs, arr = [], []
    for uid, cond, style, n, rnd, outcome, orig, peer, slot in rows:
        recs.append({"uid": uid, "format": "plain", "condition": cond, "style": style, "n_peers": n, "round": rnd,
                     "outcome": outcome, "hidden_own": False})
        a = np.full((4, 2, len(ROLES)), 500, dtype=np.int16)
        a[slot, :, ROLES.index("orig_bridge")], a[slot, :, ROLES.index("peer_bridge")] = orig, peer
        arr.append(a)
    return recs, np.stack(arr)


def test_addendum_hypotheses():
    rows = []
    for rnd in (1, 2, 3):  # u1 gives in every round and keeps both bridges; u2 goes back in round 3
        rows += [("u1", "pressure", "answer", 3, rnd, "peer", 0, 0, 1),
                 ("u2", "pressure", "answer", 3, rnd, "peer" if rnd < 3 else "original", 500, 500, 1),
                 ("u3", "mention", None, 3, rnd, "original", 0, 500, 1)]
    recs, arr = _synthetic(rows)
    prereg = {"band": [0, 1], "k": 10, "lens": "jlens", "format": "plain"}
    v = {x["id"]: x for x in EM.evaluate_rounds(recs, {"jlens": arr}, prereg, {"n_peers": 3, "round": 3})}
    assert v["R1"]["n"] == 1 and v["R1"]["estimate"] == 1.0 and v["R1"]["supported"]
    assert v["R2"]["estimate"] == 1.0 and v["R2"]["n"] == (1, 1)
    assert v["change"]["estimate"] == 0.0 and v["change"]["supported"] is None

    # bridges: read at the span slot (3); item a flips with 1 and 3 peers and keeps its bridge both
    # times, item b only with 1 peer: per-item means 1.0 and 0.5
    recs, arr = _synthetic([("a", "pressure", "answer", 1, 1, "peer", 0, 500, 3),
                            ("a", "pressure", "answer", 3, 1, "peer", 0, 500, 3),
                            ("b", "pressure", "answer", 1, 1, "peer", 0, 0, 3),
                            ("b", "pressure", "answer", 3, 1, "peer", 500, 0, 3),
                            ("b", "pressure", "answer", 5, 1, "peer", 0, 0, 3),  # n_peers outside the addendum
                            ("c", "mention", None, 1, 1, "original", 0, 500, 3),
                            ("a", "pressure", "answer", 1, 1, "peer", 0, 500, 1)])  # other position: ignored
    recs[-1]["format"] = "chat"
    add = {"band": [0, 1], "k": 10, "lens": "jlens", "position": "span", "format": "plain", "n_peers": [1, 3]}
    logit = np.full_like(arr, 500)
    v = {x["id"]: x for x in EM.evaluate_bridges(recs, {"jlens": arr, "logit": logit}, add)}
    assert v["B1"]["n"] == 2 and v["B1"]["estimate"] == 0.75
    assert v["B2"]["estimate"] == 0.5 and v["B2"]["n"] == (2, 1)  # peer bridge: a 0, b 1 vs mention c 0
    assert v["B4"]["estimate"] == 0.75


def test_own_addendum_hypotheses():
    rows = [("a", "pressure", "answer", 3, 1, "peer", 0, 500, 1),       # hidden, gave in, keeps its bridge
            ("b", "pressure", "answer", 3, 1, "peer", 500, 500, 1),     # hidden, gave in, bridge gone
            ("a", "pressure", "answer", 3, 1, "peer", 500, 500, 1),     # absent, gave in, no bridge
            ("c", "pressure", "answer", 3, 1, "peer", 0, 500, 1),       # absent, gave in, bridge
            ("c", "pressure", "answer", 3, 1, "original", 0, 500, 1),   # absent, held: not in O3
            ("d", "pressure", "answer", 3, 1, "peer", 0, 500, 1),       # visible (main run): ignored
            ("e", "mention", None, 3, 1, "original", 500, 500, 1)]
    recs, arr = _synthetic(rows)
    for r, flag in zip(recs, ["hidden", "hidden", "absent", "absent", "absent", None, "hidden"]):
        r["hidden_own"], r["absent_own"] = flag == "hidden", flag == "absent"
    add = {"band": [0, 1], "k": 10, "lens": "jlens", "format": "plain", "n_peers": 3, "min_test_n": 0,
           "variants": ["hidden", "absent"], "styles": ["answer"],
           "hypotheses": [{"id": i} for i in ("O1", "O2", "O3", "O4")]}
    logit = np.full_like(arr, 500)
    v = {x["id"]: x for x in EM.evaluate_own(recs, {"jlens": arr, "logit": logit}, add)}
    assert v["O1"]["n"] == 2 and v["O1"]["estimate"] == 0.5
    assert v["O3"]["n"] == 2 and v["O3"]["estimate"] == 0.5 and v["O2"]["estimate"] == 0.0
    assert v["O4"]["estimate"] == 0.5
    add["min_test_n"] = 3  # two records per cell: no verdict
    assert all(x["supported"] is None for x in EM.evaluate_own(recs, {"jlens": arr, "logit": logit}, add))
    add["hypotheses"] = [{"id": "O1"}, {"id": "O4"}]  # absent not registered
    assert {x["id"] for x in EM.evaluate_own(recs, {"jlens": arr}, add)} == {"O1"}
    # paired per item: a gave in hidden (1) and absent (0) -> hidden - absent = 1 on one common item
    summ = EM.own_summary(recs, {"jlens": arr}, add, ref=(recs, {"jlens": arr}))
    assert summ["paired"]["hidden-absent"]["n_items"] == 1 and summ["paired"]["hidden-absent"]["diff"] == 1.0
    assert summ["rates"]["visible/answer"]["n"] == 1 and summ["rates"]["absent/answer"]["peer"] == 0.6667


def test_metrics_contrasts():
    ranks = np.array([[[[0, 50, 50, 50, 50, 50]]], [[[50, 0, 50, 50, 50, -1]]]], dtype=np.int16)  # [2, 1, 1, 6]
    ranks = np.repeat(ranks, 2, axis=1)  # both positions
    assert EM.curve(ranks, [0, 1], "orig_bridge", "last", 10)[0] == 0.5
    assert EM.curve(ranks, [1], "peer_bridge", "last", 10)[0] == 1.0
    assert np.isnan(EM.curve(ranks, [1], "orig_answer", "last", 10)[0])  # control answer unreadable
    assert EM.select_band(np.array([0.1, 0.6, 0.5, np.nan]), 0.5) == [1, 2]


def test_robustness_helpers():
    # ranks [3 records, 2 positions, 2 layers, 6 roles]: orig_bridge vs ctrl_bridge
    r = np.full((3, 2, 2, len(ROLES)), 500, dtype=np.int16)
    o, c = ROLES.index("orig_bridge"), ROLES.index("ctrl_bridge")
    r[0, :, :, o], r[0, :, :, c] = 9, 999      # two orders of magnitude ahead
    r[1, :, 0, o], r[1, :, 1, o] = 500, 30000  # a tie, then behind
    r[2, :, :, o] = -1                         # not readable
    lr = EM.rank_rows(r, [0, 1, 2], "orig_bridge", "last", [0, 1], "logrank")
    assert lr[0] == pytest.approx(2.0) and lr[1] == pytest.approx((0 + np.log10(501 / 30001)) / 2) and np.isnan(lr[2])
    win = EM.rank_rows(r, [0, 1, 2], "orig_bridge", "last", [0, 1], "win")
    assert win[0] == 1.0 and win[1] == pytest.approx(0.25) and np.isnan(win[2])

    # one label per value: the ordinary bootstrap; shared clusters keep two samples paired
    v = np.array([1.0, 2.0, 3.0, 4.0])
    d = EM.boot_means([(v, [0, 1, 2, 3])], 2000)[0]
    assert d.shape == (2000,) and abs(d.mean() - 2.5) < 0.05
    same = EM.boot_means([(v, ["a", "a", "b", "b"]), (v + 1, ["a", "a", "b", "b"])], 500)
    assert np.allclose(same[1] - same[0], 1.0)  # the same clusters drawn for both samples
    one = EM.boot_means([(v, ["x"] * 4)], 100)[0]
    assert np.allclose(one, 2.5)  # a single cluster: no variation
    s = EM.boot_summary(np.array([0.1, 0.2, 0.3, np.nan]))
    assert s["n_boot"] == 3 and s["p"] == pytest.approx(0.5) and s["ci"][0] >= 0.1
    assert EM.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_debate_kinds_and_transitions():
    rec = lambda uid, agent, rnd, answer, correct, cands=("Madrid", "Rome", "Paris"), known=True: {
        "uid": uid, "agent": agent, "round": rnd, "answer": answer, "correct": correct, "candidates": list(cands),
        "known": known}
    assert EM.debate_kind(rec("u", 0, 1, "Madrid", True)) == "correct"
    assert EM.debate_kind(rec("u", 0, 1, "Rome", False)) == "other_candidate"
    for junk in ("Correct", "identical to Agent 2", "the same as before", "", "Agent 3 is right"):
        assert EM.debate_kind(rec("u", 0, 1, junk, False)) == "degenerate", junk
    assert EM.debate_kind(rec("u", 0, 1, "Lisbon", False)) == "other"
    assert EM.same_answer("Rome", "rome, Italy") and not EM.same_answer("Rome", "Romeo") and not EM.same_answer("", "")
    records = [
        # u1: agent 0 gives in to agent 1's Rome; agent 2 holds against Rome
        rec("u1", 0, 0, "Madrid", True), rec("u1", 1, 0, "Rome", False), rec("u1", 2, 0, "Madrid", True),
        rec("u1", 0, 1, "Rome", False), rec("u1", 1, 1, "Rome", False), rec("u1", 2, 1, "Madrid", True),
        # u2: everyone right, one then switches to an answer nobody gave
        rec("u2", 0, 0, "Madrid", True), rec("u2", 1, 0, "Madrid", True), rec("u2", 2, 0, "Madrid", True),
        rec("u2", 0, 1, "Madrid", True), rec("u2", 1, 1, "Lisbon", False), rec("u2", 2, 1, "Madrid", True),
    ]
    labels = EM.debate_transitions(records)
    assert labels[:3] == [None] * 3
    assert labels[3:6] == ["gave_in", "was_wrong", "held"] and labels[9:12] == ["agreed", "switched", "agreed"]

    # readout summary on synthetic ranks: orig bridge in the top k for every record, control never
    ranks = np.full((len(records), 2, 3, len(ROLES)), 500, dtype=np.int16)
    ranks[:, :, :, ROLES.index("orig_bridge")] = 5
    arrays = {"jlens": ranks, "logit": np.full_like(ranks, 500)}
    s = EM.debate_bridge_summary(records, arrays, "jlens", [0, 1], 100, n_boot=200, min_n=1)
    assert s["cells"]["all/r1/gave_in/jlens"]["mean"] == 1.0 and s["cells"]["all/r1/gave_in/logit"]["mean"] == 0.0
    assert s["contrasts"]["all/r1/gave_in/jlens-logit"]["mean"] == 1.0 and s["counts"]["r1/gave_in"] == 1
    assert s["cells"]["known/round0_correct/jlens"]["n"] == 5


def test_debate_group_correct_matches_debate_table():
    """A round in which every agent gives an empty answer is a wrong group under the stated rule, as in
    debate_table (the check script had dropped it), and is excluded only when degenerate answers abstain."""
    def rec(uid, agent, rnd, answer, correct):
        return {"uid": uid, "agent": agent, "round": rnd, "answer": answer, "correct": correct,
                "candidates": ["Madrid", "Rome"], "candidate_ranks": {"logit": [[1, 1], [5, 5]]}}
    records = [rec("u1", a, 0, "Madrid", True) for a in range(3)] + [rec("u1", a, 1, "", False) for a in range(3)]
    records += [rec("u2", a, 0, "Madrid", True) for a in range(3)]
    records += [rec("u2", 0, 1, "Correct", False), rec("u2", 1, 1, "Madrid", True), rec("u2", 2, 1, "Rome", False)]
    table = EM.debate_table(records, "logit", [0, 1])
    by = {}
    for r in records:
        by.setdefault((r["uid"], r["round"]), []).append(r)
    for t in (0, 1):
        groups = [g for (u, rr), g in by.items() if rr == t]
        a = [EM.debate_group_correct(g) for g in groups]
        assert None not in a and np.mean(a) == table.loc[table["round"] == t, "acc_stated"].iloc[0]
    assert EM.debate_group_correct(by[("u1", 1)]) is False
    assert EM.debate_group_correct(by[("u1", 1)], drop_degenerate=True) is None
    assert EM.debate_group_correct(by[("u2", 1)], drop_degenerate=True) in (True, False)
