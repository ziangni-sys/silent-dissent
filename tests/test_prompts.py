import json
import random
from pathlib import Path

import pytest

from silent_dissent.experiments import build_states, expand_grid, replay_state
from silent_dissent.prompts import (WITHHELD_TURN, DebateState, answer_turn, choose_control, choose_target,
                                    format_instruction)


def _state(item, cond, n=3, original="A", target="C", control="D"):
    return DebateState(item=item, condition=cond, n_peers=n, original=original, target=target, control=control)


def run_rounds(st, answers):
    for a in answers:
        st.advance()
        st.record(a)
    return st


def test_choose_target_and_control():
    letters = list("ABCD")
    for seed in range(50):
        rng = random.Random(seed)
        t = choose_target("A", "B", letters, "wrong", rng)
        assert t not in ("A", "B")
        c = choose_control("A", t, "B", letters, rng)
        assert c not in ("A", "B", t)
    assert choose_target("B", "B", letters, "correct", random.Random(0)) is None
    assert choose_target("A", "B", letters, "correct", random.Random(0)) == "B"


def test_peer_answers_by_condition(items):
    it = items[0]
    assert _state(it, "pressure").peer_answers() == ["C"] * 3
    assert _state(it, "agree").peer_answers() == ["A"] * 3
    assert _state(it, "split", n=3).peer_answers() == ["C", "C", "A"]
    with pytest.raises(ValueError):
        _state(it, "split", n=1)


def test_mention_control_matches_original_count(items):
    st = run_rounds(_state(items[0], "mention_control"), ["A", "C", "C"])
    peer_text = "\n".join(st.peer_messages)
    own_original = sum(1 for s in st.stated[:-1] if s == "A")  # answers visible before the last peer message
    assert peer_text.count("considered D") == own_original == st.control_mentions


def test_remove_original_hides_own_answers(items):
    st = run_rounds(_state(items[0], "remove_original"), ["C", "C"])
    st.advance()
    assistant = [m["content"] for m in st.messages() if m["role"] == "assistant"]
    assert assistant == [WITHHELD_TURN] * 3
    st2 = run_rounds(_state(items[0], "pressure"), ["C", "C"])
    st2.advance()
    assert [m["content"] for m in st2.messages() if m["role"] == "assistant"] == [answer_turn(a) for a in "ACC"]


def test_instructed_positive_control(items):
    st = run_rounds(_state(items[0], "instructed", n=0), ["C", "C"])
    st.advance()
    msgs = st.messages()
    assert [m["content"] for m in msgs if m["role"] == "user"][1:] == [format_instruction("C")] * 3
    assert [m["content"] for m in msgs if m["role"] == "assistant"] == [answer_turn(a) for a in "ACC"]
    assert "Agent 2" not in format_instruction("C") and "C" in format_instruction("C")
    assert st.peer_answers() == []
    with pytest.raises(ValueError):
        _state(items[0], "instructed", n=3)
    with pytest.raises(ValueError):
        _state(items[0], "pressure", n=0)


def test_instructed_prompt_matches_prereg_addendum():
    addendum = json.loads((Path(__file__).parents[1] / "prereg/qwen35_4b_addendum1.json").read_text())
    assert f"'{format_instruction('{target}')}'" in addendum["prompt"]


def test_instructed_only_in_its_own_cell(items):
    baseline = [{"item_id": it.item_id, "original": "A", "original_correct": it.gold == "A"} for it in items]
    by_id = {it.item_id: it for it in items}
    settings = expand_grid({"condition": ["instructed"], "n_peers": [0, 3], "peer_style": ["answer_only", "with_reason"],
                            "target_mode": ["wrong"]})
    states, meta = build_states(by_id, baseline, settings, seed=0)
    assert len(states) == len(items)
    assert all(m["n_peers"] == 0 and m["peer_style"] == "answer_only" for m in meta)
    # same target as the pressure condition for the same item and target mode
    pressure, _ = build_states(by_id, baseline, [{"condition": "pressure", "n_peers": 3, "peer_style": "answer_only",
                                                  "target_mode": "wrong"}], seed=0)
    assert [s.target for s in states] == [s.target for s in pressure]


def test_replay_reproduces_messages(items):
    baseline = [{"item_id": it.item_id, "original": "A", "original_correct": it.gold == "A"} for it in items]
    by_id = {it.item_id: it for it in items}
    settings = expand_grid({"condition": ["mention_control"], "n_peers": [3], "peer_style": ["answer_only"],
                            "target_mode": ["wrong"]})
    states, meta = build_states(by_id, baseline, settings, seed=0)
    st, m = states[0], meta[0]
    history = ["A", "B", "C"]
    for r in range(1, 3):
        st.advance()
        msgs = st.messages()
        rec = {**m, "round": r, "stated_history": history[: r + 1]}
        assert replay_state(by_id[m["item_id"]], rec).messages() == msgs
        st.record(history[r])


def test_agree_deduplicated_across_target_modes(items):
    baseline = [{"item_id": it.item_id, "original": "A", "original_correct": it.gold == "A"} for it in items]
    by_id = {it.item_id: it for it in items}
    settings = expand_grid({"condition": ["agree"], "n_peers": [3], "peer_style": ["answer_only"],
                            "target_mode": ["wrong", "correct"]})
    states, _ = build_states(by_id, baseline, settings, seed=0)
    assert len(states) == len(items)
