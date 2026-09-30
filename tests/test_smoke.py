"""End-to-end run on a tiny randomly initialised Llama with a word-level tokenizer (no downloads)."""
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from silent_dissent import metrics as M
from silent_dissent.debate import run_debate
from silent_dissent.experiments import (build_states, expand_grid, generate_reasons, run_baseline,
                                        run_intervention, run_pressure)
from silent_dissent.intervene import inject_last_position, letter_direction
from silent_dissent.lenses import LogitLens
from silent_dissent.model import LM
from silent_dissent.prompts import ANSWER_PREFIX, CONDITIONS

@pytest.fixture(scope="module")
def lm(request):
    from silent_dissent.data import load_items
    from silent_dissent.debug import make_model, make_tokenizer
    from tests.conftest import FIXTURES

    items = load_items(str(FIXTURES / "items.jsonl"))
    tok = make_tokenizer(items)
    model = make_model(tok)
    return LM(model, tok, list("ABCD"), ANSWER_PREFIX), items


def test_logit_lens_last_layer_matches_output(lm):
    lm_, items = lm
    lens = LogitLens(lm_.model)
    base = run_baseline(lm_, lens, items, batch_size=4)
    for r in base:
        assert r["lens_logits"][-1] == pytest.approx(r["final_logits"], abs=1e-2)
    assert len(base[0]["lens_logits"]) == lm_.n_layers


def test_batching_does_not_change_readout(lm):
    lm_, items = lm
    lens = LogitLens(lm_.model)
    a = run_baseline(lm_, lens, items, batch_size=1)
    b = run_baseline(lm_, lens, items, batch_size=6)
    for x, y in zip(a, b):
        assert x["final_logits"] == pytest.approx(y["final_logits"], abs=1e-3)


def test_injection(lm):
    lm_, items = lm
    lens = LogitLens(lm_.model)
    base = run_baseline(lm_, lens, items, batch_size=6)
    last = lm_.n_layers - 1
    for k, letter in enumerate(lm_.letters):
        d = letter_direction(lens, lm_.letter_ids, k, last).detach()
        D = d.expand(len(items), -1)
        from silent_dissent.experiments import batched_readout
        from silent_dissent.prompts import render, solo_messages

        texts = [render(lm_.tok, solo_messages(it)) for it in items]
        outs = batched_readout(lm_, lens, texts, 6, ctx_fn=lambda idx: inject_last_position(lm_.model, last, D[idx], 100.0))
        assert all(o["stated"] == letter for o in outs)
    # alpha = 0 is a no-op
    outs0 = batched_readout(lm_, lens, texts, 6, ctx_fn=lambda idx: inject_last_position(lm_.model, last, D[idx], 0.0))
    for o, b in zip(outs0, base):
        assert o["final_logits"] == pytest.approx(b["final_logits"], abs=1e-3)


def test_position_readout_matches_answer_readout(lm):
    import numpy as np
    from silent_dissent.experiments import batched_position_readout
    from silent_dissent.prompts import render, solo_messages

    lm_, items = lm
    lens = LogitLens(lm_.model)
    texts = [render(lm_.tok, solo_messages(it)) for it in items]
    base = run_baseline(lm_, lens, items, batch_size=4)
    outs, arrays, tails = batched_position_readout(lm_, {"logit": lens}, texts, 4, n_last=3)
    assert arrays["logit"].shape == (len(items), 3, lm_.n_layers, 4)
    for i, (o, b) in enumerate(zip(outs, base)):
        assert o["stated"] == b["stated"]
        assert np.allclose(arrays["logit"][i, -1], np.array(b["lens_logits"]), rtol=1e-2, atol=5e-2)  # float16
    # left padding / batching does not change the earlier positions either
    _, single, tails1 = batched_position_readout(lm_, {"logit": lens}, texts, 1, n_last=3)
    assert np.allclose(single["logit"], arrays["logit"], rtol=1e-2, atol=5e-2)
    assert (tails1 == tails).all()


def test_end_to_end(lm):
    lm_, items = lm
    lens = LogitLens(lm_.model)
    base = run_baseline(lm_, lens, items, batch_size=4)
    by_id = {it.item_id: it for it in items}
    settings = expand_grid({"condition": list(CONDITIONS), "n_peers": [0, 1, 3], "peer_style": ["answer_only", "with_reason"],
                            "target_mode": ["wrong", "correct"]})
    settings = [s for s in settings if (s["condition"] == "instructed") == (s["n_peers"] == 0)]
    reasons = generate_reasons(lm_, items[:2], {items[0].item_id: {"A", "C"}, items[1].item_id: {"B"}}, batch_size=2)
    assert set(reasons[items[0].item_id]) == {"A", "C"}
    states, meta = build_states(by_id, base, settings, seed=0, reasons=reasons)
    recs = run_pressure(lm_, lens, states, meta, rounds=2, batch_size=8)
    assert len(recs) == 2 * len(states)
    assert all(len(r["stated_history"]) == r["round"] + 1 for r in recs)
    assert {r["n_peers"] for r in recs if r["condition"] == "instructed"} == {0}

    M.flip_table(recs)
    M.silent_dissent_table(recs, layer=1, delta=1.0)
    M.mention_control_table(recs, layer=1)

    src = [r for r in recs if r["condition"] == "pressure" and r["round"] == 2][:6]
    src += [r for r in recs if r["condition"] == "instructed" and r["round"] == 2][:2]
    inter = run_intervention(lm_, lens, by_id, src, layer=2, alphas=[0.0, 0.5],
                             kinds=["original", "majority", "other", "random"], batch_size=4, seed=0)
    assert len(inter) == len(src) * 8
    # alpha=0 must reproduce the original pressure run exactly (replay is faithful)
    for r in inter:
        if r["alpha"] == 0.0:
            assert r["stated"] == r["stated_before"]
    M.intervention_table(inter)
    run_intervention(lm_, lens, by_id, base[:3], layer=2, alphas=[0.1], kinds=["other"], batch_size=4, seed=0)

    deb = run_debate(lm_, lens, items, n_agents=3, rounds=2, temperature=1.0, batch_size=8, seed=0)
    assert len(deb) == len(items) * 3 * 3
    M.aggregation_table(deb, layer=1)
