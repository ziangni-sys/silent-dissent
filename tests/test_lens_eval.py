import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from silent_dissent.lens_eval import (make_two_hop_item, probe_ranks, text_fidelity, token_ranks,  # noqa: E402
                                      two_hop_ranks)
from silent_dissent.lenses import LogitLens  # noqa: E402
from silent_dissent.model import LM  # noqa: E402
from silent_dissent.prompts import ANSWER_PREFIX, format_question  # noqa: E402


@pytest.fixture(scope="module")
def lm():
    from silent_dissent.data import load_items
    from silent_dissent.debug import make_model, make_tokenizer
    from tests.conftest import FIXTURES

    items = load_items(str(FIXTURES / "items.jsonl"))
    tok = make_tokenizer(items)
    return LM(make_model(tok), tok, list("ABCD"), ANSWER_PREFIX), items


def test_token_ranks():
    logits = torch.tensor([[0.1, 3.0, 2.0, -1.0], [5.0, 4.0, 3.0, 2.0]])
    ids = torch.tensor([[1, 2, 3], [3, 0, 1]])
    assert token_ranks(logits, ids).tolist() == [[0, 1, 3], [3, 0, 1]]


def test_probe_ranks_matches_model_output_in_input_order(lm):
    lm_, _ = lm
    prompts = ["Do not mention any other option and the correct answer is", "the correct answer is", "Agent 2 : one"]
    offsets = [-3, -2, -1]
    probe = [lm_.tok(w, add_special_tokens=False)["input_ids"][0] for w in ("one", "two", "Agent", "the")]
    top1, ranks = probe_ranks(lm_, {"logit": LogitLens(lm_.model)}, prompts, offsets, probe, batch_size=2, n_last=4)
    r = ranks["logit"]
    assert r.shape == (3, 2, lm_.n_layers, 4) and r.dtype == np.int16
    for i, p in enumerate(prompts):  # batches are length-sorted; results must come back in input order
        enc = lm_._encode([p])
        with torch.no_grad():
            logits = lm_.model(**enc).logits[:, -1].float()
        assert top1[i] == int(logits.argmax())
        assert np.array_equal(r[i, 1, -1], token_ranks(logits, torch.tensor([probe])).numpy()[0])


def test_text_fidelity_last_layer_is_exact(lm):
    lm_, items = lm
    stream = [i for it in items for i in lm_.tok(format_question(it), add_special_tokens=False)["input_ids"]]
    windows = torch.tensor(stream[: 3 * 24]).view(3, 24)
    res = text_fidelity(lm_.model, {"logit": LogitLens(lm_.model)}, windows, batch_size=2, skip_first=4)["logit"]
    assert res["kl"][-1] == pytest.approx(0.0, abs=1e-4)
    assert res["top1_agree"][-1] == 1.0 and res["top5_contains"][-1] == 1.0
    assert (res["kl"][:-1] >= -1e-6).all()


def test_two_hop_item_offset_and_ranks(lm):
    lm_, _ = lm
    it = make_two_hop_item(lm_.tok, "t", "the correct answer is ", "Agent", " in the end", "Argue", "one", "two")
    assert it is not None and it.subject_offset == -4
    toks = lm_.tok(it.prompt, add_special_tokens=False)["input_ids"]
    assert toks[it.subject_offset] == lm_.tok("Agent", add_special_tokens=False)["input_ids"][0]
    assert make_two_hop_item(lm_.tok, "t", "the ", "Agent", " end", "one", "one", "two") is None  # probes must differ

    items = [it, make_two_hop_item(lm_.tok, "t", "Do not mention ", "Agent", " is", "Argue", "two", "one")]
    lens = LogitLens(lm_.model)
    top1, ranks = two_hop_ranks(lm_, {"logit": lens}, items, batch_size=2, n_last=6)
    assert ranks["logit"].shape == (2, 2, lm_.n_layers, 3)
    enc = lm_._encode([x.prompt for x in items])
    with torch.no_grad():
        logits = lm_.model(**enc).logits[:, -1].float()
    probe = torch.tensor([x.ids for x in items])
    # logit lens at the last layer and last position == ranks in the model's own output
    assert np.array_equal(ranks["logit"][:, 1, -1], token_ranks(logits, probe).numpy())
    assert np.array_equal(top1, logits.argmax(-1).numpy())
