"""Jacobian lens: estimator vs. brute-force autograd, file format, and lens behaviour."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from silent_dissent import jlens
from silent_dissent.lenses import JLens, LogitLens
from silent_dissent.model import get_layers, layer_output, replace_layer_output

SKIP = 2


@pytest.fixture(scope="module")
def tiny():
    from silent_dissent.data import load_items
    from silent_dissent.debug import make_model, make_tokenizer
    from tests.conftest import FIXTURES

    items = load_items(str(FIXTURES / "items.jsonl"))
    tok = make_tokenizer(items)
    model = make_model(tok)
    ids = tok((FIXTURES / "corpus.txt").read_text().splitlines()[0], return_tensors="pt").input_ids[:, :12]
    return model, tok, ids


def brute_force_J(model, ids, src, tgt):
    """sum_{p'} d h_tgt[p'] / d h_src[p], averaged over p, both over valid positions."""
    layers = get_layers(model)
    with torch.no_grad():
        store = {}
        hk = layers[src].register_forward_hook(lambda m, i, o: store.__setitem__("h", layer_output(o)))
        model(input_ids=ids, use_cache=False)
        hk.remove()
    x0 = store["h"].detach().clone()

    def f(x):
        out = {}
        h1 = layers[src].register_forward_hook(lambda m, i, o: replace_layer_output(o, x))
        h2 = layers[tgt].register_forward_hook(lambda m, i, o: out.__setitem__("h", layer_output(o)))
        try:
            model(input_ids=ids, use_cache=False)
        finally:
            h1.remove(), h2.remove()
        return out["h"]

    full = torch.autograd.functional.jacobian(f, x0)[0, :, :, 0]  # [S_out, d, S_in, d]
    pos = jlens.valid_positions(ids.shape[1], SKIP)
    return full[pos][:, :, pos].sum(0).mean(1)  # [d, d]


@pytest.mark.parametrize("dim_batch", [8, 5])  # 5 does not divide d_model = 64
def test_estimator_matches_brute_force(tiny, dim_batch):
    model, _, ids = tiny
    n = len(get_layers(model))
    J = jlens.jacobian_for_prompt(model, ids, [0, 2], n - 1, dim_batch=dim_batch, skip_first=SKIP)
    for l in (0, 2):
        ref = brute_force_J(model, ids, l, n - 1)
        assert torch.allclose(J[l], ref, atol=1e-4, rtol=1e-3), (J[l] - ref).abs().max()


def test_fit_save_merge_and_lens(tiny, tmp_path):
    model, tok, _ = tiny
    prompts = (pytest.importorskip("tests.conftest").FIXTURES / "corpus.txt").read_text().splitlines()
    a = jlens.fit(model, tok, prompts[:3], str(tmp_path / "a.pt"), dim_batch=16, skip_first=SKIP)
    b = jlens.fit(model, tok, prompts[3:], str(tmp_path / "b.pt"), dim_batch=16, skip_first=SKIP)
    whole = jlens.fit(model, tok, prompts, str(tmp_path / "all.pt"), dim_batch=16, skip_first=SKIP)
    jlens.merge([a, b], str(tmp_path / "merged.pt"))
    Jm, nm = jlens.load_lens(str(tmp_path / "merged.pt"))
    Jw, nw = jlens.load_lens(whole)
    n_layers = len(get_layers(model))
    assert nm == nw == 6 and sorted(Jw) == list(range(n_layers - 1))
    for l in Jw:
        assert torch.allclose(Jm[l], Jw[l], atol=2e-3)  # fp16 storage
    assert not (tmp_path / "all.pt.ckpt").exists()

    lens, logit = JLens(model, whole), LogitLens(model)
    h = torch.randn(3, model.config.hidden_size)
    # final layer: identity transport, identical to the logit lens / model output
    assert torch.allclose(lens.logits(h, n_layers - 1), logit.logits(h, n_layers - 1))
    # earlier layers: logit lens applied to J_l h
    J0 = lens.maps[0][0]
    assert torch.allclose(lens.logits(h, 0), logit.logits(h @ J0.T, 0), atol=1e-5)
    assert torch.allclose(lens.direction(5, 0), J0.T @ logit.direction(5, 0))


def test_resume_from_checkpoint(tiny, tmp_path):
    model, tok, _ = tiny
    prompts = (pytest.importorskip("tests.conftest").FIXTURES / "corpus.txt").read_text().splitlines()
    out = str(tmp_path / "r.pt")
    ref = jlens.fit(model, tok, prompts, str(tmp_path / "ref.pt"), dim_batch=16, skip_first=SKIP)

    class Stop(Exception):
        pass

    orig = jlens.jacobian_for_prompt
    calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 5:
            raise Stop
        return orig(*a, **k)

    jlens.jacobian_for_prompt = flaky
    try:
        with pytest.raises(Stop):
            jlens.fit(model, tok, prompts, out, dim_batch=16, skip_first=SKIP, checkpoint_every=2)
    finally:
        jlens.jacobian_for_prompt = orig
    jlens.fit(model, tok, prompts, out, dim_batch=16, skip_first=SKIP, checkpoint_every=2)
    J1, n1 = jlens.load_lens(out)
    J2, n2 = jlens.load_lens(ref)
    assert n1 == n2 and all(torch.equal(J1[l], J2[l]) for l in J1)


def test_logit_lens_soft_cap(tiny):
    """A final-logit soft cap (Gemma 2 / 4) is applied like the model does, ranks unchanged."""
    model, _, ids = tiny
    with torch.no_grad():
        h = torch.randn(3, model.config.hidden_size) * 5
        plain = LogitLens(model).logits(h, 0).float()
        model.config.final_logit_softcapping = 2.0
        try:
            capped = LogitLens(model).logits(h, 0)
        finally:
            model.config.final_logit_softcapping = None
    assert capped.dtype == torch.float32 and capped.abs().max() <= 2.0
    assert torch.allclose(capped, 2.0 * torch.tanh(plain / 2.0), atol=1e-5)
    assert torch.equal(capped.argsort(-1, descending=True)[:, :20], plain.argsort(-1, descending=True)[:, :20])
