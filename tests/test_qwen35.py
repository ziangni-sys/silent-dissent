"""Qwen3.5-specific behaviour on a tiny random model: (1 + w) final norm, hybrid
linear attention under left padding, the J-lens, and the thinking-mode template."""
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "Qwen3_5TextConfig"):
    pytest.skip("transformers without Qwen3.5", allow_module_level=True)

from silent_dissent import jlens
from silent_dissent.lenses import JLens, LogitLens, _norm_gain
from silent_dissent.model import LM, get_final_norm
from silent_dissent.prompts import ANSWER_PREFIX, render, solo_messages

# Once causal-conv1d / fla are installed, transformers binds Qwen3.5's linear attention to
# those CUDA-only kernels at import time, whatever the tensor's device. So on a GPU box the
# tiny model must run on CUDA (which also tests the real kernel path); without them, CPU is fine.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def items():
    from silent_dissent.data import load_items
    from tests.conftest import FIXTURES

    return load_items(str(FIXTURES / "items.jsonl"))


@pytest.fixture(scope="module")
def q35(items):
    from silent_dissent.debug import make_tokenizer

    tok = make_tokenizer(items)
    torch.manual_seed(0)
    cfg = transformers.Qwen3_5TextConfig(
        vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16, linear_key_head_dim=16, linear_value_head_dim=16,
        linear_num_key_heads=2, linear_num_value_heads=4,
        layer_types=["linear_attention"] * 3 + ["full_attention"], tie_word_embeddings=True,
        pad_token_id=tok.pad_token_id)
    model = transformers.AutoModelForCausalLM.from_config(cfg).to(DEVICE).eval()
    with torch.no_grad():  # zero-initialised norms would hide the (1 + w) gain
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.3)
    return LM(model, tok, list("ABCD"), ANSWER_PREFIX)


def test_norm_gain_is_one_plus_w(q35):
    norm = get_final_norm(q35.model)
    assert torch.allclose(_norm_gain(norm), 1.0 + norm.weight.float(), atol=1e-5)


def test_logit_direction_is_lens_gradient(q35):
    """For a residual with rms 1, the lens-logit gradient equals the direction."""
    lens = LogitLens(q35.model)
    d = q35.model.config.hidden_size
    h = torch.ones(1, d, device=DEVICE, requires_grad=True)
    lens.logits(h, 0)[0, 7].backward()
    # d/dh of w_t . (g * h / rms(h)) at h = 1 is g*w_t minus a component along h
    g, u = h.grad[0], lens.direction(7, 0)
    proj = lambda v: v - v.mean() * torch.ones(d, device=DEVICE)
    assert torch.allclose(proj(g), proj(u), atol=1e-4)


def test_left_padding_invariance(q35, items):
    lens = LogitLens(q35.model)
    texts = [render(q35.tok, solo_messages(it)) for it in items[:2]]
    texts[1] += " A B C D E ." * 4 + " " + ANSWER_PREFIX  # force different lengths
    assert len(set(len(q35.tok(t).input_ids) for t in texts)) == 2
    alone = [q35.readout([t], lens) for t in texts]
    both = q35.readout(texts, lens)
    for i, a in enumerate(alone):
        assert torch.allclose(a.final_letter_logits[0], both.final_letter_logits[i], atol=1e-4)
        assert torch.allclose(a.lens_letter_logits[0], both.lens_letter_logits[i], atol=1e-4)


def test_jlens_on_hybrid_model(q35, items, tmp_path):
    from tests.conftest import FIXTURES

    prompts = (FIXTURES / "corpus.txt").read_text().splitlines()
    path = jlens.fit(q35.model, q35.tok, prompts[:2], str(tmp_path / "q.pt"), dim_batch=32, skip_first=2)
    lens = JLens(q35.model, path)
    ro = q35.readout([render(q35.tok, solo_messages(items[0]))], lens)
    assert torch.allclose(ro.lens_letter_logits[0, -1], ro.final_letter_logits[0], atol=1e-4)


def test_template_kwargs_reach_the_template(items):
    from silent_dissent.debug import make_tokenizer

    tok = make_tokenizer(items)
    tok.chat_template = ("{% for m in messages %}{{ m['content'] }}\n{% endfor %}"
                         "{% if enable_thinking is defined and enable_thinking is false %}<think></think>"
                         "{% else %}<think>{% endif %}")
    msgs = solo_messages(items[0])
    assert render(tok, msgs).endswith("<think>" + ANSWER_PREFIX)
    tok.sd_template_kwargs = {"enable_thinking": False}
    assert render(tok, msgs).endswith("<think></think>" + ANSWER_PREFIX)
