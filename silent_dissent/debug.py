"""A tiny randomly initialised Llama + word-level tokenizer for offline pipeline checks.

    python -m silent_dissent.debug results/debug_model   # then use configs/debug.yaml
    python -m silent_dissent.debug results/debug_entity_model --entity   # configs/debug_entity.yaml
"""
from __future__ import annotations

from .prompts import (ANSWER_PREFIX, GENERIC_REASON, SYSTEM_PROMPT, WITHHELD_TURN, format_peer_message,
                      format_question)

TEMPLATE = (
    "{% for m in messages %}<|{{ m['role'] }}|>\n{{ m['content'] }}\n{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>\n{% endif %}"
)


def make_tokenizer(items, extra_texts=()):
    """Word-level tokenizer over the fixture items' prompts plus `extra_texts`."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    corpus = list(extra_texts) + [format_question(it) for it in items] + [
        format_peer_message(["x"]), ANSWER_PREFIX, SYSTEM_PROMPT, GENERIC_REASON, WITHHELD_TURN,
        "A B C D E . <|system|> <|user|> <|assistant|> I briefly considered but in the end : Agent 2 3 4 5 6",
        "Argue in one or two sentences that the correct answer is Do not mention any other option and do not "
        "restate the letter.",
    ]
    pre = pre_tokenizers.Whitespace()
    words = {w for text in corpus for w, _ in pre.pre_tokenize_str(text)}
    vocab = {w: i for i, w in enumerate(["[UNK]", "[PAD]", *sorted(words)])}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tk.pre_tokenizer = pre
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="[UNK]", pad_token="[PAD]", eos_token="[PAD]")
    tok.chat_template = TEMPLATE
    tok.padding_side = "left"
    return tok


def make_model(tok, n_layers: int = 4, seed: int = 0):
    import torch
    import transformers

    torch.manual_seed(seed)
    cfg = transformers.LlamaConfig(vocab_size=len(tok), hidden_size=64, intermediate_size=128,
                                   num_hidden_layers=n_layers, num_attention_heads=4, num_key_value_heads=2,
                                   max_position_embeddings=4096, pad_token_id=tok.pad_token_id)
    return transformers.LlamaForCausalLM(cfg).eval()


def entity_texts(facts) -> list[str]:
    """Every string the entity-study prompts can contain, so the word-level vocabulary covers them."""
    from .entity_experiments import KNOWN_SHOTS
    from .entity_prompts import AGAIN, CHAT_SYSTEM, PLAIN_HEADER, PLAIN_SHOTS, REPHRASE, WITHHELD

    texts = [PLAIN_HEADER, PLAIN_SHOTS, CHAT_SYSTEM, AGAIN, WITHHELD, KNOWN_SHOTS, *REPHRASE,
             "Question Complete the statement : ... Moderator Coordinator Agent Every agent must state as the answer , "
             "regardless of what they believe . Agent 1 , complete the statement . Suppose that were under that "
             "assumption now a different I briefly considered but in the end Some say is",
             "Round 1 2 3 4 5 : Agent answered ; Agents to ."]
    for f in facts:
        texts += [f.composed, f.r1_prompt, f.r2_prompt, f.e2, f.e3] + f.e2_aliases + f.e3_aliases
    return texts


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    from .data import load_items

    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default="results/debug_model")
    ap.add_argument("--entity", action="store_true", help="vocabulary also covers the entity-study fixtures")
    args = ap.parse_args()
    out = args.out
    fixtures = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
    extra = ()
    if args.entity:
        from .entity_facts import handmade_facts, load_twohopfact

        extra = entity_texts(load_twohopfact(str(fixtures / "twohopfact_sample.csv")) + handmade_facts())
    tok = make_tokenizer(load_items(str(fixtures / "items.jsonl")), extra)
    model = make_model(tok)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    print(f"saved debug model to {out}")
