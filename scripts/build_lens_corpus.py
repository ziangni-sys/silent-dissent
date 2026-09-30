"""E6: corpora for refitting the J-lens outside wikitext.

The reference J-lens is fit on wikitext-103 and helped least on chat-formatted text. This writes
fitting corpora ({"text": ...} per line) for scripts/fit_jlens.py:

  chat    conversations from HuggingFaceH4/ultrachat_200k (split test_sft), rendered with the
          model's chat template (the format the debate prompts use)
  mixed   the chat documents interleaved with wikitext-103 *train* paragraphs, half and half

None of this text contains the study's questions (TwoHopFact / hand-made facts are not used).

    python scripts/build_lens_corpus.py --config configs/qwen35_4b_jlens_chat.yaml --n 2000
    python scripts/fit_jlens.py --config configs/qwen35_4b_jlens_chat.yaml
"""
import argparse
import json
import random
from pathlib import Path

from silent_dissent.config import load_config

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True, help="a lens-fitting config (its model's chat template is used)")
ap.add_argument("--n", type=int, default=2000, help="documents per corpus")
ap.add_argument("--out", default=None, help="directory (default: the parent of jlens.corpus)")
args = ap.parse_args()

cfg = load_config(args.config)
from datasets import load_dataset  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(cfg["model"]["name"])
kw = cfg["model"].get("chat_template_kwargs") or {}
rng = random.Random(cfg["seed"])
out = Path(args.out or Path(cfg["jlens"]["corpus"]).parent)
out.mkdir(parents=True, exist_ok=True)

import fnmatch  # noqa: E402

import pandas as pd  # noqa: E402
from huggingface_hub import hf_hub_download, list_repo_files  # noqa: E402

# only the test_sft file (81 MB); load_dataset with data_files still verifies every split and fails
repo = "HuggingFaceH4/ultrachat_200k"
name = next(f for f in list_repo_files(repo, repo_type="dataset") if fnmatch.fnmatch(f, "data/test_sft-*.parquet"))
chat_df = pd.read_parquet(hf_hub_download(repo, name, repo_type="dataset"))
idx = rng.sample(range(len(chat_df)), min(args.n, len(chat_df)))
chat = []
for i in idx:
    msgs = [{"role": m["role"], "content": m["content"]} for m in chat_df.iloc[i]["messages"]]
    chat.append(tok.apply_chat_template(msgs, tokenize=False, **kw))
wiki = [t for t in load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")["text"]
        if len(t) >= 600 and not t.lstrip().startswith("=")]
rng.shuffle(wiki)
mixed = [d for pair in zip(chat[: args.n // 2], wiki[: args.n // 2]) for d in pair]

for name, docs in (("chat", chat), ("mixed", mixed)):
    with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps({"text": d}, ensure_ascii=False) + "\n")
    print(f"{name}: {len(docs)} documents -> {out / f'{name}.jsonl'}")
