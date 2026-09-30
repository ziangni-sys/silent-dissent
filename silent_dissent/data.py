"""Multiple-choice datasets normalised to a single `MCQItem` format.

Every item is mapped onto letter labels ("A", "B", ...) so that the agent's
answer is always a single letter token at a known position.
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

LETTERS = "ABCDEFGH"


@dataclass
class MCQItem:
    item_id: str
    dataset: str
    question: str
    choices: list[str]
    gold: str  # letter
    subject: str = ""

    @property
    def letters(self) -> list[str]:
        return list(LETTERS[: len(self.choices)])

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "MCQItem":
        return cls(**{k: d[k] for k in ("item_id", "dataset", "question", "choices", "gold", "subject") if k in d})


def _load_mmlu(split: str, subset: str):
    from datasets import load_dataset

    ds = load_dataset("cais/mmlu", subset, split=split)
    for i, ex in enumerate(ds):
        yield MCQItem(
            item_id=f"mmlu-{ex.get('subject', subset)}-{i}",
            dataset="mmlu",
            question=ex["question"],
            choices=list(ex["choices"]),
            gold=LETTERS[int(ex["answer"])],
            subject=ex.get("subject", subset),
        )


def _load_arc(split: str, subset: str):
    from datasets import load_dataset

    ds = load_dataset("allenai/ai2_arc", subset, split=split)
    for ex in ds:
        labels = list(ex["choices"]["label"])
        if ex["answerKey"] not in labels:
            continue
        yield MCQItem(
            item_id=f"arc-{ex['id']}",
            dataset="arc",
            question=ex["question"],
            choices=list(ex["choices"]["text"]),
            # ARC sometimes uses "1".."4" as labels; re-letter by position.
            gold=LETTERS[labels.index(ex["answerKey"])],
        )


def _load_csqa(split: str, subset: str):
    from datasets import load_dataset

    ds = load_dataset("tau/commonsense_qa", split=split)
    for ex in ds:
        labels = list(ex["choices"]["label"])
        if not ex["answerKey"]:
            continue
        yield MCQItem(
            item_id=f"csqa-{ex['id']}",
            dataset="csqa",
            question=ex["question"],
            choices=list(ex["choices"]["text"]),
            gold=LETTERS[labels.index(ex["answerKey"])],
        )


def _load_jsonl(path: str):
    with open(path) as f:
        for line in f:
            if line.strip():
                yield MCQItem.from_dict(json.loads(line))


LOADERS = {"mmlu": _load_mmlu, "arc": _load_arc, "csqa": _load_csqa}


def load_items(
    name: str,
    split: str = "test",
    subset: str | None = None,
    n_choices: int | None = 4,
    limit: int | None = None,
    seed: int = 0,
) -> list[MCQItem]:
    """Load a dataset. `name` is a registered loader or a path to a .jsonl file.

    `n_choices` filters to items with exactly that many options (default 4, so
    that all conditions share the same chance level); None keeps everything.
    `limit` randomly subsamples (deterministically for a given seed).
    """
    if name.endswith(".jsonl"):
        items = list(_load_jsonl(name))
    else:
        default_subset = {"mmlu": "all", "arc": "ARC-Challenge", "csqa": None}[name]
        items = list(LOADERS[name](split, subset or default_subset))
    if n_choices is not None:
        items = [it for it in items if len(it.choices) == n_choices]
    if limit is not None and limit < len(items):
        items = random.Random(seed).sample(items, limit)
    return items


def save_items(items: list[MCQItem], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for it in items:
            f.write(json.dumps(it.to_dict(), ensure_ascii=False) + "\n")
