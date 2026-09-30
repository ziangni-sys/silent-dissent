"""Exploratory: is the J-lens better than the logit lens outside the four-letter readout?

Not part of the pre-registered study; no pressure data are read.

1. Text fidelity. Per layer, KL(model output || lens), top-1 agreement with the model's own
   prediction and next-token cross-entropy, on 128-token windows from
     wikitext   wikitext-103 *test* paragraphs (the reference J-lens was fit on the train split)
     exam       the MMLU / ARC items of this run as plain text (out of domain)
     chat       the rendered round-0 task prompts, chat template included (out of domain)
2. Latent two-hop recall. Few-shot prompts such as "The capital city of the country where
   the Eiffel Tower is located is" -> Paris; the bridge entity (France) is never stated.
   Among prompts the model answers correctly, per layer: fraction with the bridge's first
   token in the lens top-10, minus the same fraction for an unrelated control country, at
   the subject's last token and at the last prompt token.

Comparisons fixed before running: per layer, which lens has lower KL / higher top-1
agreement; per template and position, the peak of bridge@10 - control@10 over layers and
the first layer where it reaches 0.2.

    python scripts/lens_benchmark.py --config configs/qwen35_4b.yaml
Writes <out_dir>/lens_benchmark/{text_fidelity.csv, two_hop_items.jsonl, two_hop_ranks.npz, two_hop_summary.csv}.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from silent_dissent.config import get_items, load_config, setup
from silent_dissent.experiments import read_jsonl
from silent_dissent.lens_eval import make_two_hop_item, text_fidelity, two_hop_ranks
from silent_dissent.lenses import JLens, LogitLens
from silent_dissent.prompts import format_question, render, solo_messages

# country (as its bridge word), capital, currency, official language (None where there is no single one)
COUNTRIES = [
    ("France", "Paris", "euro", "French"), ("Germany", "Berlin", "euro", "German"),
    ("Italy", "Rome", "euro", "Italian"), ("Spain", "Madrid", "euro", "Spanish"),
    ("Portugal", "Lisbon", "euro", "Portuguese"), ("Greece", "Athens", "euro", "Greek"),
    ("Netherlands", "Amsterdam", "euro", "Dutch"), ("Austria", "Vienna", "euro", "German"),
    ("Finland", "Helsinki", "euro", "Finnish"), ("Ireland", "Dublin", "euro", None),
    ("Belgium", "Brussels", "euro", None), ("Japan", "Tokyo", "yen", "Japanese"),
    ("China", "Beijing", "yuan", "Chinese"), ("Korea", "Seoul", "won", "Korean"),
    ("India", "New Delhi", "rupee", "Hindi"), ("Russia", "Moscow", "ruble", "Russian"),
    ("Brazil", "Brasília", "real", "Portuguese"), ("Argentina", "Buenos Aires", "peso", "Spanish"),
    ("Chile", "Santiago", "peso", "Spanish"), ("Colombia", "Bogotá", "peso", "Spanish"),
    ("Peru", "Lima", "sol", "Spanish"), ("Egypt", "Cairo", "pound", "Arabic"),
    ("Turkey", "Ankara", "lira", "Turkish"), ("Sweden", "Stockholm", "krona", "Swedish"),
    ("Norway", "Oslo", "krone", "Norwegian"), ("Denmark", "Copenhagen", "krone", "Danish"),
    ("Poland", "Warsaw", "zloty", "Polish"), ("Switzerland", "Bern", "franc", None),
    ("Thailand", "Bangkok", "baht", "Thai"), ("Vietnam", "Hanoi", "dong", "Vietnamese"),
    ("Indonesia", "Jakarta", "rupiah", "Indonesian"), ("Australia", "Canberra", "dollar", "English"),
    ("Canada", "Ottawa", "dollar", None), ("Nigeria", "Abuja", "naira", "English"),
    ("Israel", "Jerusalem", "shekel", "Hebrew"), ("Iran", "Tehran", "rial", "Persian"),
    ("Hungary", "Budapest", "forint", "Hungarian"), ("Czech", "Prague", "koruna", None),
    ("Ukraine", "Kyiv", "hryvnia", "Ukrainian"), ("Pakistan", "Islamabad", "rupee", "Urdu"),
    ("Bangladesh", "Dhaka", "taka", "Bengali"), ("Saudi", "Riyadh", "riyal", "Arabic"),
    ("Morocco", "Rabat", "dirham", "Arabic"), ("Cuba", "Havana", "peso", "Spanish"),
    ("Romania", "Bucharest", "leu", "Romanian"), ("Malaysia", "Kuala Lumpur", "ringgit", "Malay"),
    ("Iceland", "Reykjavik", "krona", "Icelandic"), ("Nepal", "Kathmandu", "rupee", "Nepali"),
    ("Ethiopia", "Addis Ababa", "birr", "Amharic"), ("Mongolia", "Ulaanbaatar", "tugrik", "Mongolian"),
    ("Iraq", "Baghdad", "dinar", "Arabic"), ("Serbia", "Belgrade", "dinar", "Serbian"),
    ("Cambodia", "Phnom Penh", "riel", "Khmer"), ("Jordan", "Amman", "dinar", "Arabic"),
    ("Kenya", "Nairobi", "shilling", None), ("Philippines", "Manila", "peso", None),
    ("Ecuador", "Quito", "dollar", "Spanish"),
]
LANDMARKS = [
    ("the Eiffel Tower", "France"), ("the Louvre", "France"), ("the Colosseum", "Italy"),
    ("the Leaning Tower of Pisa", "Italy"), ("the Uffizi Gallery", "Italy"), ("the Taj Mahal", "India"),
    ("the Golden Temple", "India"), ("the Forbidden City", "China"), ("the Temple of Heaven", "China"),
    ("Tiananmen Square", "China"), ("the Kremlin", "Russia"), ("Red Square", "Russia"),
    ("the Hermitage Museum", "Russia"), ("the Sydney Opera House", "Australia"), ("Uluru", "Australia"),
    ("the Christ the Redeemer statue", "Brazil"), ("Machu Picchu", "Peru"), ("the Pyramids of Giza", "Egypt"),
    ("the Great Sphinx", "Egypt"), ("Mount Fuji", "Japan"), ("the Acropolis", "Greece"),
    ("the Parthenon", "Greece"), ("the Sagrada Familia", "Spain"), ("the Alhambra", "Spain"),
    ("the Prado Museum", "Spain"), ("Angkor Wat", "Cambodia"), ("the Brandenburg Gate", "Germany"),
    ("Neuschwanstein Castle", "Germany"), ("Petra", "Jordan"), ("the Hagia Sophia", "Turkey"),
    ("the Blue Mosque", "Turkey"), ("the CN Tower", "Canada"), ("the Matterhorn", "Switzerland"),
    ("the Little Mermaid statue", "Denmark"), ("the Rijksmuseum", "Netherlands"),
    ("Schönbrunn Palace", "Austria"), ("the Petronas Towers", "Malaysia"), ("Borobudur", "Indonesia"),
    ("Ha Long Bay", "Vietnam"), ("the Moai statues of Easter Island", "Chile"),
    ("the Galápagos Islands", "Ecuador"), ("Bran Castle", "Romania"), ("the Blue Lagoon", "Iceland"),
    ("Hallgrímskirkja", "Iceland"), ("Wawel Castle", "Poland"), ("Charles Bridge", "Czech"),
    ("Wat Arun", "Thailand"), ("the Atomium", "Belgium"),
]
# template: (few-shot lines, prefix, suffix); the few-shot facts are not in the item set
TEMPLATES = {
    "capital->currency": (
        "The currency used in the country whose capital city is Washington is called the dollar.\n"
        "The currency used in the country whose capital city is London is called the pound.\n",
        "The currency used in the country whose capital city is ", " is called the"),
    "capital->language": (
        "The official language of the country whose capital city is London is English.\n"
        "The official language of the country whose capital city is Mexico City is Spanish.\n",
        "The official language of the country whose capital city is ", " is"),
    "landmark->capital": (
        "The capital city of the country where the Statue of Liberty is located is Washington.\n"
        "The capital city of the country where Big Ben is located is London.\n",
        "The capital city of the country where ", " is located is"),
}


def two_hop_items(tok, seed: int):
    by_country = {c[0]: c for c in COUNTRIES}
    rows = []  # (template, subject, bridge, answer)
    for country, capital, currency, language in COUNTRIES:
        rows.append(("capital->currency", capital, country, currency))
        if language:
            rows.append(("capital->language", capital, country, language))
    for landmark, country in LANDMARKS:
        rows.append(("landmark->capital", landmark, country, by_country[country][1]))
    items = []
    for template, subject, bridge, answer in rows:
        shots, prefix, suffix = TEMPLATES[template]
        rng = random.Random(f"{seed}-{template}-{subject}")
        for control in rng.sample([c[0] for c in COUNTRIES if c[0] != bridge], len(COUNTRIES) - 1):
            it = make_two_hop_item(tok, template, shots + prefix, subject, suffix, bridge, answer, control)
            if it is not None:
                items.append(it)
                break
    return items


def token_windows(tok, docs: list[str], n: int, seq_len: int) -> torch.Tensor:
    """Concatenate documents (blank line between) and cut n non-overlapping windows."""
    stream: list[int] = []
    for d in docs:
        stream += tok(d + "\n\n", add_special_tokens=False)["input_ids"]
        if len(stream) >= n * seq_len:
            break
    k = min(n, len(stream) // seq_len)
    return torch.tensor(stream[: k * seq_len]).view(k, seq_len)


ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--n-windows", type=int, default=200)
ap.add_argument("--seq-len", type=int, default=128)
ap.add_argument("--batch-size", type=int, default=4)
ap.add_argument("--out", help="output directory (default: <out_dir>/lens_benchmark)")
ap.add_argument("--extra-lens", action="append", default=[], metavar="NAME=PATH",
                help="another fitted J-lens to compare, e.g. jlens_chat=lenses/qwen35_4b_jlens_chat.pt")
args = ap.parse_args()

cfg = load_config(args.config)
lm, lens = setup(cfg)
lenses = {cfg["lens"]["name"]: lens, "logit": LogitLens(lm.model)}
for spec in args.extra_lens:
    name, path = spec.split("=", 1)
    lenses[name] = JLens(lm.model, path=path)
out = Path(args.out) if args.out else cfg["out_dir"] / "lens_benchmark"
out.mkdir(parents=True, exist_ok=True)
rng = random.Random(cfg["seed"])

# ---- 1. text fidelity
from datasets import load_dataset  # noqa: E402

wiki = [t for t in load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")["text"]
        if len(t) >= 600 and not t.lstrip().startswith("=")]
rng.shuffle(wiki)
items = get_items(cfg)
exam = [format_question(it) for it in rng.sample(items, min(3000, len(items)))]
chat = [render(lm.tok, solo_messages(it)) for it in rng.sample(items, min(3000, len(items)))]
rows = []
for corpus, docs in (("wikitext", wiki), ("exam", exam), ("chat", chat)):
    win = token_windows(lm.tok, docs, args.n_windows, args.seq_len)
    print(f"[text] {corpus}: {len(win)} windows x {args.seq_len} tokens")
    res = text_fidelity(lm.model, lenses, win, args.batch_size)
    for name, metrics in res.items():
        for layer in range(len(metrics["kl"])):
            rows.append({"corpus": corpus, "lens": name, "layer": layer, **{k: float(v[layer]) for k, v in metrics.items()}})
tf = pd.DataFrame(rows)
tf.to_csv(out / "text_fidelity.csv", index=False)

# ---- 2. latent two-hop recall
th = two_hop_items(lm.tok, cfg["seed"])
print(f"[two-hop] {len(th)} prompts")
top1, ranks = two_hop_ranks(lm, lenses, th, batch_size=16)
correct = np.array([t == it.ids[1] for t, it in zip(top1, th)])
with open(out / "two_hop_items.jsonl", "w") as f:
    for it, c in zip(th, correct):
        f.write(json.dumps({**it.__dict__, "correct": bool(c)}, ensure_ascii=False) + "\n")
np.savez_compressed(out / "two_hop_ranks.npz", **ranks)
summ = []
templates = np.array([it.template for it in th])
for name, r in ranks.items():
    for template in list(TEMPLATES) + ["all"]:
        mask = correct & ((templates == template) if template != "all" else True)
        for k, pos in enumerate(("subject", "last")):
            if not mask.any():
                continue
            bridge10 = (r[mask, k, :, 0] < 10).mean(0)
            control10 = (r[mask, k, :, 2] < 10).mean(0)
            answer1 = (r[mask, k, :, 1] == 0).mean(0)
            for layer in range(r.shape[2]):
                summ.append({"lens": name, "template": template, "position": pos, "layer": layer, "n": int(mask.sum()),
                             "bridge_top10": bridge10[layer], "control_top10": control10[layer],
                             "bridge_minus_control": bridge10[layer] - control10[layer], "answer_top1": answer1[layer],
                             "bridge_median_rank": float(np.median(r[mask, k, layer, 0]))})
sm = pd.DataFrame(summ)
sm.to_csv(out / "two_hop_summary.csv", index=False)

# ---- report
print("\n=== text fidelity (J = " + cfg["lens"]["name"] + ", L = logit)")
for corpus in tf.corpus.unique():
    t = tf[tf.corpus == corpus].pivot(index="layer", columns="lens")
    j, l = cfg["lens"]["name"], "logit"
    print(f"\n-- {corpus}\nlayer   KL_J   KL_L | top1_J top1_L | top5_J top5_L |  CE_J   CE_L")
    for layer, row in t.iterrows():
        print(f"{layer:5d} {row[('kl', j)]:6.2f} {row[('kl', l)]:6.2f} | {row[('top1_agree', j)]:6.3f} {row[('top1_agree', l)]:6.3f} |"
              f" {row[('top5_contains', j)]:6.3f} {row[('top5_contains', l)]:6.3f} | {row[('ce', j)]:6.2f} {row[('ce', l)]:6.2f}")
    better = t[("kl", j)] < t[("kl", l)]
    print(f"J-lens lower KL at {int(better.sum())}/{len(t)} layers: {list(t.index[better])}")

mid = tf[tf.layer.between(8, 24)].groupby(["corpus", "lens"])[["kl", "top1_agree"]].mean().unstack("lens")
print("\n=== mean over layers 8-24, every lens (KL lower / top-1 higher is better)")
print(mid.round(3).to_string())

print(f"\n=== two-hop: {int(correct.sum())}/{len(th)} answered correctly;"
      f" per template: { {tp: int((correct & (templates == tp)).sum()) for tp in TEMPLATES} }")
for template in list(TEMPLATES) + ["all"]:
    for pos in ("subject", "last"):
        print(f"\n-- {template} @ {pos}: bridge@10 - control@10 by layer (layers 8..31)")
        for name in ranks:
            s = sm[(sm.lens == name) & (sm.template == template) & (sm.position == pos)].set_index("layer")
            if s.empty:
                continue
            d = s["bridge_minus_control"]
            first = next((l for l, v in d.items() if v >= 0.2), None)
            print(f"{name:6s} n={s.n.iloc[0]:3d} " + " ".join(f"{d[l]:5.2f}" for l in range(8, len(d)))
                  + f" | peak {d.max():.2f} @ L{int(d.idxmax())}, first >= 0.2: {first}")
