"""Structure of the TwoHopFact categories the study uses (public data only; no model, no results).

For each category: the share of facts whose answer e3 belongs to more than one bridge e2 (then the
peers' answer does not determine their bridge), the share whose bridge shares a content word with
the question (the bridge is partly stated), and the share whose bridge shares a content word with the
answer (such tokens are removed from the probes; with all words shared, the bridge probe is empty).
Content words follow silent_dissent.entity_facts.probe_tokens (>= 3 letters, STOPWORDS removed).

    python scripts/twohopfact_audit.py [--csv TwoHopFact.csv]   # downloads the CSV from the Hub if omitted
"""
import argparse
import re
from collections import defaultdict

from silent_dissent.entity_facts import STOPWORDS, handmade_facts, load_twohopfact

CATEGORIES = ["landmark-cntry-capital", "person-birthcity-cntry", "univ-hqcity-cntry",
              "person-uguniv-hqcity", "person-uguniv-hqcntry"]

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default=None)
args = ap.parse_args()


def words(name: str) -> set[str]:
    return {w.lower() for w in re.findall(r"[^\W_]+(?:['’][^\W_]+)?", name)
            if w.lower() not in STOPWORDS and len(w) >= 3}


facts = load_twohopfact(args.csv) + handmade_facts()
print("category                     facts  answer->many bridges  bridge word in question  shares word with answer (all words)")
for c in CATEGORIES:
    fs = [f for f in facts if f.category == c]
    bridges = defaultdict(set)
    for f in fs:
        bridges[f.e3].add(f.e2)
    share = lambda cond: sum(1 for f in fs if cond(f)) / len(fs)
    multi = share(lambda f: len(bridges[f.e3]) > 1)
    in_q = share(lambda f: bool(words(f.e2) & words(f.composed)))
    any_a = share(lambda f: bool(words(f.e2) & words(f.e3)))
    all_a = share(lambda f: bool(words(f.e2)) and words(f.e2) <= words(f.e3))
    print(f"{c:26s} {len(fs):7d}  {multi:20.0%}  {in_q:23.0%}  {any_a:12.0%} ({all_a:.0%})")
