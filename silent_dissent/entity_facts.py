"""Two-hop facts for the entity-answer study.

A fact is e1 -r1-> e2 -r2-> e3, e.g. Nineteen Eighty-Four -author-> George Orwell
-birthcity-> Motihari. Prompts complete the composed statement ("The author of the novel
Nineteen Eighty-Four was born in the city of"), so the bridge e2 is never written unless a
condition writes it. Sources: TwoHopFact (Yang et al. 2024, CC-BY-4.0, 45,595 facts over
52 relation compositions) and a small hand-made set of capital questions.

Every fact gets, from the same category (same r1 and r2):
  peer      another fact whose (e2', e3') is the world the peers argue for: they push e3',
            which implies e2'; both are true facts about another entity
  control   a third fact, the baseline for any entity readout
and, where TwoHopFact has one, a `sibling`: the same e1 and r1 with another r2 (retention
control: the agent must use e2 again while saying something else).
"""
from __future__ import annotations

import ast
import csv
import hashlib
import random
import re
import unicodedata
from dataclasses import asdict, dataclass, field

# ----------------------------------------------------------------------------- facts


@dataclass
class Fact:
    uid: str
    source: str
    category: str
    e1: str
    e2: str
    e3: str
    r1_prompt: str        # "The author of the novel X is"            (-> e2)
    r2_template: str      # "{} was born in the city of"              (-> e3)
    composed: str         # "The author of the novel X was born in the city of"  (-> e3)
    e2_aliases: list[str] = field(default_factory=list)
    e3_aliases: list[str] = field(default_factory=list)
    e2_type: str = ""
    e3_type: str = ""

    @property
    def r2_prompt(self) -> str:
        return self.r2_template.format(self.e2)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Fact":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


def _aliases(raw: str, value: str) -> list[str]:
    out = [value]
    try:
        parsed = ast.literal_eval(raw) if raw else ()
    except (ValueError, SyntaxError):
        parsed = ()
    stack = list(parsed) if isinstance(parsed, (tuple, list)) else [parsed]
    while stack:
        x = stack.pop(0)
        if isinstance(x, (tuple, list)):
            stack = list(x) + stack
        elif isinstance(x, str) and x not in out:
            out.append(x)
    return out


TWOHOPFACT = ("soheeyang/TwoHopFact", "TwoHopFact.csv")


def load_twohopfact(path: str | None = None) -> list[Fact]:
    """TwoHopFact rows as Facts (downloads the CSV from the Hub unless `path` is given)."""
    if path is None:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(TWOHOPFACT[0], TWOHOPFACT[1], repo_type="dataset")
    csv.field_size_limit(10**8)
    facts = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            facts.append(Fact(
                uid=f"thf:{r['uid']}", source="twohopfact", category=r["category"],
                e1=r["e1.value"], e2=r["e2.value"], e3=r["e3.value"],
                r1_prompt=r["r1(e1).prompt"], r2_template=r["r2.template"], composed=r["r2(r1(e1)).prompt"],
                e2_aliases=_aliases(r["e2.aliases"], r["e2.value"]), e3_aliases=_aliases(r["e3.aliases"], r["e3.value"]),
                e2_type=r["e2.rough_category"], e3_type=r["e3.rough_category"]))
    return facts


# hand-made capital questions: the bridge is a country, the answer its capital
CAPITALS = {
    "France": "Paris", "Germany": "Berlin", "Italy": "Rome", "Spain": "Madrid", "Portugal": "Lisbon",
    "Greece": "Athens", "Netherlands": "Amsterdam", "Belgium": "Brussels", "Austria": "Vienna",
    "Switzerland": "Bern", "Denmark": "Copenhagen", "Sweden": "Stockholm", "Norway": "Oslo",
    "Finland": "Helsinki", "Iceland": "Reykjavik", "Poland": "Warsaw", "Czech Republic": "Prague",
    "Hungary": "Budapest", "Romania": "Bucharest", "Russia": "Moscow", "Turkey": "Ankara", "Jordan": "Amman",
    "Egypt": "Cairo", "Kenya": "Nairobi", "Nigeria": "Abuja", "India": "New Delhi", "Iran": "Tehran",
    "China": "Beijing", "South Korea": "Seoul", "Japan": "Tokyo", "Vietnam": "Hanoi", "Cambodia": "Phnom Penh",
    "Thailand": "Bangkok", "Malaysia": "Kuala Lumpur", "Indonesia": "Jakarta", "Australia": "Canberra",
    "Canada": "Ottawa", "Brazil": "Brasília", "Argentina": "Buenos Aires", "Chile": "Santiago", "Peru": "Lima",
    "Colombia": "Bogotá", "Ecuador": "Quito",
}
ARTICLE = {"Netherlands": "the Netherlands", "Czech Republic": "the Czech Republic"}
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
    ("Hallgrímskirkja", "Iceland"), ("Wawel Castle", "Poland"), ("Charles Bridge", "Czech Republic"),
    ("Wat Arun", "Thailand"), ("the Atomium", "Belgium"),
]
PEOPLE = [  # birthplace in today's country, undisputed
    ("Wolfgang Amadeus Mozart", "Austria"), ("Ludwig van Beethoven", "Germany"), ("Pablo Picasso", "Spain"),
    ("Frédéric Chopin", "Poland"), ("Leo Tolstoy", "Russia"), ("Dante Alighieri", "Italy"),
    ("Mahatma Gandhi", "India"), ("Confucius", "China"), ("Pelé", "Brazil"), ("Gabriel García Márquez", "Colombia"),
    ("Pablo Neruda", "Chile"), ("Hans Christian Andersen", "Denmark"), ("Vincent van Gogh", "Netherlands"),
    ("Leonardo da Vinci", "Italy"), ("Albert Einstein", "Germany"), ("Marie Curie", "Poland"),
    ("Jean Sibelius", "Finland"), ("Edvard Grieg", "Norway"), ("Henrik Ibsen", "Norway"),
    ("Astrid Lindgren", "Sweden"), ("Alfred Nobel", "Sweden"), ("Fyodor Dostoevsky", "Russia"),
    ("Miguel de Cervantes", "Spain"), ("Victor Hugo", "France"), ("Katsushika Hokusai", "Japan"),
    ("Akira Kurosawa", "Japan"), ("Rabindranath Tagore", "India"), ("Omar Khayyam", "Iran"),
    ("Naguib Mahfouz", "Egypt"), ("Jorge Luis Borges", "Argentina"), ("Diego Maradona", "Argentina"),
    ("Ferenc Puskás", "Hungary"), ("Antonín Dvořák", "Czech Republic"), ("Franz Kafka", "Czech Republic"),
    ("Nicolaus Copernicus", "Poland"), ("Ho Chi Minh", "Vietnam"), ("Salvador Dalí", "Spain"),
    ("Rembrandt", "Netherlands"), ("Hergé", "Belgium"), ("Björk", "Iceland"), ("Haruki Murakami", "Japan"),
    ("Paulo Coelho", "Brazil"), ("Mario Vargas Llosa", "Peru"), ("Chinua Achebe", "Nigeria"),
    ("Wangari Maathai", "Kenya"),
]
COMPANIES = [
    ("Toyota", "Japan"), ("Sony", "Japan"), ("Samsung", "South Korea"), ("Nokia", "Finland"),
    ("Volkswagen", "Germany"), ("Siemens", "Germany"), ("Nestlé", "Switzerland"), ("Rolex", "Switzerland"),
    ("Philips", "Netherlands"), ("Heineken", "Netherlands"), ("LEGO", "Denmark"), ("Ferrari", "Italy"),
    ("Huawei", "China"), ("Alibaba", "China"), ("Infosys", "India"), ("Petrobras", "Brazil"), ("Gazprom", "Russia"),
    ("Spotify", "Sweden"), ("Renault", "France"), ("L'Oréal", "France"), ("Zara", "Spain"),
    ("Škoda", "Czech Republic"),
]
HANDMADE = {  # category: (facts, r1 phrase with {s}, type of e1)
    "landmark-cntry-capital": (LANDMARKS, "the country where {s} is located"),
    "person-birthcntry-capital": (PEOPLE, "the country where {s} was born"),
    "company-hqcntry-capital": (COMPANIES, "the country where {s} is headquartered"),
}


def handmade_facts() -> list[Fact]:
    facts = []
    for category, (rows, phrase) in HANDMADE.items():
        for subject, country in rows:
            mu = phrase.format(s=subject)
            e2 = ARTICLE.get(country, country)
            facts.append(Fact(
                uid=f"hand:{category}:{subject}", source="handmade", category=category, e1=subject, e2=e2,
                e3=CAPITALS[country], r1_prompt=f"{mu[0].upper()}{mu[1:]} is",
                r2_template="The capital city of {} is", composed=f"The capital city of {mu} is",
                e2_aliases=list(dict.fromkeys([e2, country])), e3_aliases=[CAPITALS[country]],
                e2_type="country", e3_type="city"))
    return facts


def load_facts(sources: list[dict]) -> list[Fact]:
    facts = []
    for s in sources:
        if s["name"] == "twohopfact":
            facts += load_twohopfact(s.get("path"))
        elif s["name"] == "handmade":
            facts += handmade_facts()
        else:
            raise ValueError(f"unknown fact source {s['name']!r}")
    return facts


# ------------------------------------------------------------------ answer matching


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return " ".join(re.sub(r"[^a-z0-9]+", " ", s).split())


def matches(text: str, aliases: list[str]) -> bool:
    """Does the completion `text` start with (a normalised form of) one of the aliases?"""
    t = norm(text)
    for a in aliases:
        a = norm(a)
        if len(a) >= 2 and (t == a or t.startswith(a + " ")):
            return True
    return False


def same_entity(a_aliases: list[str], b_aliases: list[str]) -> bool:
    return bool({norm(x) for x in a_aliases} & {norm(x) for x in b_aliases})


# --------------------------------------------------------------------- probe tokens

STOPWORDS = {
    "the", "of", "a", "an", "and", "de", "del", "della", "di", "da", "van", "von", "der", "den", "la", "le",
    "el", "al", "bin", "ibn", "y", "du", "des", "st", "saint", "sir", "jr", "sr", "city", "university", "college",
    "inc", "ltd", "company", "corporation", "co", "group", "games", "studios", "studio", "entertainment",
    "software", "interactive", "republic", "kingdom", "new", "north", "south", "east", "west",
}


PROBE_MODES = ("words", "last_word")


def probe_tokens(tok, name: str, mode: str = "words") -> list[int]:
    """First token of " <word>" for each content word of `name` (a lens hit on any of them
    counts as the entity); mode "last_word": only the last content word (a person's surname).
    Falls back to the first token of the name."""
    if mode not in PROBE_MODES:
        raise ValueError(f"unknown probe mode {mode!r}")
    words = [w for w in re.findall(r"[^\W_]+(?:['’][^\W_]+)?", name) if w.lower() not in STOPWORDS and len(w) >= 3]
    if mode == "last_word":
        words = words[-1:]
    ids = []
    for w in words:
        t = tok(" " + w, add_special_tokens=False)["input_ids"]
        if t and t[0] not in ids:
            ids.append(t[0])
    return ids or tok(" " + name, add_special_tokens=False)["input_ids"][:1]


ROLES = ("orig_bridge", "peer_bridge", "ctrl_bridge", "orig_answer", "peer_answer", "ctrl_answer")


def role_probes(tok, fact: Fact, peer: Fact | None, control: Fact, mode: str = "words") -> dict[str, list[int]]:
    """Probe tokens per role with tokens shared by two roles removed (a shared token cannot
    tell the roles apart); a role left empty cannot be read for this item. peer None (free
    debate, no scripted peer): the peer roles are empty."""
    pt = lambda name: probe_tokens(tok, name, mode)
    raw = {"orig_bridge": pt(fact.e2), "peer_bridge": pt(peer.e2) if peer else [], "ctrl_bridge": pt(control.e2),
           "orig_answer": pt(fact.e3), "peer_answer": pt(peer.e3) if peer else [], "ctrl_answer": pt(control.e3)}
    count: dict[int, int] = {}
    for ids in raw.values():
        for t in set(ids):
            count[t] = count.get(t, 0) + 1
    return {role: [t for t in ids if count[t] == 1] for role, ids in raw.items()}


# ------------------------------------------------------------------ splits, pairing


def split_of(fact: Fact, dev_frac: float, seed: int) -> str:
    """dev / test by the bridge entity, so no bridge is seen in both splits."""
    h = int(hashlib.md5(f"{seed}-{norm(fact.e2)}".encode()).hexdigest(), 16) % 10_000
    return "dev" if h < dev_frac * 10_000 else "test"


def assign_partners(facts: list[Fact], pool: list[Fact], seed: int) -> list[tuple[Fact, Fact, Fact]]:
    """(fact, peer, control) with peer and control drawn from `pool`, same category, and
    bridges and answers all different from each other. Facts without two such partners are dropped."""
    by_cat: dict[str, list[Fact]] = {}
    for p in pool:
        by_cat.setdefault(p.category, []).append(p)
    out = []
    for f in facts:
        rng = random.Random(f"{seed}-partners-{f.uid}")
        cands = [p for p in by_cat.get(f.category, []) if p.uid != f.uid
                 and not same_entity(p.e2_aliases, f.e2_aliases) and not same_entity(p.e3_aliases, f.e3_aliases)]
        rng.shuffle(cands)
        peer = cands[0] if cands else None
        control = next((c for c in cands[1:] if not same_entity(c.e2_aliases, peer.e2_aliases)
                        and not same_entity(c.e3_aliases, peer.e3_aliases)), None) if peer else None
        if peer and control:
            out.append((f, peer, control))
    return out


def siblings(facts: list[Fact], pool: list[Fact]) -> dict[str, Fact]:
    """uid -> a pool fact with the same e1 and r1 but another r2 and another answer."""
    by_key: dict[tuple[str, str], list[Fact]] = {}
    for p in pool:
        by_key.setdefault((p.e1, p.r1_prompt), []).append(p)
    out = {}
    for f in facts:
        for s in by_key.get((f.e1, f.r1_prompt), []):
            if s.r2_template != f.r2_template and not same_entity(s.e3_aliases, f.e3_aliases):
                out[f.uid] = s
                break
    return out
