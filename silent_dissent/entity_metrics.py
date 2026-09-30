"""Analysis of the entity-answer study (no model needed).

Readout arrays are int16 ranks [N, P, L, R]: position (POS: 0 = subject's last token, 1 = last
token; extended readouts add 2 = the mention's last token and 3 = the minimum over mention ..
last token), layer, role (entity_facts.ROLES); -1 = role not readable for that item. The basic quantity is
"hit@k" (rank < k); a contrast is hit(role) - hit(baseline role) within the same record, e.g.
orig_bridge - ctrl_bridge: how much more often the original bridge is in the lens top-k than
an unrelated entity of the same kind.
"""
from __future__ import annotations

import warnings
from collections import Counter

import numpy as np
import pandas as pd

from .entity_facts import ROLES, matches, norm
from .metrics import bootstrap_ci

POS = {"subject": 0, "last": 1, "mention": 2, "span": 3}
CONTRASTS = {
    "orig_bridge": ("orig_bridge", "ctrl_bridge"), "peer_bridge": ("peer_bridge", "ctrl_bridge"),
    "orig_answer": ("orig_answer", "ctrl_answer"), "peer_answer": ("peer_answer", "ctrl_answer"),
}


def hits(ranks: np.ndarray, k: int) -> np.ndarray:
    """float hit@k with NaN where the role is not readable."""
    return np.where(ranks < 0, np.nan, (ranks < k).astype(float))


def contrast_values(ranks: np.ndarray, rows, contrast: str, pos: str, k: int) -> np.ndarray:
    """[n_rows, L]: hit(role) - hit(baseline) per record and layer."""
    a, b = CONTRASTS[contrast]
    h = hits(ranks[np.asarray(rows, dtype=int), POS[pos]], k)  # [n, L, R]
    return h[..., ROLES.index(a)] - h[..., ROLES.index(b)]


def curve(ranks, rows, contrast, pos, k) -> np.ndarray:
    v = contrast_values(ranks, rows, contrast, pos, k)
    with warnings.catch_warnings():  # a layer with no readable record is NaN, not an error
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(v, 0) if len(v) else np.full(ranks.shape[2], np.nan)


def band_summary(ranks, rows, contrast, pos, k, band) -> dict:
    """Mean over the band of the per-record contrast, with a bootstrap CI over records."""
    v = contrast_values(ranks, rows, contrast, pos, k)[:, band]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        per = np.nanmean(v, 1) if len(v) else np.array([])
    per = per[~np.isnan(per)]
    lo, hi = bootstrap_ci(per)
    return {"n": len(per), "mean": float(per.mean()) if len(per) else np.nan, "ci_lo": lo, "ci_hi": hi}


def band_rows(ranks, rows, contrast, pos, k, band) -> np.ndarray:
    """Per-record band mean of the contrast, aligned with `rows` (NaN: not readable)."""
    v = contrast_values(ranks, rows, contrast, pos, k)[:, band]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(v, 1) if len(v) else np.array([])


def band_values(ranks, rows, contrast, pos, k, band) -> np.ndarray:
    """Per-record band mean of the contrast (records with no readable value dropped)."""
    per = band_rows(ranks, rows, contrast, pos, k, band)
    return per[~np.isnan(per)]


def item_means(records: list[dict], rows, values: np.ndarray) -> dict[str, float]:
    """{uid: mean of the readable values of that item's rows} (items with none dropped)."""
    by: dict[str, list[float]] = {}
    for i, v in zip(rows, values):
        if not np.isnan(v):
            by.setdefault(records[i]["uid"], []).append(float(v))
    return {u: float(np.mean(v)) for u, v in by.items()}


def _one_sample(hid: str, x: np.ndarray, **extra) -> dict:
    lo, hi = bootstrap_ci(x)
    return {"id": hid, "estimate": float(x.mean()) if len(x) else np.nan, "ci_lo": lo, "ci_hi": hi, "n": len(x),
            "supported": bool(len(x) and lo > 0), **extra}


def _two_sample(hid: str, a: np.ndarray, b: np.ndarray, **extra) -> dict:
    d = diff_ci(a, b)
    return {"id": hid, "estimate": d["diff"], "ci_lo": d["ci_lo"], "ci_hi": d["ci_hi"], "n": (d["n_a"], d["n_b"]),
            "supported": bool(d["ci_lo"] > 0), **extra}


def diff_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict:
    """mean(a) - mean(b) for independent samples, percentile bootstrap 95% CI."""
    if len(a) == 0 or len(b) == 0:
        return {"diff": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "n_a": len(a), "n_b": len(b)}
    rng = np.random.default_rng(seed)
    d = rng.choice(a, (n_boot, len(a))).mean(1) - rng.choice(b, (n_boot, len(b))).mean(1)
    return {"diff": float(a.mean() - b.mean()), "ci_lo": float(np.quantile(d, 0.025)),
            "ci_hi": float(np.quantile(d, 0.975)), "n_a": len(a), "n_b": len(b)}


def evaluate_hypotheses(records: list[dict], arrays: dict[str, np.ndarray], prereg: dict, n_peers: int) -> list[dict]:
    """H1-H4 of the entity prereg on a pressure run (round 1, the prereg format and band)."""
    band, k, lens = prereg["band"], prereg["k"], prereg["lens"]
    sel = lambda **kw: [i for i, r in enumerate(records) if r["round"] == 1 and r["format"] == prereg["format"]
                        and all(r.get(a) == b for a, b in kw.items())]
    flip = lambda style: sel(condition="pressure", style=style, n_peers=n_peers, outcome="peer")
    val = lambda idx, c, name=lens: band_values(arrays[name], idx, c, "last", k, band)
    out = []
    h1 = val(flip("answer"), "orig_bridge")
    lo, hi = bootstrap_ci(h1)
    out.append({"id": "H1", "estimate": float(h1.mean()) if len(h1) else np.nan, "ci_lo": lo, "ci_hi": hi,
                "n": len(h1), "supported": bool(len(h1) and lo > 0)})
    mention = sel(condition="mention", n_peers=n_peers, outcome="original")
    h2 = diff_ci(val(flip("answer"), "peer_bridge"), val(mention, "peer_bridge"))
    out.append({"id": "H2", "estimate": h2["diff"], "ci_lo": h2["ci_lo"], "ci_hi": h2["ci_hi"],
                "n": (h2["n_a"], h2["n_b"]), "supported": bool(h2["ci_lo"] > 0)})
    h3 = diff_ci(val(flip("hop2"), "orig_bridge"), val(flip("hop1"), "orig_bridge"))
    out.append({"id": "H3", "estimate": h3["diff"], "ci_lo": h3["ci_lo"], "ci_hi": h3["ci_hi"],
                "n": (h3["n_a"], h3["n_b"]), "supported": bool(h3["ci_lo"] > 0)})
    idx = flip("answer")
    if "logit" in arrays and len(idx):  # paired: same records under both lenses
        a = contrast_values(arrays[lens], idx, "orig_bridge", "last", k)[:, band]
        b = contrast_values(arrays["logit"], idx, "orig_bridge", "last", k)[:, band]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            d = np.nanmean(a, 1) - np.nanmean(b, 1)
        d = d[~np.isnan(d)]
        lo, hi = bootstrap_ci(d)
        out.append({"id": "H4", "estimate": float(d.mean()) if len(d) else np.nan, "ci_lo": lo, "ci_hi": hi,
                    "n": len(d), "supported": bool(len(d) and lo > 0)})
    return out


def state_key(r: dict) -> tuple:
    """One agent's conversation: the same key in every round."""
    return (r["uid"], r.get("format"), r["condition"], r.get("style"), r.get("n_peers"), r.get("hidden_own", False),
            r.get("absent_own", False))


def evaluate_rounds(records: list[dict], arrays: dict[str, np.ndarray], prereg: dict, addendum: dict) -> list[dict]:
    """Addendum "rounds" (later rounds written with a fixed variant): R1 / R2 are H1 / H2 at the
    last round, on agents that gave the peers' answer in every round 1..last; plus the paired
    change of the H1 statistic from round 1 (exploratory, no prediction)."""
    band, k, lens, n, last = prereg["band"], prereg["k"], prereg["lens"], addendum["n_peers"], addendum["round"]
    outcomes: dict[tuple, dict[int, str]] = {}
    index: dict[tuple, dict[int, int]] = {}
    for i, r in enumerate(records):
        outcomes.setdefault(state_key(r), {})[r["round"]] = r["outcome"]
        index.setdefault(state_key(r), {})[r["round"]] = i
    stayed = [key for key, o in outcomes.items() if key[2] == "pressure" and key[3] == "answer" and key[4] == n
              and key[1] == prereg["format"] and all(o.get(t) == "peer" for t in range(1, last + 1))]
    at = lambda rnd: [index[key][rnd] for key in stayed]
    val = lambda idx, c: band_values(arrays[lens], idx, c, "last", k, band)
    mention = [i for i, r in enumerate(records) if r["round"] == last and r["condition"] == "mention"
               and r.get("n_peers") == n and r["format"] == prereg["format"] and r["outcome"] == "original"]
    out = [_one_sample("R1", val(at(last), "orig_bridge")),
           _two_sample("R2", val(at(last), "peer_bridge"), val(mention, "peer_bridge"))]
    a = band_rows(arrays[lens], at(last), "orig_bridge", "last", k, band)
    b = band_rows(arrays[lens], at(1), "orig_bridge", "last", k, band)
    d = (a - b)[~np.isnan(a - b)] if len(a) else np.array([])
    ch = _one_sample("change", d)
    ch.update({"supported": None, "note": f"round {last} minus round 1, same agents (exploratory)"})
    return out + [ch]


def evaluate_bridges(records: list[dict], arrays: dict[str, np.ndarray], addendum: dict) -> list[dict]:
    """Addendum "bridges" (non-place bridges, the addendum's readout position and band): B1, B2, B4
    as H1, H2, H4 at round 1, pooled over the addendum's n_peers with one value per item (the mean
    of its records), bootstrap over items."""
    band, k, lens, pos, fmt = addendum["band"], addendum["k"], addendum["lens"], addendum["position"], addendum["format"]
    sel = lambda **kw: [i for i, r in enumerate(records) if r["round"] == 1 and r["format"] == fmt
                        and r.get("n_peers") in addendum["n_peers"] and all(r.get(a) == b for a, b in kw.items())]
    flip = sel(condition="pressure", style="answer", outcome="peer")
    mention = sel(condition="mention", outcome="original")
    per_item = lambda idx, c, name=lens: item_means(records, idx, band_rows(arrays[name], idx, c, pos, k, band))
    b1 = per_item(flip, "orig_bridge")
    out = [_one_sample("B1", np.array(list(b1.values())), unit="item"),
           _two_sample("B2", np.array(list(per_item(flip, "peer_bridge").values())),
                       np.array(list(per_item(mention, "peer_bridge").values())), unit="item")]
    if "logit" in arrays:
        lg = per_item(flip, "orig_bridge", "logit")
        paired = np.array([b1[u] - lg[u] for u in b1 if u in lg])
        out.append(_one_sample("B4", paired, unit="item"))
    return out


OWN_FLAG = {"hidden": "hidden_own", "absent": "absent_own"}


def _own_rows(records: list[dict], variant: str | None, fmt: str, n_peers: int, **kw) -> list[int]:
    """Round-1 rows of one own-variant (None: the agent's answer visible, i.e. neither flag)."""
    def is_variant(r):
        flags = {v: bool(r.get(f, False)) for v, f in OWN_FLAG.items()}
        return not any(flags.values()) if variant is None else flags[variant]
    return [i for i, r in enumerate(records) if r["round"] == 1 and r["format"] == fmt and r.get("n_peers") == n_peers
            and is_variant(r) and all(r.get(a) == b for a, b in kw.items())]


def evaluate_own(records: list[dict], arrays: dict[str, np.ndarray], addendum: dict) -> list[dict]:
    """Addendum "own": O1-O4 on the pressure_test_own run (readout, format and n_peers from the
    addendum, which copies them from the main prereg). A cell with fewer than min_test_n records
    gives no verdict (supported None)."""
    band, k, lens, fmt, n = addendum["band"], addendum["k"], addendum["lens"], addendum["format"], addendum["n_peers"]
    flip = lambda v: _own_rows(records, v, fmt, n, condition="pressure", style="answer", outcome="peer")
    val = lambda idx, c="orig_bridge", name=lens: band_values(arrays[name], idx, c, "last", k, band)
    few = lambda *xs: min(len(x) for x in xs) < addendum["min_test_n"]
    ids = {h["id"] for h in addendum["hypotheses"]}

    def verdict(h, *xs):
        if few(*xs):
            h.update({"supported": None, "note": f"fewer than {addendum['min_test_n']} records: no verdict"})
        return h

    hid = val(flip("hidden"))
    out = [verdict(_one_sample("O1", hid), hid)]
    if "O2" in ids:
        ab = val(flip("absent"))
        out.append(verdict(_two_sample("O2", hid, ab), hid, ab))
        out.append(verdict(_one_sample("O3", ab), ab))
    if "logit" in arrays:
        a = band_rows(arrays[lens], flip("hidden"), "orig_bridge", "last", k, band)
        b = band_rows(arrays["logit"], flip("hidden"), "orig_bridge", "last", k, band)
        d = (a - b)[~np.isnan(a - b)] if len(a) else np.array([])
        out.append(verdict(_one_sample("O4", d), d))
    return out


def own_summary(records: list[dict], arrays: dict[str, np.ndarray], addendum: dict,
                ref: tuple[list[dict], dict[str, np.ndarray]] | None = None) -> dict:
    """Exploratory companion of evaluate_own: give-in rates, band-mean cells per variant, condition,
    lens and role, the H2 analogue within each variant, and per-item paired differences of the
    original-bridge readout for items that gave in under both of two variants (hidden - visible
    from the main run `ref`, hidden - absent)."""
    band, k, lens, fmt, n = addendum["band"], addendum["k"], addendum["lens"], addendum["format"], addendum["n_peers"]
    runs = {v: (records, arrays) for v in addendum["variants"]}
    if ref is not None:
        runs["visible"] = ref
    rnd = lambda x: None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 4)
    fmt_s = lambda s: {"mean": rnd(s["mean"]), "ci": [rnd(s["ci_lo"]), rnd(s["ci_hi"])], "n": s["n"]}
    out = {"rates": {}, "cells": {}, "h2": {}, "paired": {}}
    per_item = {}
    for v, (recs, arrs) in runs.items():
        var = None if v == "visible" else v
        rows = lambda **kw: _own_rows(recs, var, fmt, n, **kw)
        for st in addendum["styles"]:
            idx = rows(condition="pressure", style=st)
            out["rates"][f"{v}/{st}"] = {"peer": rnd(np.mean([recs[i]["outcome"] == "peer" for i in idx])) if idx else None,
                                         "other": rnd(np.mean([recs[i]["outcome"] == "other" for i in idx])) if idx else None,
                                         "n": len(idx)}
        groups = {"agree": rows(condition="agree", outcome="original"),
                  "mention": rows(condition="mention", outcome="original"),
                  "held": rows(condition="pressure", style="answer", outcome="original"),
                  "flipped": rows(condition="pressure", style="answer", outcome="peer"),
                  "flipped_hop1": rows(condition="pressure", style="hop1", outcome="peer")}
        for g, idx in groups.items():
            for name in arrs:
                for role in ("orig_bridge", "peer_bridge"):
                    out["cells"][f"{v}/{g}/{name}/{role}"] = fmt_s(band_summary(arrs[name], idx, role, "last", k, band))
        d = diff_ci(band_values(arrs[lens], groups["flipped"], "peer_bridge", "last", k, band),
                    band_values(arrs[lens], groups["mention"], "peer_bridge", "last", k, band))
        out["h2"][v] = {"diff": rnd(d["diff"]), "ci": [rnd(d["ci_lo"]), rnd(d["ci_hi"])], "n": [d["n_a"], d["n_b"]]}
        per_item[v] = item_means(recs, groups["flipped"],
                                 band_rows(arrs[lens], groups["flipped"], "orig_bridge", "last", k, band))
    for a, b in (("hidden", "visible"), ("hidden", "absent")):
        if a in per_item and b in per_item:
            common = sorted(set(per_item[a]) & set(per_item[b]))
            x = np.array([per_item[a][u] - per_item[b][u] for u in common])
            s = _one_sample(f"{a}-{b}", x)
            out["paired"][f"{a}-{b}"] = {"diff": rnd(s["estimate"]), "ci": [rnd(s["ci_lo"]), rnd(s["ci_hi"])],
                                         "n_items": len(common)}
    return out


GROUP = ["split", "format", "condition", "style", "n_peers", "round"]


def group_keys(records: list[dict], keys: list[str] = GROUP) -> list[str]:
    """`keys` plus the own-addendum flags that some record sets (so runs without them keep their columns)."""
    return keys + [f for f in OWN_FLAG.values() if any(r.get(f) for r in records)]


def _key(r: dict, keys) -> tuple:
    return tuple(r.get(k) if r.get(k) is not None else "-" for k in keys)


def outcome_table(records: list[dict]) -> pd.DataFrame:
    """Share of answers that are the original / the peer's / the sibling's / other, per cell."""
    group = group_keys(records)
    df = pd.DataFrame([{**{k: r.get(k) if r.get(k) is not None else "-" for k in group}, "outcome": r["outcome"]}
                       for r in records])
    t = df.groupby(group)["outcome"].value_counts(normalize=True).unstack(fill_value=0.0)
    t["n"] = df.groupby(group).size()
    return t.reset_index()


def latent_table(records: list[dict], ranks: np.ndarray, k: int, band: list[int], pos: str = "last",
                 split_by_outcome: bool = True) -> pd.DataFrame:
    """Band-mean contrasts per cell (and per outcome), with bootstrap CIs."""
    keys = group_keys(records) + (["outcome"] if split_by_outcome else [])
    groups: dict[tuple, list[int]] = {}
    for i, r in enumerate(records):
        groups.setdefault(_key(r, keys), []).append(i)
    rows = []
    for key, idx in sorted(groups.items(), key=lambda kv: str(kv[0])):
        row = dict(zip(keys, key))
        for c in CONTRASTS:
            s = band_summary(ranks, idx, c, pos, k, band)
            row.update({f"{c}": s["mean"], f"{c}_lo": s["ci_lo"], f"{c}_hi": s["ci_hi"]})
        row["n"] = len(idx)
        rows.append(row)
    return pd.DataFrame(rows)


def layer_curves(records: list[dict], ranks: np.ndarray, k: int, pos: str, keys: list[str]) -> pd.DataFrame:
    """Long table: cell, contrast, layer, value (for plots)."""
    groups: dict[tuple, list[int]] = {}
    for i, r in enumerate(records):
        groups.setdefault(_key(r, keys), []).append(i)
    rows = []
    for key, idx in groups.items():
        for c in CONTRASTS:
            for layer, v in enumerate(curve(ranks, idx, c, pos, k)):
                rows.append({**dict(zip(keys, key)), "contrast": c, "layer": layer, "value": v, "n": len(idx)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- robustness (exploratory)


def rank_rows(ranks: np.ndarray, rows, contrast: str, pos: str, band: list[int], kind: str) -> np.ndarray:
    """Per-record band mean of a readout that needs no top-k threshold, aligned with `rows` (NaN: not
    readable). From the ranks of the role and its baseline (clipped at the readout's maximum rank):
    kind "logrank": log10(rank_baseline + 1) - log10(rank_role + 1), orders of magnitude, 0 = no
    preference; kind "win": 1 if the role outranks the baseline, 1/2 on a tie, else 0 (1/2 = no preference)."""
    a, b = CONTRASTS[contrast]
    r = ranks[np.asarray(rows, dtype=int), POS[pos]][:, band].astype(float)  # [n, band, R]
    ra, rb = r[..., ROLES.index(a)], r[..., ROLES.index(b)]
    if kind == "logrank":
        with np.errstate(divide="ignore", invalid="ignore"):  # unreadable (-1) is masked below
            v = np.log10(rb + 1) - np.log10(ra + 1)
    elif kind == "win":
        v = (ra < rb) + 0.5 * (ra == rb)
    else:
        raise ValueError(kind)
    v = np.where((ra >= 0) & (rb >= 0), v, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(v, 1) if len(v) else np.array([])


def boot_means(samples: list[tuple[np.ndarray, list]], n_boot: int, seed: int = 0) -> np.ndarray:
    """[len(samples), n_boot] bootstrap means with whole clusters resampled: each sample is (values,
    cluster labels); one draw of clusters (with replacement) is shared by all samples, so samples
    that share clusters (the same items in two conditions) stay paired. A replicate without any value
    of a sample is NaN. With one label per value this is the ordinary bootstrap over values."""
    labels = sorted({c for _, cl in samples for c in cl}, key=str)
    if not labels:
        return np.full((len(samples), n_boot), np.nan)
    at = {c: j for j, c in enumerate(labels)}
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(labels), (n_boot, len(labels)))
    mult = np.zeros((n_boot, len(labels)))
    np.add.at(mult, (np.repeat(np.arange(n_boot), len(labels)), draw.ravel()), 1)
    out = []
    for v, cl in samples:
        s, n = np.zeros(len(labels)), np.zeros(len(labels))
        j = [at[c] for c in cl]
        np.add.at(s, j, np.asarray(v, dtype=float))
        np.add.at(n, j, 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            out.append((mult @ s) / (mult @ n))
    return np.array(out)


def boot_summary(dist: np.ndarray, alpha: float = 0.05) -> dict:
    """Percentile CI and two-sided bootstrap p-value (against 0) of a bootstrap distribution."""
    d = dist[~np.isnan(dist)]
    if not len(d):
        return {"ci": [np.nan, np.nan], "p": np.nan, "n_boot": 0}
    p = min(1.0, 2 * (1 + min((d <= 0).sum(), (d >= 0).sum())) / (len(d) + 1))
    return {"ci": [float(np.quantile(d, alpha / 2)), float(np.quantile(d, 1 - alpha / 2))], "p": p, "n_boot": len(d)}


def holm(pvals: list[float]) -> list[float]:
    """Holm-adjusted p-values (family-wise error), in the input order."""
    order = np.argsort(pvals)
    m, adj, run = len(pvals), [np.nan] * len(pvals), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * pvals[i]))
        adj[i] = run
    return adj


def select_band(solo_curve: np.ndarray, threshold: float) -> list[int]:
    """Layers where the round-0 readout of the original bridge beats the control by >= threshold."""
    return [int(l) for l in np.flatnonzero(np.nan_to_num(solo_curve, nan=-1) >= threshold)]


# ----------------------------------------------------------------------------- E3


def timeline_table(meta: list[dict], arrays: list[np.ndarray], labels: list[list[str]], k: int,
                   layers: list[int]) -> pd.DataFrame:
    """Mean contrast per (condition, style, outcome, segment label, token index within the segment)."""
    rows = []
    for m, arr, lab in zip(meta, arrays, labels):
        h = hits(arr, k)  # [T, L', R]
        within: Counter = Counter()
        for t, seg in enumerate(lab):
            within[seg] += 1
            for c, (a, b) in CONTRASTS.items():
                v = h[t, :, ROLES.index(a)] - h[t, :, ROLES.index(b)]
                for j, layer in enumerate(layers):
                    rows.append({"condition": m.get("condition"), "style": m.get("style") or "-",
                                 "outcome": m.get("outcome"), "segment": seg or "-", "index": within[seg] - 1,
                                 "t": t, "contrast": c, "layer": layer, "value": v[j]})
    df = pd.DataFrame(rows)
    return df.groupby(["condition", "style", "outcome", "segment", "contrast", "layer"])["value"].mean().reset_index()


# ----------------------------------------------------------------------------- E4


def injection_table(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    keys = ["condition", "style", "outcome", "inject_kind", "inject_layer", "alpha"]
    keys = [c for c in keys if c in df]
    df = df.fillna({"style": "-"})
    t = df.groupby(keys)["top1_is"].value_counts(normalize=True).unstack(fill_value=0.0)
    t["n"] = df.groupby(keys).size()
    return t.reset_index()


# ----------------------------------------------------------------------------- E5


def _plurality(answers: list[str]) -> str | None:
    c = Counter(norm(a) for a in answers if a)
    return c.most_common(1)[0][0] if c else None


def debate_table(records: list[dict], lens: str, band: list[int], stated_only: bool = True) -> pd.DataFrame:
    """Group accuracy per round under three rules: plurality of stated answers, plurality of each
    agent's latent choice (the candidate with the best mean rank over the band), and plurality of
    round-0 answers. A group answer is correct if it is the truth (candidate 0).

    stated_only: latent choices range only over answers some agent actually gave in round 0; the
    truth is a candidate only if an agent stated it (otherwise the readout could pick an answer
    nobody proposed, which inflates the latent rule)."""
    by: dict[tuple[str, int], list[dict]] = {}
    for r in records:
        by.setdefault((r["uid"], r["round"]), []).append(r)
    rows = []
    for (uid, rnd), recs in by.items():
        truth = norm(recs[0]["candidates"][0])
        stated = _plurality([r["answer"] if not r["correct"] else recs[0]["candidates"][0] for r in recs])
        truth_proposed = any(x["correct"] for x in by.get((uid, 0), []))
        latent = []
        for r in recs:
            cr = np.array(r["candidate_ranks"][lens], dtype=float)[:, band]  # [C, band]
            cr[cr < 0] = np.nan
            if stated_only and not truth_proposed:
                cr[0] = np.nan
            if np.isnan(cr).all():
                continue
            with warnings.catch_warnings():  # a candidate with no readable token is NaN
                warnings.simplefilter("ignore", RuntimeWarning)
                latent.append(r["candidates"][int(np.nanargmin(np.nanmean(cr, 1)))])
        r0 = [x for x in by.get((uid, 0), recs)]
        initial = _plurality([x["answer"] if not x["correct"] else recs[0]["candidates"][0] for x in r0])
        rows.append({"uid": uid, "round": rnd, "acc_stated": stated == truth,
                     "acc_latent": _plurality(latent) == truth if latent else np.nan, "acc_initial": initial == truth,
                     "n_correct_agents": sum(r["correct"] for r in recs)})
    df = pd.DataFrame(rows)
    return df.groupby("round")[["acc_stated", "acc_latent", "acc_initial"]].mean().reset_index()


# ----------------------------------------------------------------------------- E5 follow-ups (exploratory)

VERDICT_WORDS = {"correct", "incorrect", "identical", "same", "agree", "agreed", "right", "true", "yes", "no",
                 "unchanged"}


def debate_kind(r: dict) -> str:
    """Class of a free-debate answer (docs/EXPERIMENT_BACKLOG.md, B07): correct; other_candidate (a
    wrong answer some agent gave in round 0); degenerate (empty, a verdict word such as 'correct' or
    'identical', 'the same ...', or a mention of an agent); other (any other entity)."""
    a = norm(r["answer"])
    w = a.split()
    if r["correct"]:
        return "correct"
    if any(matches(r["answer"], [c]) for c in r["candidates"][1:]):
        return "other_candidate"
    if not a or w[0] in VERDICT_WORDS or a.startswith("the same") or "agent" in w:
        return "degenerate"
    return "other"


def debate_group_correct(recs: list[dict], drop_degenerate: bool = False) -> bool | None:
    """Is the plurality of one fact's answers in one round the truth? Correct answers count as the
    truth. Without drop_degenerate this is debate_table's stated rule: a group with no usable answer
    (all empty) is wrong. With drop_degenerate, degenerate answers are abstentions and a group with
    no answer left gives None (the caller counts it as excluded or as wrong)."""
    truth = recs[0]["candidates"][0]
    answers = [truth if r["correct"] else r["answer"] for r in recs
               if not (drop_degenerate and debate_kind(r) == "degenerate")]
    p = _plurality(answers)
    if p is None:
        return None if drop_degenerate else False
    return p == norm(truth)


def same_answer(a: str, b: str) -> bool:
    """Two free-debate answers name the same entity: equal after normalisation, or one is the other
    followed by more words."""
    x, y = norm(a), norm(b)
    return len(x) >= 2 and len(y) >= 2 and (x == y or x.startswith(y + " ") or y.startswith(x + " "))


def debate_transitions(records: list[dict]) -> list[str | None]:
    """Label of every record of round r >= 1 against round r - 1 (None in round 0), for an agent
    that was correct at r - 1: gave_in (wrong at r, stating the wrong answer another agent gave at
    r - 1), switched (wrong at r otherwise), held (correct at r, while at least one other agent
    gave a wrong, non-degenerate answer at r - 1), agreed (correct at r, all others correct at r - 1).
    An agent wrong at r - 1 gets 'was_wrong'."""
    by: dict[tuple, list[dict]] = {}
    for r in records:
        by.setdefault((r["uid"], r["round"]), []).append(r)
    out: list[str | None] = []
    for r in records:
        if r["round"] == 0:
            out.append(None)
            continue
        before = by[(r["uid"], r["round"] - 1)]
        prev = next(x for x in before if x["agent"] == r["agent"])
        others = [x for x in before if x["agent"] != r["agent"]]
        if not prev["correct"]:
            out.append("was_wrong")
        elif not r["correct"]:
            wrong = [x["answer"] for x in others if not x["correct"]]
            out.append("gave_in" if any(same_answer(r["answer"], w) for w in wrong) else "switched")
        elif any(debate_kind(x) in ("other_candidate", "other") for x in others):
            out.append("held")
        elif all(x["correct"] for x in others):
            out.append("agreed")
        else:
            out.append("held_degenerate")  # correct, the only wrong peers gave degenerate answers
    return out


def debate_bridge_summary(records: list[dict], arrays: dict[str, np.ndarray], lens: str, band: list[int], k: int,
                          n_boot: int = 10000, min_n: int = 20) -> dict:
    """Original-bridge readout (hit@k minus control, band mean, last token) of free-debate agents per
    transition label and round, for known facts and for all facts, under `lens` and the logit lens;
    paired lens difference and logrank for agents that gave in; gave in minus held. Records are
    resampled by fact (agents of one fact are not independent); intervals only with >= min_n
    readable records."""
    labels = debate_transitions(records)

    def stat(values, rows):
        ok = ~np.isnan(values)
        v, cl = values[ok], [records[i]["uid"] for i, o in zip(rows, ok) if o]
        s = {"n": int(len(v)), "n_facts": len(set(cl)), "mean": float(v.mean()) if len(v) else None}
        if len(v) >= min_n:
            s["ci"] = boot_summary(boot_means([(v, cl)], n_boot)[0])["ci"]
        return s, v, cl

    out = {"lens": lens, "band": band, "k": k, "min_n": min_n, "n_boot": n_boot, "cells": {}, "contrasts": {}}
    rounds = sorted({r["round"] for r in records})
    for subset in ("known", "all"):
        keep = lambda i: subset == "all" or records[i].get("known")
        base = [i for i, r in enumerate(records) if r["round"] == 0 and r["correct"] and keep(i)]
        for name in (lens, "logit"):
            out["cells"][f"{subset}/round0_correct/{name}"] = stat(
                band_rows(arrays[name], base, "orig_bridge", "last", k, band), base)[0]
        for t in rounds[1:]:
            groups = {}
            for lab in ("gave_in", "switched", "held", "agreed", "held_degenerate", "was_wrong"):
                rows = [i for i, r in enumerate(records) if r["round"] == t and labels[i] == lab and keep(i)]
                groups[lab] = rows
                for name in (lens, "logit"):
                    out["cells"][f"{subset}/r{t}/{lab}/{name}"] = stat(
                        band_rows(arrays[name], rows, "orig_bridge", "last", k, band), rows)[0]
            g = groups["gave_in"]
            d = band_rows(arrays[lens], g, "orig_bridge", "last", k, band) - band_rows(arrays["logit"], g, "orig_bridge",
                                                                                     "last", k, band)
            out["contrasts"][f"{subset}/r{t}/gave_in/{lens}-logit"] = stat(d, g)[0]
            out["contrasts"][f"{subset}/r{t}/gave_in/logrank"] = stat(
                rank_rows(arrays[lens], g, "orig_bridge", "last", band, "logrank"), g)[0]
            _, a, ca = stat(band_rows(arrays[lens], g, "orig_bridge", "last", k, band), g)
            h = groups["held"]
            _, b, cb = stat(band_rows(arrays[lens], h, "orig_bridge", "last", k, band), h)
            diff = {"n": [len(a), len(b)], "mean": float(a.mean() - b.mean()) if len(a) and len(b) else None}
            if min(len(a), len(b)) >= min_n:
                bm = boot_means([(a, ca), (b, cb)], n_boot)
                diff["ci"] = boot_summary(bm[0] - bm[1])["ci"]
            out["contrasts"][f"{subset}/r{t}/gave_in-held/{lens}"] = diff
    out["counts"] = {f"r{t}/{lab}": sum(1 for i, r in enumerate(records) if r["round"] == t and labels[i] == lab)
                     for t in rounds[1:] for lab in sorted({l for l in labels if l})}
    return out

