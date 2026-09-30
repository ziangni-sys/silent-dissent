"""Entity study: configs parse, and every script runs end to end on the debug model."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from silent_dissent.config import load_config

ROOT = Path(__file__).resolve().parent.parent
ENTITY_CONFIGS = sorted(p for p in (ROOT / "configs").glob("*entity*.yaml"))
LENS_CONFIGS = sorted((ROOT / "configs").glob("*_jlens_*.yaml"))


@pytest.mark.parametrize("path", ENTITY_CONFIGS, ids=lambda p: p.name)
def test_entity_configs(path):
    cfg = load_config(str(path))
    ec = cfg["entity"]
    assert cfg["entity_prereg"] and cfg["data"]["n_choices"] == 4
    assert {"k", "band_threshold", "min_accuracy", "min_n"} <= set(ec["select"])
    assert ec["primary_n_peers"] in ec["n_peers"] and set(ec["styles"]) <= {"answer", "hop1", "hop2"}
    assert {"styles", "n_peers"} <= set(ec["timeline"]) and set(ec["timeline"]["styles"]) <= set(ec["styles"])
    ic = ec["intervention"]
    assert {"alphas", "kinds_flipped", "kinds_held", "style", "n_peers"} <= set(ic)
    assert {"n_items", "known_share", "n_agents", "rounds", "temperature"} <= set(ec["debate"])
    for part, keys in (("rounds", {"variants", "pilot_limit", "n_peers", "round", "max_other", "min_round1_agreement"}),
                       ("bridges", {"bridge_types", "probe_modes", "positions", "band_threshold", "category_threshold",
                                    "min_category_n", "min_n", "min_adoption_gap", "min_persistence", "styles",
                                    "baselines", "n_peers"}),
                       ("own", {"variants", "styles", "baselines", "n_peers", "pilot_limit", "max_other",
                                "min_agree_n", "min_test_n"})):
        if part in ec.get("addenda", {}):
            assert keys <= set(ec["addenda"][part]), part
    if "debug" not in path.name:  # the debug-only switches never leak into a real config
        assert ec.get("known_filter", True) and ec.get("require_correct_round0", True)
        assert ec["select"].get("retention_outcome", "sibling") == "sibling" and not ic.get("debug_any_outcome")
        br = ec.get("addenda", {}).get("bridges")
        assert br is None or (br["min_adoption_gap"] is not None and br["min_persistence"] is not None)
        assert "repeat" not in ec.get("addenda", {}).get("rounds", {}).get("variants", [])  # repeat is the reference
        own = ec.get("addenda", {}).get("own")
        assert own is None or (own["min_agree_n"] is not None and own["max_other"] <= 0.10 and own["min_test_n"] > 0
                               and set(own["variants"]) <= {"hidden", "absent"})


@pytest.mark.parametrize("path", LENS_CONFIGS, ids=lambda p: p.name)
def test_lens_fit_configs(path):
    cfg = load_config(str(path))
    assert cfg["lens"]["name"] == "jlens" and cfg["lens"]["kwargs"]["path"].startswith("lenses/")
    assert cfg["jlens"]["corpus"] == "wikitext" or cfg["jlens"]["corpus"].endswith(".jsonl")


def test_entity_pipeline_end_to_end(tmp_path):
    pytest.importorskip("transformers")
    cfg = yaml.safe_load((ROOT / "configs" / "debug_entity.yaml").read_text())
    cfg["model"]["name"] = str(tmp_path / "model")
    cfg["lens"]["kwargs"]["path"] = str(tmp_path / "out" / "jlens.pt")
    cfg["out_dir"] = str(tmp_path / "out")
    cfg["entity_prereg"] = str(tmp_path / "out" / "prereg_entity.json")
    cfg["jlens"]["corpus"] = str(ROOT / "tests" / "fixtures" / "corpus.txt")
    cfg["entity"]["sources"][0]["path"] = str(ROOT / "tests" / "fixtures" / "twohopfact_sample.csv")
    c = tmp_path / "debug_entity.yaml"
    c.write_text(yaml.safe_dump(cfg, sort_keys=False))

    def run(*args):
        r = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True)
        assert r.returncode == 0, f"{args}\n{r.stdout[-2000:]}\n{r.stderr[-4000:]}"
        return r.stdout

    run("-m", "silent_dissent.debug", str(tmp_path / "model"), "--entity")
    run("scripts/fit_jlens.py", "--config", str(c))
    assert "items with partners" in run("scripts/entity_build.py", "--config", str(c))
    run("scripts/entity_run.py", "--config", str(c), "--stage", "e0", "--split", "test")
    assert "FORMAT: plain" in run("scripts/entity_select.py", "--config", str(c), "--run", "e0_test")
    v2 = {**cfg, "entity": {**cfg["entity"], "select": {  # the dialogue band rule and the F2 ratio (dry run)
        **cfg["entity"]["select"], "band_rule": "dialogue", "latent_gap": -1.0, "dialogue_min": -1.0,
        "min_category_n": 0, "f2": "adoption", "min_adoption_ratio": 0.0, "min_persistence": -1.0,
        "supersedes": cfg["entity_prereg"]}}}
    c2 = tmp_path / "debug_entity_v2.yaml"
    c2.write_text(yaml.safe_dump(v2, sort_keys=False))
    r = subprocess.run([sys.executable, "scripts/entity_select.py", "--config", str(c2), "--run", "e0_test",
                        "--dry-run"], cwd=ROOT, capture_output=True, text=True)
    assert "round0_latent_window" in r.stdout and "Traceback" not in r.stderr, r.stderr[-3000:]
    run("scripts/entity_run.py", "--config", str(c), "--stage", "pressure", "--split", "test")
    run("scripts/entity_timeline.py", "--config", str(c), "--split", "test")
    run("scripts/entity_intervene.py", "--config", str(c), "--split", "test")
    run("scripts/entity_free_debate.py", "--config", str(c), "--split", "test")
    # E5 follow-ups: the answer check (B07) and the bridge readout rebuilt from the transcripts
    run("scripts/entity_debate_check.py", "--configs", str(c), "--out", str(tmp_path / "debate_check"))
    check = json.loads((tmp_path / "debate_check" / "summary.json").read_text())["models"][c.stem]
    assert {"kinds", "strings", "accuracy"} <= set(check) and "r1/all" in check["accuracy"]
    out = run("scripts/entity_debate_bridges.py", "--config", str(c))
    assert "reconstruction check" in out and "RECONSTRUCTION FAILED" not in out
    bridges = tmp_path / "out" / "entity" / "debate_test_bridges"
    info = json.loads((bridges / "info.json").read_text())
    assert info["reconstruction_identical"] == 1.0  # same prompts, same batches, CPU: identical ranks
    assert (tmp_path / "out" / "entity" / "analysis" / "debate_bridges.json").exists()
    with pytest.raises(AssertionError):  # never overwritten
        run("scripts/entity_debate_bridges.py", "--config", str(c))
    assert "transition counts" in run("scripts/entity_debate_bridges.py", "--config", str(c), "--analyze")
    out = run("scripts/entity_analyze.py", "--config", str(c))
    assert "pre-registered hypotheses" in out
    analysis = tmp_path / "out" / "entity" / "analysis"
    for name in ("hypotheses.json", "pressure_test_jlens_2peers.png", "timeline_test_jlens.png", "inject_test.csv",
                 "debate_test_jlens.csv", "e0_test_outcomes.csv"):
        assert (analysis / name).exists(), name
    with pytest.raises(AssertionError):  # the prereg is fixed once
        run("scripts/entity_select.py", "--config", str(c), "--run", "e0_test")

    # follow-ups: dev pilots, addenda, their test runs (scripts/entity_followup_{dev,test}.sh)
    with pytest.raises(AssertionError):  # test runs take follow-up settings only from an addendum
        run("scripts/entity_run.py", "--config", str(c), "--stage", "pressure", "--split", "test", "--later", "summary")
    with pytest.raises(AssertionError):  # no addendum yet
        run("scripts/entity_run.py", "--config", str(c), "--stage", "pressure", "--split", "test", "--addendum", "rounds")
    for v in ("repeat", "rephrase", "summary"):
        run("scripts/entity_run.py", "--config", str(c), "--stage", "pressure", "--split", "dev", "--later", v,
            "--n-peers", "2", "--rounds", "2", "--limit", "6", "--tag", f"rounds_{v}", "--no-extra-lenses")
    assert "orig_bridge - ctrl_bridge" in run("scripts/entity_bridge_probe.py", "--config", str(c), "--split", "dev")
    for m in ("words", "last_word"):
        run("scripts/entity_run.py", "--config", str(c), "--stage", "e0", "--split", "dev", "--formats", "plain",
            "--bridge-types", "country", "--probe-mode", m, "--extended", "--tag", f"bridges_{m}")
    assert "CHOSEN" in run("scripts/entity_addendum.py", "--config", str(c), "--part", "rounds")
    assert "CHOSEN" in run("scripts/entity_addendum.py", "--config", str(c), "--part", "bridges")
    with pytest.raises(AssertionError):  # an addendum is fixed once
        run("scripts/entity_addendum.py", "--config", str(c), "--part", "rounds")
    rounds = json.loads((tmp_path / "out" / "prereg_entity_rounds.json").read_text())
    bridges = json.loads((tmp_path / "out" / "prereg_entity_bridges.json").read_text())
    assert rounds["later"] in ("rephrase", "summary") and rounds["round"] == 2
    assert bridges["position"] in ("last", "mention", "span") and bridges["probe_mode"] in ("words", "last_word")
    # own: pilots with the agent's round-0 answer hidden / absent, the addendum, its test run
    with pytest.raises(AssertionError):  # --own is a dev-only setting
        run("scripts/entity_run.py", "--config", str(c), "--stage", "pressure", "--split", "test", "--own", "hidden")
    for v in ("hidden", "absent"):
        run("scripts/entity_run.py", "--config", str(c), "--stage", "pressure", "--split", "dev", "--own", v,
            "--no-extra-lenses")
    strict = {**cfg, "entity": {**cfg["entity"], "addenda": {**cfg["entity"]["addenda"], "own": {
        **cfg["entity"]["addenda"]["own"], "max_other": -1.0}}}}  # the hidden variant fails: exit code 3
    c3 = tmp_path / "debug_entity_strict.yaml"
    c3.write_text(yaml.safe_dump(strict, sort_keys=False))
    r = subprocess.run([sys.executable, "scripts/entity_addendum.py", "--config", str(c3), "--part", "own"], cwd=ROOT,
                       capture_output=True, text=True)
    assert r.returncode == 3 and "NO ADDENDUM" in r.stdout, r.stderr[-3000:]
    assert not (tmp_path / "out" / "prereg_entity_own.json").exists()
    assert "REGISTERED VARIANTS" in run("scripts/entity_addendum.py", "--config", str(c), "--part", "own")
    own = json.loads((tmp_path / "out" / "prereg_entity_own.json").read_text())
    assert own["variants"] == ["hidden", "absent"] and {h["id"] for h in own["hypotheses"]} == {"O1", "O2", "O3", "O4"}
    assert own["band"] == json.loads((tmp_path / "out" / "prereg_entity.json").read_text())["band"]
    for part in ("rounds", "bridges", "own"):
        run("scripts/entity_run.py", "--config", str(c), "--stage", "pressure", "--split", "test", "--addendum", part)
    recs = [json.loads(l) for l in (tmp_path / "out" / "entity" / "pressure_test_own" / "records.jsonl").open()]
    r1 = [r for r in recs if r["round"] == 1]
    assert r1 and all(r["hidden_own"] != r["absent_own"] for r in r1) and {r["n_peers"] for r in r1} == {2}
    info = json.loads((tmp_path / "out" / "entity" / "pressure_test_rounds" / "info.json").read_text())
    assert info["later"] == rounds["later"] and info["n_peers"] == [2] and info["rounds"] == 2
    info = json.loads((tmp_path / "out" / "entity" / "pressure_test_bridges" / "info.json").read_text())
    assert info["extended"] and info["probe_mode"] == bridges["probe_mode"] and info["rounds"] == 1
    out = run("scripts/entity_analyze.py", "--config", str(c))
    assert "addendum rounds" in out and "addendum bridges" in out and "round 1 of pressure_test_rounds" in out
    assert "addendum own" in out
    for part in ("rounds", "bridges", "own"):
        assert (analysis / f"hypotheses_{part}.json").exists()
    out = run("scripts/entity_robust.py", "--configs", str(c), "--out", str(tmp_path / "robust"), "--n-boot", "200")
    assert "Holm over" in out
    rob = json.loads((tmp_path / "robust" / "summary.json").read_text())["models"][c.stem]
    assert {"tests", "threshold_free", "k_curve", "within_item", "categories"} <= set(rob)
    assert set(rob["tests"]) == {"H1", "H2", "H3", "H4"} and "flipped-agree/orig_bridge/hit" in rob["within_item"]
    summary = json.loads((analysis / "own_summary.json").read_text())
    assert {"hidden/answer", "absent/answer", "visible/answer"} <= set(summary["rates"])
    ot = (analysis / "pressure_test_own_outcomes.csv").read_text().splitlines()[0]
    assert "hidden_own" in ot and "absent_own" in ot  # the variants are separate cells
    assert "hidden_own" not in (analysis / "pressure_test_outcomes.csv").read_text().splitlines()[0]
    before = {p.name: p.stat().st_mtime_ns for p in analysis.iterdir()}
    (analysis / "own_summary.json").unlink()
    out = run("scripts/entity_analyze.py", "--config", str(c), "--addendum", "own")  # own only
    assert "addendum own" in out and "addendum rounds" not in out and (analysis / "own_summary.json").exists()
    touched = {p.name for p in analysis.iterdir() if before.get(p.name) != p.stat().st_mtime_ns}
    assert touched and all(n.startswith("pressure_test_own_") or n in ("hypotheses_own.json", "own_summary.json")
                           for n in touched), touched


def test_entity_robust_matches_preregistered_estimates(tmp_path):
    """scripts/entity_robust.py on a synthetic run: its H1-H4 estimates are evaluate_hypotheses'."""
    from silent_dissent import entity_metrics as EM
    from silent_dissent.entity_facts import ROLES
    from silent_dissent.entity_io import load_run, save_run

    rng = np.random.default_rng(0)
    cfg = yaml.safe_load((ROOT / "configs" / "debug_entity.yaml").read_text())
    cfg["out_dir"] = str(tmp_path / "out")
    cfg["entity_prereg"] = str(tmp_path / "prereg_entity.json")
    c = tmp_path / "synth_entity.yaml"
    c.write_text(yaml.safe_dump(cfg, sort_keys=False))
    prereg = {"band": [1, 2], "k": 100, "lens": "jlens", "format": "plain", "primary_n_peers": 3,
              "categories": ["cat-a", "cat-b"]}
    (tmp_path / "prereg_entity.json").write_text(json.dumps(prereg))
    root = tmp_path / "out" / "entity"
    root.mkdir(parents=True)
    fact = lambda uid, cat, e2: {"uid": uid, "source": "synth", "category": cat, "e1": "x", "e2": e2, "e3": "y",
                                 "r1_prompt": "p", "r2_template": "{} q", "composed": "c"}
    items, records = [], []
    for j in range(60):
        uid, cat = f"u{j}", ["cat-a", "cat-b"][j % 2]
        items.append({"fact": fact(uid, cat, f"B{j % 9}"), "peer": fact(f"p{j}", cat, f"P{j % 7}"),
                      "control": fact(f"c{j}", cat, "C"), "sibling": None})
        base = {"uid": uid, "category": cat, "split": "test", "format": "plain", "hidden_own": False,
                "absent_own": False}
        records.append({**base, "condition": "solo", "style": None, "n_peers": 0, "round": 0, "outcome": "original"})
        for style in ("answer", "hop1", "hop2"):
            records.append({**base, "condition": "pressure", "style": style, "n_peers": 3, "round": 1,
                            "outcome": "peer" if rng.random() < 0.4 else "original"})
        for cond in ("agree", "mention", "mention_bridge"):
            records.append({**base, "condition": cond, "style": None, "n_peers": 3, "round": 1, "outcome": "original"})
    ranks = {ln: rng.integers(0, 400, (len(records), 2, 4, len(ROLES))).astype(np.int16) for ln in ("jlens", "logit")}
    ranks["jlens"][rng.random(len(records)) < 0.05, :, :, ROLES.index("orig_bridge")] = -1
    save_run(root / "pressure_test", records, {ln: [a] for ln, a in ranks.items()}, {"n_items": 60})
    with open(root / "items_test.jsonl", "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")

    r = subprocess.run([sys.executable, "scripts/entity_robust.py", "--configs", str(c), "--out", str(tmp_path / "rob"),
                        "--n-boot", "300"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    rob = json.loads((tmp_path / "rob" / "summary.json").read_text())["models"]["synth_entity"]
    recs, arrs, _ = load_run(root / "pressure_test")
    for h in EM.evaluate_hypotheses(recs, arrs, prereg, 3):
        t = rob["tests"][h["id"]]
        assert t["estimate"] == pytest.approx(h["estimate"], abs=1e-4), h["id"]
        assert t["n"] == (list(h["n"]) if isinstance(h["n"], tuple) else h["n"])
        assert t["cluster"]["ci"][0] <= t["estimate"] <= t["cluster"]["ci"][1]
    assert rob["tests"]["H1"]["n_clusters"] <= 9 and rob["tests"]["H2"]["n_clusters"] <= 7
    # within item: flipped - agree on the same items, the pre-registered readout
    fl = {x["uid"]: i for i, x in enumerate(recs) if x["condition"] == "pressure" and x["style"] == "answer"
          and x["outcome"] == "peer"}
    ag = {x["uid"]: i for i, x in enumerate(recs) if x["condition"] == "agree"}
    va = EM.band_rows(arrs["jlens"], list(fl.values()), "orig_bridge", "last", 100, [1, 2])
    vb = EM.band_rows(arrs["jlens"], [ag[u] for u in fl], "orig_bridge", "last", 100, [1, 2])
    d = (va - vb)[~np.isnan(va) & ~np.isnan(vb)]
    wi = rob["within_item"]["flipped-agree/orig_bridge/hit"]
    assert wi["estimate"] == pytest.approx(d.mean(), abs=1e-4) and wi["n"] == len(d)
    assert set(rob["categories"]) == {"cat-a", "cat-b"}
    assert json.loads((tmp_path / "rob" / "summary.json").read_text())["holm"]["family"]
