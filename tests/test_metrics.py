import numpy as np

from silent_dissent import metrics as M

L = list("ABCD")


def rec(original, stated, rows, **kw):
    return {"letters": L, "original": original, "stated": stated, "lens_logits": rows, **kw}


def test_silent_dissent_definitions():
    # 3 layers; original A, stated C.
    rows = [[3.0, 0, 1.0, 0], [2.0, 0, 1.5, 0], [0.0, 0, 5.0, 0]]
    r = rec("A", "C", rows)
    assert M.is_silent_dissent(r, 0, "strict")
    assert M.is_silent_dissent(r, 1, "strict")
    assert not M.is_silent_dissent(r, 2, "strict")
    assert not M.is_silent_dissent(r, 2, "lenient", delta=1.0)
    assert M.is_silent_dissent(r, 2, "lenient", delta=6.0)
    assert not M.is_silent_dissent(rec("A", "A", rows), 0)  # no flip, no dissent


def test_decision_flip_layer():
    rows = [[0, 0, 1, 0], [2, 0, 1, 0], [0, 0, 1, 0], [0, 0, 2, 0]]
    assert M.decision_flip_layer(rec("A", "C", rows)) == 2
    assert M.decision_flip_layer(rec("A", "A", rows)) is None


def test_select_layer():
    base = [rec("B", "B", [[1, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]]) for _ in range(10)]
    assert M.select_layer(base, 0.9) == 1
    np.testing.assert_allclose(M.agreement_curve(base), [0, 1, 1])


def test_aggregation_internal_vote_recovers_dissent():
    # Gold A. Round 0: agents say A, A, B. Round 1: all say B but internally prefer A.
    lens_a = [[5, 0, 0, 0]]
    recs = []
    for a, s0 in enumerate("AAB"):
        recs.append({"item_id": "x", "round": 0, "gold": "A", "letters": L, "stated": s0,
                     "final_logits": [1, 0, 0, 0], "lens_logits": lens_a})
        recs.append({"item_id": "x", "round": 1, "gold": "A", "letters": L, "stated": "B",
                     "final_logits": [0, 1, 0, 0], "lens_logits": lens_a})
    t = M.aggregation_table(recs, layer=0).set_index("round")
    assert t.loc[1, "acc_stated"] == 0 and t.loc[1, "acc_internal"] == 1 and t.loc[1, "acc_initial"] == 1


def test_position_summary_flags_positions_that_keep_the_own_answer():
    import numpy as np

    def m(variant, stated, ok=True, target="C"):
        return {"variant": variant, "letters": list("ABCD"), "original": "A", "original_correct": ok,
                "target": None if variant == "solo" else target, "stated": stated}

    meta = [m("solo", "A"), m("solo", "A", ok=False)] + [m(v, "C", ok) for v in ("instructed", "instructed_hidden")
                                                          for ok in (True, False)]
    meta.append(m("instructed", "D"))  # did not comply: ignored by sens_*/target_*
    logits = np.zeros((len(meta), 2, 1, 4))  # 2 positions, 1 layer
    logits[:, 0, 0, 0] = 1.0  # position -2 reads the original letter A everywhere
    for i, r in enumerate(meta):  # position -1 reads what is stated
        logits[i, 1, 0, "ABCD".index(r["stated"])] = 1.0
    df = M.position_summary(meta, logits).set_index("position")
    assert "passes_shifted" not in df  # family without records is skipped
    assert df.loc[-2, "passes_instructed"] and df.loc[-2, "sens_instructed_hidden_wrong"] == 1.0
    assert not df.loc[-1, "passes_instructed"] and df.loc[-1, "reliability"] == 1.0
    assert df.loc[-1, "target_instructed"] == 1.0 and df.attrs["counts"]["target_instructed"] == 2

    # a family with an empty stratum never passes
    shifted = meta[:2] + [m("shifted", "B", ok, target="B") for ok in (True, False)] + [m("shifted_hidden", "B", target="B")]
    logits = np.zeros((len(shifted), 1, 1, 4))
    logits[:, 0, 0, 0] = 1.0
    df = M.position_summary(shifted, logits)
    assert df.loc[0, "sens_shifted_ok"] == 1.0 and not df.loc[0, "passes_shifted"]


def test_mention_control_table_separates_flips():
    def rec(stated, orig_logit, ctrl_logit):
        # letters A-D; original A, control C
        return {"condition": "mention_control", "round": 1, "original_correct": False, "letters": list("ABCD"),
                "original": "A", "stated": stated, "control": "C",
                "lens_logits": [[orig_logit, 0.0, ctrl_logit, 0.0]]}

    recs = [rec("A", 5.0, 0.0)] * 3 + [rec("B", -1.0, 0.0), rec("B", 1.0, 0.0)]
    t = M.mention_control_table(recs, 0).set_index("flipped")
    assert t.loc[False, "margin_n"] == 3 and t.loc[False, "p_orig_above_control"] == 1.0
    assert t.loc[True, "margin_n"] == 2 and t.loc[True, "p_orig_above_control"] == 0.5
