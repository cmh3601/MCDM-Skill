"""Tests for ELECTRE II, III, IV and Tri-B."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import mcdm_outranking as eo  # noqa: E402

# Benefit-only data so results can be compared with pyDecision, which assumes maximisation.
X = np.array([
    [15, 9, 6, 7],
    [10, 5, 7, 8],
    [22, 12, 1, 14],
    [31, 10, 6, 18],
    [8, 9, 0, 9],
    [21, 14, 4, 2],
    [24, 7, 5, 10],
], dtype=float)
W = np.array([0.4, 0.3, 0.1, 0.2])
T = np.ones(4)
Q = np.array([2.0, 2.0, 1.0, 1.0])
P = np.array([5.0, 4.0, 3.0, 4.0])
V = np.array([12.0, 9.0, 6.0, 10.0])
CHAIN = np.array([[5.0, 5, 5], [4, 4, 4], [3, 3, 3], [2, 2, 2]])


# --- ELECTRE III ----------------------------------------------------------------------------

def test_electre_iii_credibility_by_hand():
    Xs = np.array([[10.0], [7.0]])
    out = eo.electre_iii(Xs, [1.0], [1], q=1, p=4, v=6)
    # a1 vs a2: diff = -3 -> c = 1, sigma = 1. a2 vs a1: diff = 3 -> c = (4-3)/(4-1) = 1/3,
    # d = 0 (diff < p), sigma = 1/3.
    assert out["credibility"][0, 1] == pytest.approx(1.0)
    assert out["credibility"][1, 0] == pytest.approx(1 / 3)
    Xv = np.array([[10.0, 0.0], [3.0, 1.0]])
    out = eo.electre_iii(Xv, [0.5, 0.5], [1, 1], q=[1, 1], p=[3, 3], v=[5, 10])
    # a2 vs a1 on c1: diff = 7 >= v = 5 -> full veto
    assert out["credibility"][1, 0] == pytest.approx(0.0)


def test_electre_iii_chain_gives_total_order():
    out = eo.electre_iii(CHAIN, [1 / 3] * 3, [1, 1, 1], q=0.1, p=0.5, v=3)
    assert out["descending"]["ranks"].tolist() == [1, 2, 3, 4]
    assert out["ascending"]["ranks"].tolist() == [1, 2, 3, 4]
    assert out["relation"][0, 3] == "P+" and out["relation"][3, 0] == "P-"


def test_electre_iii_cost_criterion_is_oriented():
    a = eo.electre_iii(CHAIN, [1 / 3] * 3, [-1, -1, -1], q=0.1, p=0.5, v=3)
    assert a["descending"]["ranks"].tolist() == [4, 3, 2, 1]


def test_electre_iii_relation_is_consistent():
    out = eo.electre_iii(X, W, T, Q, P, V)
    rel = out["relation"]
    for i in range(len(X)):
        for j in range(len(X)):
            pair = {"P+": "P-", "P-": "P+", "I": "I", "R": "R"}[rel[i, j]]
            assert rel[j, i] == pair
    assert sorted(np.concatenate(out["descending"]["classes"]).tolist()) == list(range(len(X)))


def test_electre_iii_matches_pydecision_credibility():
    pytest.importorskip("pyDecision.algorithm", reason="pyDecision not installed")
    from pyDecision.algorithm.e_iii import credibility_matrix, global_concordance_matrix
    C_ref = global_concordance_matrix(X, P, Q, W)
    S_ref = credibility_matrix(X, C_ref, P, V)
    out = eo.electre_iii(X, W, T, Q, P, V)
    off = ~np.eye(len(X), dtype=bool)
    assert np.allclose(out["concordance"][off], C_ref[off])
    assert np.allclose(out["credibility"], S_ref)


# --- ELECTRE II -----------------------------------------------------------------------------

def test_electre_ii_chain():
    out = eo.electre_ii(CHAIN, [1 / 3] * 3, [1, 1, 1])
    assert out["descending"]["ranks"].tolist() == [1, 2, 3, 4]
    assert out["ascending"]["ranks"].tolist() == [1, 2, 3, 4]
    assert out["strong"][0, 1] and not out["strong"][1, 0]


def test_electre_ii_strong_implies_weak():
    out = eo.electre_ii(X, W, T)
    assert np.all(out["weak"][out["strong"]])


def test_electre_ii_matches_pydecision_relations():
    pytest.importorskip("pyDecision.algorithm", reason="pyDecision not installed")
    from pyDecision.algorithm.e_ii import concordance_matrix, discordance_matrix, dominance_matrix
    Cr, Dr = concordance_matrix(X, W), discordance_matrix(X)
    sF, sf = dominance_matrix(Cr, Dr)
    out = eo.electre_ii(X, W, T, discordance="global_range")
    assert np.array_equal(out["strong"], sF.astype(bool))
    assert np.array_equal(out["weak"], sf.astype(bool))


def test_electre_ii_handles_cycles():
    # three alternatives beating each other in a cycle on concordance
    Xc = np.array([[3.0, 2, 1], [1, 3, 2], [2, 1, 3]])
    out = eo.electre_ii(Xc, [1 / 3] * 3, [1, 1, 1], c_minus=0.6, c_zero=0.6, c_plus=0.6,
                        d_minus=1.0, d_plus=1.0)
    assert out["descending"]["classes"] == [[0, 1, 2]]


# --- ELECTRE IV -----------------------------------------------------------------------------

def test_electre_iv_relations_are_nested():
    out = eo.electre_iv(X, T, Q, P, V)
    r = out["relations"]
    for inner, outer in (("S_q", "S_c"), ("S_c", "S_p"), ("S_p", "S_s"), ("S_s", "S_v")):
        assert np.all(r[outer][r[inner]]), f"{inner} must imply {outer}"


def test_electre_iv_counts_by_hand():
    Xs = np.array([[10.0, 5, 3], [7.0, 6, 3]])
    cnt = eo.electre_iv_counts(Xs, [1, 1, 1], q=1, p=2)
    # b = a2 over a = a1: c1 diff -3 (no), c2 diff +1 (indifference band), c3 equal
    assert (cnt["mp_ba"][0, 1], cnt["mq_ba"][0, 1], cnt["mi_ba"][0, 1]) == (0, 0, 1)
    # b = a1 over a = a2: c1 diff +3 > p -> strict
    assert (cnt["mp_ba"][1, 0], cnt["mq_ba"][1, 0], cnt["mi_ba"][1, 0]) == (1, 0, 0)
    out = eo.electre_iv(Xs, [1, 1, 1], q=1, p=2)
    assert out["credibility"][0, 1] == 1.0  # a1 quasi-dominates a2
    assert out["descending"]["ranks"].tolist() == [1, 2]


def test_electre_iv_chain():
    out = eo.electre_iv(CHAIN, [1, 1, 1], q=0.1, p=0.5)
    assert out["descending"]["ranks"].tolist() == [1, 2, 3, 4]


# --- ELECTRE Tri-B --------------------------------------------------------------------------

PROFILES = np.array([[10, 6, 2, 6], [20, 10, 5, 12]], dtype=float)


def test_tri_b_extremes_and_order():
    Xt = np.vstack([X, [40, 20, 10, 25], [0, 0, 0, 0]])
    out = eo.electre_tri_b(Xt, W, T, PROFILES, Q, P, V, cut=0.75)
    assert out["pessimistic"][-2] == 3 and out["optimistic"][-2] == 3
    assert out["pessimistic"][-1] == 1 and out["optimistic"][-1] == 1
    assert np.all(out["pessimistic"] <= out["optimistic"])


def test_tri_b_one_criterion_by_hand():
    out = eo.electre_tri_b(np.array([[5.0], [15.0], [25.0]]), [1.0], [1], [[10.0], [20.0]],
                           q=1, p=2, cut=0.75)
    assert out["pessimistic"].tolist() == [1, 2, 3]
    assert out["optimistic"].tolist() == [1, 2, 3]


def test_tri_b_rejects_unordered_profiles():
    with pytest.raises(ValueError):
        eo.electre_tri_b(X, W, T, PROFILES[::-1], Q, P, V)


def test_tri_b_pessimistic_uses_outranking_not_strict_preference():
    out = eo.electre_tri_b(X, W, T, PROFILES, Q, P, V, cut=0.75)
    # a3 = [22, 12, 1, 14] vs b2 = [20, 10, 5, 12]: sigma(a3, b2) = 0.4 + 0.3 + 0 + 0.2 = 0.9 and
    # b2 also outranks a3 (indifference). The pessimistic rule needs a S b_h, so a3 -> C3.
    assert out["sigma_a_b"][2, 1] == pytest.approx(0.9)
    assert out["relation_to_profiles"][2, 1] == "I"
    assert out["pessimistic"][2] == 3


def test_tri_b_optimistic_matches_pydecision():
    pytest.importorskip("pyDecision.algorithm", reason="pyDecision not installed")
    from pyDecision.algorithm import electre_tri_b
    out = eo.electre_tri_b(X, W, T, PROFILES, Q, P, V, cut=0.75)
    oc = electre_tri_b(X, W=W.tolist(), Q=Q.tolist(), P=P.tolist(), V=V.tolist(),
                       B=PROFILES.tolist(), cut_level=0.75, verbose=False, rule="oc")
    # pyDecision numbers categories from the best (0) downward.
    assert np.array_equal(out["optimistic"], out["categories"] - np.array(oc))
