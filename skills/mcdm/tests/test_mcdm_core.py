"""Tests for mcdm_core. Cross-checks against pymcdm run only when it is installed."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import mcdm_core as mc  # noqa: E402

X = np.array([
    [25.0, 82, 14, 900],
    [28.5, 90, 10, 750],
    [22.0, 75, 18, 1100],
    [26.0, 88, 12, 820],
    [24.0, 79, 16, 1000],
])
W = np.array([0.35, 0.30, 0.20, 0.15])
T = np.array([-1, 1, -1, 1])
X_BENEFIT = np.array([[7, 9, 9, 8], [8, 7, 8, 7], [9, 6, 8, 9], [6, 7, 8, 6], [7, 8, 6, 8.5]])


# --- ranks: the bug found in third-party skills --------------------------------------------

def test_rank_is_not_argsort():
    scores = np.array([0.2, 0.9, 0.5])
    assert mc.rank(scores).tolist() == [3, 1, 2]
    assert (np.argsort(-scores) + 1).tolist() == [2, 3, 1]  # the wrong idiom
    assert mc.rank([3.0, 1.0, 2.0], higher_is_better=False).tolist() == [3, 1, 2]
    assert mc.rank([1.0, 1.0, 0.5]).tolist() == [1, 1, 3]


# --- AHP ------------------------------------------------------------------------------------

def test_ahp_consistent_matrix_recovers_weights():
    w = np.array([0.5, 0.3, 0.2])
    A = np.outer(w, 1 / w)
    for method in ("eigen", "geometric"):
        res = mc.ahp_weights(A, method)
        assert np.allclose(res["weights"], w)
        assert res["CR"] == pytest.approx(0.0, abs=1e-10)
        assert res["lambda_max"] == pytest.approx(3.0)


def test_ahp_inconsistent_matrix_flags_judgment():
    A = mc.pairwise_from_judgments(["a", "b", "c"], {("a", "b"): 3, ("b", "c"): 3, ("a", "c"): 1 / 3})
    res = mc.ahp_weights(A)
    assert res["CR"] > 0.1 and not res["consistent"]
    assert res["most_inconsistent"][0]["log_deviation"] > 1.0


def test_ahp_rejects_non_reciprocal_and_large_n_without_ri():
    with pytest.raises(ValueError):
        mc.ahp_weights(np.array([[1, 2], [2, 1]]))
    w = np.linspace(1, 2, 11)
    with pytest.raises(ValueError):
        mc.ahp_weights(np.outer(w, 1 / w))
    assert mc.ahp_weights(np.outer(w, 1 / w), ri=1.51)["CR"] == pytest.approx(0, abs=1e-9)


def test_group_aggregation():
    A1 = mc.pairwise_from_judgments("abc", {("a", "b"): 2, ("a", "c"): 4, ("b", "c"): 2})
    A2 = mc.pairwise_from_judgments("abc", {("a", "b"): 8, ("a", "c"): 4, ("b", "c"): 0.5})
    G = mc.aggregate_judgments([A1, A2])
    assert G[0, 1] == pytest.approx(4.0)  # sqrt(2*8)
    assert np.allclose(G * G.T, 1.0)  # AIJ keeps reciprocity
    p = mc.aggregate_priorities([[0.6, 0.4], [0.2, 0.8]])
    assert np.allclose(p, [0.4, 0.6])


# --- fuzzy AHP ------------------------------------------------------------------------------

def test_fuzzy_ahp_crisp_tfns_match_geometric_ahp():
    w = np.array([0.5, 0.3, 0.2])
    A = np.outer(w, 1 / w)
    F = np.repeat(A[:, :, None], 3, axis=2)
    res = mc.fuzzy_ahp_buckley(F)
    assert np.allclose(res["weights"], w)
    assert np.allclose(res["fuzzy_weights"][:, 0], res["fuzzy_weights"][:, 2])


def test_fuzzy_ahp_scale_mapping_and_order():
    A = mc.pairwise_from_judgments("abc", {("a", "b"): 3, ("a", "c"): 5, ("b", "c"): 3})
    F = mc.tfn_from_crisp(A)
    assert F[0, 1].tolist() == [2, 3, 4]
    assert np.allclose(F[1, 0], [1 / 4, 1 / 3, 1 / 2])
    res = mc.fuzzy_ahp_buckley(F)
    fw = res["fuzzy_weights"]
    assert np.all(fw[:, 0] <= fw[:, 1]) and np.all(fw[:, 1] <= fw[:, 2])
    assert np.argmax(res["weights"]) == 0 and np.argmin(res["weights"]) == 2
    assert res["weights"].sum() == pytest.approx(1.0)


# --- BWM ------------------------------------------------------------------------------------

def test_bwm_fully_consistent():
    w = np.array([0.5, 0.25, 0.125, 0.125])
    best, worst = 0, 3
    aB = w[best] / w
    aW = w / w[worst]
    res = mc.bwm_weights(aB, aW, best, worst)
    assert np.allclose(res["weights"], w, atol=1e-8)
    assert res["xi"] == pytest.approx(0.0, abs=1e-9)
    assert res["CR_input"] == pytest.approx(0.0, abs=1e-12)


def test_bwm_inconsistent_has_positive_xi():
    res = mc.bwm_weights([1, 2, 4, 8], [8, 6, 1.5, 1], 0, 3)
    assert res["xi"] > 0 and res["CR_input"] > 0
    assert np.argmax(res["weights"]) == 0 and np.argmin(res["weights"]) == 3


# --- objective weights ----------------------------------------------------------------------

def test_entropy_handles_zero_with_0log0_convention():
    w = mc.entropy_weights(np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]]))
    assert w[0] == pytest.approx(0.0)  # constant criterion carries no information
    assert w[1] == pytest.approx(1.0)


def test_roc_weights():
    assert np.allclose(mc.roc_weights([1, 2, 3]), [11 / 18, 5 / 18, 2 / 18])


# --- ranking methods: hand-checkable properties ----------------------------------------------

@pytest.mark.parametrize("fn", [mc.wsm, mc.wpm, mc.waspas, mc.topsis, mc.vikor, mc.promethee])
def test_dominating_alternative_ranks_first(fn):
    Xd = np.vstack([X, [21.0, 95, 9, 1200]])  # best on every criterion
    assert fn(Xd, W, T).ranks[-1] == 1


def test_topsis_vector_by_hand():
    Xs = np.array([[1.0, 2.0], [2.0, 1.0]])
    res = mc.topsis(Xs, [0.5, 0.5], [1, -1])
    # A2 is better on both criteria (higher benefit, lower cost) -> closeness 1 vs 0
    assert np.allclose(res.scores, [0.0, 1.0])


def test_vikor_compromise_conditions():
    res = mc.vikor(X, W, T)
    d = res.details
    assert d["DQ"] == pytest.approx(0.25)
    assert res.ranks[d["compromise_set"][0]] == 1
    assert set(d["compromise_set"]) <= set(range(X.shape[0]))
    with pytest.raises(ValueError):
        mc.vikor(np.array([[1.0, 5], [1.0, 3]]), [0.5, 0.5], [1, 1])


def test_promethee_flows_match_pymcdm_docstring_example():
    Xp = np.array([[4, 3, 2], [3, 2, 4], [5, 1, 3]], dtype=float)
    res = mc.promethee(Xp, [0.5, 0.3, 0.2], [1, 1, 1])
    assert np.allclose(res.scores, [0.1, -0.3, 0.2])
    assert np.allclose(res.details["phi_plus"] - res.details["phi_minus"], res.scores)


def test_promethee_preference_functions():
    d = np.array([-1.0, 0.0, 0.5, 1.5, 3.0])
    assert mc.preference_degree(d, "usual").tolist() == [0, 0, 1, 1, 1]
    assert mc.preference_degree(d, "ushape", q=1).tolist() == [0, 0, 0, 1, 1]
    assert np.allclose(mc.preference_degree(d, "vshape", p=2), [0, 0, 0.25, 0.75, 1])
    assert mc.preference_degree(d, "level", q=1, p=2).tolist() == [0, 0, 0, 0.5, 1]
    assert np.allclose(mc.preference_degree(d, "linear", q=1, p=2), [0, 0, 0, 0.5, 1])
    g = mc.preference_degree(d, "gaussian", s=1)
    assert g[0] == 0 and g[1] == 0 and g[3] == pytest.approx(1 - np.exp(-1.125))


def test_electre_i_kernel_simple_chain():
    Xe = np.array([[3.0, 3.0], [2.0, 2.0], [1.0, 1.0]])
    res = mc.electre_i(Xe, [0.5, 0.5], [1, 1], c_hat=0.6, d_hat=0.1)
    assert res["kernel"] == [0] and not res["graph_has_cycle"]
    assert res["outranks"][0, 1] and res["outranks"][1, 2] and not res["outranks"][2, 0]


def test_fuzzy_topsis_by_hand():
    # Two alternatives, one benefit criterion, crisp weight 1.
    F = np.array([[[2, 3, 4]], [[4, 5, 6]]], dtype=float)
    res = mc.fuzzy_topsis(F, [1.0], [1])
    r1 = np.array([2, 3, 4]) / 6
    r2 = np.array([4, 5, 6]) / 6
    dp = [np.sqrt(np.mean((r - 1) ** 2)) for r in (r1, r2)]
    dm = [np.sqrt(np.mean(r ** 2)) for r in (r1, r2)]
    expected = np.array(dm) / (np.array(dp) + np.array(dm))
    assert np.allclose(res.scores, expected)
    assert res.ranks.tolist() == [2, 1]


def test_fuzzy_topsis_cost_criterion_reverses_order():
    F = np.array([[[2, 3, 4]], [[4, 5, 6]]], dtype=float)
    assert mc.fuzzy_topsis(F, [1.0], [-1]).ranks.tolist() == [1, 2]


def test_dematel_identity_and_series():
    Z = np.array([[0, 3, 2], [1, 0, 3], [2, 1, 0]], dtype=float)
    res = mc.dematel(Z)
    N, Tm = res["N"], res["T"]
    assert np.allclose(Tm, N + N @ Tm)
    series = sum(np.linalg.matrix_power(N, k) for k in range(1, 200))
    assert np.allclose(Tm, series, atol=1e-8)
    assert np.allclose(res["prominence"], Tm.sum(1) + Tm.sum(0))
    with pytest.raises(ValueError):
        mc.dematel(np.ones((3, 3)) - np.eye(3))  # equal sums -> singular


# --- robustness tools -----------------------------------------------------------------------

def test_weight_sensitivity_endpoints():
    out = mc.weight_sensitivity(mc.topsis, X, W, T, criterion=0)
    assert out["grid"][0]["weight"] == 0.0 and out["grid"][-1]["weight"] == 1.0
    # With all weight on unit cost, the cheapest supplier (S3) must win.
    assert out["grid"][-1]["best"] == 2


def test_rank_reversal_report_structure():
    rep = mc.rank_reversal_check(mc.topsis, X, W, T)
    assert len(rep["removals"]) == X.shape[0]


def test_acceptability_rows_sum_to_one():
    acc = mc.weight_acceptability(mc.topsis, X, W, T, n_samples=200, seed=1)
    assert np.allclose(acc["rank_acceptability"].sum(axis=1), 1.0)  # one rank per alternative per draw
    assert acc["first_rank_acceptability"].sum() >= 1.0 - 1e-12
    again = mc.weight_acceptability(mc.topsis, X, W, T, n_samples=200, seed=1)
    assert np.array_equal(acc["rank_acceptability"], again["rank_acceptability"])


def test_correlations_identity_and_reversal():
    r = np.array([1, 2, 3, 4, 5])
    c = mc.compare_rankings({"a": r, "b": r[::-1]})
    assert c["spearman"][0, 1] == pytest.approx(-1.0)
    assert mc.weighted_spearman(r, r) == pytest.approx(1.0)
    assert mc.ws_coefficient(r, r) == pytest.approx(1.0)


def test_weights_must_sum_to_one():
    with pytest.raises(ValueError):
        mc.topsis(X, [0.5, 0.5, 0.5, 0.5], T)


# --- cross-checks against pymcdm (where conventions coincide) --------------------------------

pymcdm = pytest.importorskip("pymcdm", reason="pymcdm not installed; cross-checks skipped")
from pymcdm import correlations as pc  # noqa: E402
from pymcdm import methods as pm  # noqa: E402
from pymcdm import normalizations as pn  # noqa: E402
from pymcdm import weights as pw  # noqa: E402


def test_topsis_vector_matches_pymcdm_with_vector_normalization():
    ref = pm.TOPSIS(pn.vector_normalization)(X, W, T)
    assert np.allclose(mc.topsis(X, W, T).scores, ref)


def test_topsis_minmax_matches_pymcdm_default():
    ref = pm.TOPSIS()(X, W, T)
    assert np.allclose(mc.topsis(X, W, T, normalization="minmax").scores, ref)


def test_vikor_matches_pymcdm_with_minmax():
    ref = pm.VIKOR(pn.minmax_normalization)(X, W, T)
    assert np.allclose(mc.vikor(X, W, T).scores, ref)


def test_waspas_matches_pymcdm():
    assert np.allclose(mc.waspas(X, W, T).scores, pm.WASPAS()(X, W, T))


@pytest.mark.parametrize("kind,ours,kw", [
    ("usual", "usual", {}),
    ("ushape", "ushape", {"q": 2.0}),
    ("vshape", "vshape", {"p": 3.0}),
    ("level", "level", {"q": 1.0, "p": 3.0}),
    ("vshape_2", "linear", {"q": 1.0, "p": 3.0}),
])
def test_promethee_matches_pymcdm(kind, ours, kw):
    n = X_BENEFIT.shape[1]
    ref_kw = {k: np.full(n, v) for k, v in kw.items()}
    ref = pm.PROMETHEE_II(kind, **ref_kw)(X_BENEFIT, W, np.ones(n))
    res = mc.promethee(X_BENEFIT, W, np.ones(n), [dict(kind=ours, **kw)] * n)
    assert np.allclose(res.scores, ref)


def test_entropy_and_critic_match_pymcdm_on_positive_benefit_data():
    assert np.allclose(mc.entropy_weights(X_BENEFIT), pw.entropy_weights(X_BENEFIT))
    assert np.allclose(mc.critic_weights(X_BENEFIT, np.ones(4)), pw.critic_weights(X_BENEFIT))


def test_correlations_match_pymcdm():
    a = np.array([1, 2, 3, 4, 5], dtype=float)
    b = np.array([2, 1, 3, 5, 4], dtype=float)
    assert mc.weighted_spearman(a, b) == pytest.approx(pc.weighted_spearman(a, b))
    assert mc.ws_coefficient(a, b) == pytest.approx(pc.ws(a, b))


# --- CLI ------------------------------------------------------------------------------------

def test_cli_example_runs(tmp_path):
    out = tmp_path / "out.json"
    subprocess.run([sys.executable, str(SCRIPTS / "mcdm_cli.py"), str(SCRIPTS / "example_spec.json"),
                    "--out", str(out)], check=True)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert set(data["results"]) == {"topsis", "vikor", "waspas", "promethee"}
    assert abs(sum(data["weights"].values()) - 1) < 1e-9
    assert data["weight_info"]["CR"] < 0.1
    assert {"sensitivity", "rank_reversal", "acceptability", "electre_i", "ranking_agreement"} <= set(data)
    for res in data["results"].values():
        assert sorted(row["rank"] for row in res["table"])[0] == 1
