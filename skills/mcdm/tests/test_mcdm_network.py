"""Tests for ANP and DANP."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import mcdm_core as mc  # noqa: E402
import mcdm_network as net  # noqa: E402


def stationary(W):
    vals, vecs = np.linalg.eig(W)
    v = np.abs(vecs[:, np.argmin(np.abs(vals - 1))].real)
    return v / v.sum()


def test_hierarchy_as_network_reproduces_ahp_synthesis():
    elements = ["goal", "c1", "c2", "a1", "a2", "a3"]
    clusters = {"goal": ["goal"], "criteria": ["c1", "c2"], "alts": ["a1", "a2", "a3"]}
    wc = np.array([0.7, 0.3])
    a1, a2 = np.array([0.5, 0.3, 0.2]), np.array([0.1, 0.3, 0.6])
    built = net.supermatrix_from_comparisons(elements, clusters, {
        "goal": {"criteria": wc}, "c1": {"alts": a1}, "c2": {"alts": a2}})
    idx = net._cluster_index(elements, clusters)
    res = net.anp(built["unweighted"], idx, sinks="self_loop", reference=0)
    assert np.allclose(res["priorities_by_cluster"][2], wc[0] * a1 + wc[1] * a2)
    with pytest.raises(ValueError):
        net.anp(built["unweighted"], idx, sinks="self_loop")  # columns differ without reference


def test_pairwise_blocks_record_consistency():
    elements = ["c1", "c2", "c3"]
    clusters = {"criteria": elements}
    w = np.array([0.5, 0.3, 0.2])
    A = np.outer(w, 1 / w)
    comps = {e: {"criteria": A} for e in elements}
    built = net.supermatrix_from_comparisons(elements, clusters, comps)
    assert np.allclose(built["unweighted"][:, 0], w)
    assert all(c["CR"] == pytest.approx(0, abs=1e-10) for c in built["consistency"])


def test_primitive_network_limit_matches_powers_and_eigenvector():
    rng = np.random.default_rng(0)
    W = rng.random((5, 5))
    W /= W.sum(axis=0)
    lim = net.limit_supermatrix(W)
    assert not lim["cyclic"]
    assert np.allclose(lim["limit"], np.linalg.matrix_power(W, 200))
    assert np.allclose(lim["limit"][:, 2], stationary(W))


def test_cyclic_network_uses_cesaro_limit():
    # criteria <-> alternatives only: period 2, plain powers oscillate
    B = np.array([[0.6, 0.2, 0.3], [0.4, 0.8, 0.7]])          # criteria rows, alt columns
    A = np.array([[0.5, 0.1], [0.3, 0.3], [0.2, 0.6]])          # alt rows, criteria columns
    W = np.block([[np.zeros((2, 2)), B], [A, np.zeros((3, 3))]])
    idx = [np.array([0, 1]), np.array([2, 3, 4])]
    res = net.anp(W, idx)
    assert res["cyclic"]
    P = np.linalg.matrix_power(W, 200)
    assert not np.allclose(P, P @ W)
    L = res["limit"]
    assert np.allclose(W @ L, L)
    assert np.allclose(L[:, 0], stationary(W))
    cesaro = sum(np.linalg.matrix_power(W, k) for k in range(1, 2001)) / 2000
    assert np.allclose(L, cesaro, atol=1e-3)


def test_cluster_weights_make_columns_stochastic():
    W = np.array([[0, .5, .4, .3], [0, .5, .6, .7], [.6, .2, 0, 0], [.4, .8, 0, 0]])
    W[0:2, 0] = [0.0, 0.0]
    W[:, 0] = [0, 0, .6, .4]
    idx = [np.array([0, 1]), np.array([2, 3])]
    Wt = net.weighted_supermatrix(W, idx, cluster_weights=[[0.3, 1.0], [0.7, 0.0]])
    assert np.allclose(Wt.sum(axis=0), 1.0)
    with pytest.raises(ValueError):
        net.weighted_supermatrix(np.zeros((2, 2)), [np.array([0, 1])])


def test_danp_properties():
    Z = np.array([
        [0, 3, 2, 1, 2],
        [2, 0, 3, 2, 1],
        [1, 2, 0, 3, 2],
        [3, 1, 2, 0, 3],
        [2, 2, 1, 2, 0],
    ], dtype=float)
    out = net.danp(Z, [[0, 1], [2, 3, 4]])
    assert np.allclose(out["W_alpha"].sum(axis=0), 1.0)
    assert out["weights"].sum() == pytest.approx(1.0)
    assert out["dimension_weights"].sum() == pytest.approx(1.0)
    assert np.allclose(out["limit"], out["limit"][:, [0]])
    assert np.allclose(out["T_D"][0, 1], out["T_C"][np.ix_([0, 1], [2, 3, 4])].mean())


def test_danp_one_criterion_per_dimension_reduces_to_stationary_distribution():
    Z = np.array([[0, 3, 1], [2, 0, 4], [1, 2, 0]], dtype=float)
    out = net.danp(Z, [[0], [1], [2]])
    T = mc.dematel(Z)["T"]
    Ta = T / T.sum(axis=1, keepdims=True)
    assert np.allclose(out["weights"], stationary(Ta.T))


def test_limit_matches_pydecision_power_method_on_primitive_matrix():
    pyd = pytest.importorskip("pyDecision.algorithm.anp", reason="pyDecision not installed")
    rng = np.random.default_rng(3)
    W = rng.random((6, 6))
    W /= W.sum(axis=0)
    ref = pyd.anp_method(W, max_iter=500, tol=1e-14, cesaro=False)
    assert np.allclose(net.limit_supermatrix(W)["limit"], ref, atol=1e-9)
