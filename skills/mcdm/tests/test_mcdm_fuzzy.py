"""Tests for fuzzy VIKOR, fuzzy DEMATEL, CFCS and GRA."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import mcdm_core as mc  # noqa: E402
import mcdm_fuzzy as mf  # noqa: E402
from test_mcdm_core import T, W, X  # noqa: E402


def crisp_tfn(A):
    A = np.asarray(A, dtype=float)
    return np.repeat(A[..., None], 3, axis=-1)


def fuzzy_ratings(seed=0):
    rng = np.random.default_rng(seed)
    m = rng.uniform(1, 9, size=(5, 4))
    spread = rng.uniform(0.2, 1.5, size=(5, 4, 2))
    return np.stack([m - spread[..., 0], m, m + spread[..., 1]], axis=-1)


# --- fuzzy VIKOR ----------------------------------------------------------------------------

@pytest.mark.parametrize("types", [[1, 1, 1, 1], [-1, 1, -1, 1]])
def test_fuzzy_vikor_reduces_to_crisp_vikor(types):
    crisp = mc.vikor(X, W, types)
    fuzzy = mf.fuzzy_vikor(crisp_tfn(X), W, types)
    assert np.allclose(fuzzy.scores, crisp.scores)
    assert np.allclose(fuzzy.details["S"], crisp.details["S"])
    assert np.allclose(fuzzy.details["R"], crisp.details["R"])
    assert fuzzy.details["compromise_set"] == crisp.details["compromise_set"]


def test_fuzzy_vikor_tfns_stay_ordered_and_dominance_holds():
    F = fuzzy_ratings()
    best = F.max(axis=0) + 1.0  # dominates everyone on every bound
    F = np.concatenate([F, best[None]], axis=0)
    Wf = np.array([[0.2, 0.3, 0.4], [0.1, 0.2, 0.3], [0.2, 0.25, 0.3], [0.15, 0.25, 0.35]])
    res = mf.fuzzy_vikor(F, Wf, [1, 1, 1, 1])
    for key in ("S_fuzzy", "R_fuzzy", "Q_fuzzy"):
        Z = res.details[key]
        assert np.all(Z[:, 0] <= Z[:, 1] + 1e-12) and np.all(Z[:, 1] <= Z[:, 2] + 1e-12)
    assert res.ranks[-1] == 1


def test_tfn_arithmetic():
    assert mf.tfn_sub([3, 4, 5], [1, 2, 3]).tolist() == [0, 2, 4]
    assert mf.tfn_mul([1, 2, 3], [-1, 1, 2]).tolist() == [-3, 2, 6]


# --- fuzzy DEMATEL and CFCS ------------------------------------------------------------------

def test_fuzzy_dematel_reduces_to_crisp():
    Z = np.array([[0, 3, 2], [1, 0, 3], [2, 1, 0]], dtype=float)
    fz = mf.fuzzy_dematel(crisp_tfn(Z))
    cr = mc.dematel(Z)
    assert np.allclose(fz["T"], cr["T"])
    assert np.allclose(fz["relation"], cr["relation"])


def test_fuzzy_dematel_components_ordered():
    base = np.array([[0, 3, 2], [1, 0, 3], [2, 1, 0]], dtype=float)
    Z = np.stack([np.maximum(base - 1, 0), base, base + (base > 0)], axis=-1)
    out = mf.fuzzy_dematel(Z)
    Tf = out["T_fuzzy"]
    assert np.all(Tf[..., 0] <= Tf[..., 1]) and np.all(Tf[..., 1] <= Tf[..., 2])
    with pytest.raises(ValueError):
        mf.fuzzy_dematel(Z[..., ::-1])  # l > u


def test_cfcs_crisp_limit_and_monotonic():
    F = np.array([[2.0, 2.0, 2.0], [5.0, 5.0, 5.0]])
    assert np.allclose(mf.cfcs(F), [2.0, 5.0])
    G = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [1.0, 3.0, 4.0]])
    s = mf.cfcs(G)
    assert s[1] > s[0] and s[2] > s[0]


# --- GRA ------------------------------------------------------------------------------------

def test_gra_ideal_alternative_has_grade_one():
    Xd = np.vstack([X, [21.0, 95, 9, 1200]])
    res = mc.gra(Xd, W, T)
    assert res.scores[-1] == pytest.approx(1.0)
    assert res.ranks[-1] == 1


def test_gra_matches_pydecision_up_to_scale():
    pytest.importorskip("pyDecision.algorithm", reason="pyDecision not installed")
    from pyDecision.algorithm import gra_method
    types = ["max" if t == 1 else "min" for t in T]
    ref = gra_method(X, types, W, epsilon=0.5, graph=False, verbose=False)
    # pyDecision divides the weighted sum by the number of alternatives.
    assert np.allclose(mc.gra(X, W, T).scores, ref * X.shape[0])
