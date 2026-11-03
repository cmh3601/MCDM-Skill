"""Cross-checks against pymcdm where conventions coincide (skipped if pymcdm is absent)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import mcdm_core as mc  # noqa: E402
from test_mcdm_core import W, X, X_BENEFIT, T  # noqa: E402

pytest.importorskip("pymcdm", reason="pymcdm not installed; cross-checks skipped")
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


