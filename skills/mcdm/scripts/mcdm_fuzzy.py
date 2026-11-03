"""Fuzzy VIKOR, fuzzy DEMATEL and CFCS defuzzification (fuzzy AHP and fuzzy TOPSIS are in mcdm_core).

Triangular fuzzy numbers (TFN) are arrays (..., 3) holding (l, m, u).
Each method reduces exactly to its crisp counterpart when l = m = u; the tests check this.
"""

from __future__ import annotations

import numpy as np

from mcdm_core import BENEFIT, Result, parse_types, rank, vikor_compromise


def tfn_mul(a, b) -> np.ndarray:
    """Approximate TFN product valid for signed bounds: (min of bound products, m_a m_b, max of bound products)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    prods = np.stack([a[..., 0] * b[..., 0], a[..., 0] * b[..., 2], a[..., 2] * b[..., 0], a[..., 2] * b[..., 2]])
    return np.stack([prods.min(axis=0), a[..., 1] * b[..., 1], prods.max(axis=0)], axis=-1)


def tfn_sub(a, b) -> np.ndarray:
    """a ⊖ b = (l_a − u_b, m_a − m_b, u_a − l_b)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.stack([a[..., 0] - b[..., 2], a[..., 1] - b[..., 1], a[..., 2] - b[..., 0]], axis=-1)


def defuzzify(F, method: str = "centroid") -> np.ndarray:
    F = np.asarray(F, dtype=float)
    if method == "centroid":
        return F.mean(axis=-1)
    if method == "graded_mean":
        return (F[..., 0] + 4 * F[..., 1] + F[..., 2]) / 6
    if method == "cfcs":
        return cfcs(F)
    raise ValueError("method must be centroid, graded_mean or cfcs.")


def cfcs(F) -> np.ndarray:
    """Converting Fuzzy data into Crisp Scores (Opricovic & Tzeng, 2003), using the global
    min l and max u of the whole array as the normalisation range."""
    F = np.asarray(F, dtype=float)
    lo, hi = F[..., 0].min(), F[..., 2].max()
    if hi == lo:
        return F[..., 1].copy()
    span = hi - lo
    xl, xm, xu = ((F[..., k] - lo) / span for k in range(3))
    ls = xm / (1 + xm - xl)
    us = xu / (1 + xu - xm)
    x = (ls * (1 - ls) + us * us) / (1 - ls + us)
    return lo + x * span


def fuzzy_vikor(F, weights, types, v: float = 0.5, defuzz: str = "centroid") -> Result:
    """Fuzzy VIKOR with TFN arithmetic, following the structure of Opricovic (2011).

    F: (m, n, 3) TFN ratings; weights: crisp (n,) or TFN (n, 3).
    Fuzzy best/worst per criterion are component-wise max/min (benefit) or min/max (cost);
    d_ij = (f*_j ⊖ x_ij) / (u*_j − l°_j) for benefit and (x_ij ⊖ f*_j) / (u°_j − l*_j) for cost.
    S, R and Q are fuzzy; they are defuzzified (centroid by default) before ranking and
    before checking the compromise conditions. Lower Q is better.
    """
    F = np.asarray(F, dtype=float)
    m, n, _ = F.shape
    t = parse_types(types, n)
    Wt = np.asarray(weights, dtype=float)
    if Wt.shape == (n,):
        Wt = np.repeat(Wt[:, None], 3, axis=1)
    if Wt.shape != (n, 3):
        raise ValueError("weights must have shape (n,) or (n, 3).")
    best = np.where((t == BENEFIT)[:, None], F.max(axis=0), F.min(axis=0))   # (n, 3)
    worst = np.where((t == BENEFIT)[:, None], F.min(axis=0), F.max(axis=0))
    d = np.empty_like(F)
    for j in range(n):
        if t[j] == BENEFIT:
            span = best[j, 2] - worst[j, 0]
            diff = tfn_sub(best[j], F[:, j])
        else:
            span = worst[j, 2] - best[j, 0]
            diff = tfn_sub(F[:, j], best[j])
        if span <= 0:
            raise ValueError(f"Criterion {j} does not vary across alternatives; drop it for VIKOR.")
        d[:, j] = diff / span
    G = tfn_mul(Wt[None, :, :], d)       # (m, n, 3)
    S = G.sum(axis=1)                     # (m, 3)
    R = G.max(axis=1)                     # component-wise max

    def term(Z):
        z_star = Z.min(axis=0)
        span = Z[:, 2].max() - z_star[0]
        return tfn_sub(Z, z_star) / span if span > 0 else np.zeros_like(Z)

    Qf = v * term(S) + (1 - v) * term(R)
    Qc, Sc, Rc = (defuzzify(Z, defuzz) for Z in (Qf, S, R))
    details = {"S_fuzzy": S, "R_fuzzy": R, "Q_fuzzy": Qf, "S": Sc, "R": Rc, "v": v, "defuzzification": defuzz}
    details.update(vikor_compromise(Qc, Sc, Rc))
    return Result("Fuzzy VIKOR", Qc, rank(Qc, higher_is_better=False), False, details)


def fuzzy_dematel(Z, defuzz: str = "centroid", threshold: float | None = None) -> dict:
    """Fuzzy DEMATEL in the manner of Lin & Wu (2008).

    Z: (n, n, 3) TFN direct-relation matrix or (k, n, n, 3) expert stack (averaged).
    The l, m and u matrices are normalised by the same constant (max of row and column sums
    of u), total-relation matrices are computed per component, and the fuzzy total relation
    is defuzzified before computing D, R, prominence and relation.
    """
    Z = np.asarray(Z, dtype=float)
    if Z.ndim == 4:
        Z = Z.mean(axis=0)
    n = Z.shape[0]
    if Z.shape != (n, n, 3) or np.any(Z < 0) or np.any(np.diff(Z, axis=-1) < 0):
        raise ValueError("Z must be (n, n, 3) non-negative TFNs with l <= m <= u.")
    s = max(Z[..., 2].sum(axis=1).max(), Z[..., 2].sum(axis=0).max())
    I = np.eye(n)
    Tf = np.empty_like(Z)
    for k in range(3):
        Nk = Z[..., k] / s
        Tf[..., k] = Nk @ np.linalg.inv(I - Nk)
    T = defuzzify(Tf, defuzz)
    Dr, Rc = T.sum(axis=1), T.sum(axis=0)
    alpha = float(T.mean()) if threshold is None else threshold
    return {"T_fuzzy": Tf, "T": T, "D": Dr, "R": Rc, "prominence": Dr + Rc, "relation": Dr - Rc,
            "threshold": alpha, "defuzzification": defuzz,
            "influence_edges": [(int(i), int(j), float(T[i, j])) for i, j in zip(*np.where(T > alpha))]}
