"""Network methods: ANP (Saaty, 1996) and DEMATEL-based ANP (DANP).

Supermatrix convention: column j holds the influence of element j on every
element (its priorities over each cluster), so the weighted supermatrix is
column-stochastic and priorities are read from columns of the limit matrix.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.linalg import null_space

from mcdm_core import ahp_weights, dematel


def _cluster_index(elements: Sequence[str], clusters: dict) -> list[np.ndarray]:
    pos = {e: i for i, e in enumerate(elements)}
    seen = [e for members in clusters.values() for e in members]
    if sorted(seen) != sorted(elements):
        raise ValueError("Clusters must partition the element list exactly.")
    return [np.array([pos[e] for e in members]) for members in clusters.values()]


def supermatrix_from_comparisons(elements: Sequence[str], clusters: dict, comparisons: dict,
                                 ri: float | None = None) -> dict:
    """Unweighted supermatrix from local judgments.

    comparisons[control][cluster] is either a reciprocal pairwise matrix over the
    cluster's elements (AHP eigenvector is used and its CR recorded) or a priority
    vector. Missing entries mean 'no influence' (zero block).
    """
    elements = list(elements)
    idx = _cluster_index(elements, clusters)
    names = list(clusters)
    pos = {e: i for i, e in enumerate(elements)}
    n = len(elements)
    W = np.zeros((n, n))
    consistency = []
    for control, by_cluster in comparisons.items():
        j = pos[control]
        for cname, data in by_cluster.items():
            rows = idx[names.index(cname)]
            arr = np.atleast_1d(np.asarray(data, dtype=float))
            if arr.ndim == 2:
                res = ahp_weights(arr, ri=ri) if arr.shape[0] > 2 else ahp_weights(arr, ri=0.0)
                vec = res["weights"]
                consistency.append({"control": control, "cluster": cname, "CR": res["CR"],
                                    "consistent": res["consistent"]})
            else:
                if np.any(arr < 0) or arr.sum() <= 0:
                    raise ValueError(f"Priority vector for {control}->{cname} must be non-negative.")
                vec = arr / arr.sum()
            if vec.size != rows.size:
                raise ValueError(f"{control}->{cname}: expected {rows.size} priorities, got {vec.size}.")
            W[rows, j] = vec
    return {"unweighted": W, "elements": elements, "consistency": consistency}


def weighted_supermatrix(W, cluster_index: Sequence[np.ndarray], cluster_weights=None,
                         sinks: str = "error") -> np.ndarray:
    """Make the supermatrix column-stochastic.

    cluster_weights[r, c] = importance of row-cluster r for elements of column-cluster c
    (columns of this K x K matrix are renormalised over the non-empty blocks of each column).
    sinks: "error" (default) or "self_loop" to put 1 on the diagonal of all-zero columns,
    the usual treatment of alternatives in a hierarchy written as a network.
    """
    W = np.array(W, dtype=float)
    n = W.shape[0]
    if W.shape != (n, n) or np.any(W < 0):
        raise ValueError("Supermatrix must be square and non-negative.")
    K = len(cluster_index)
    cluster_of = np.empty(n, dtype=int)
    for k, members in enumerate(cluster_index):
        cluster_of[members] = k
    out = W.copy()
    if cluster_weights is not None:
        Cw = np.asarray(cluster_weights, dtype=float)
        if Cw.shape != (K, K):
            raise ValueError(f"cluster_weights must be {K} x {K}.")
        for j in range(n):
            for r in range(K):
                out[cluster_index[r], j] *= Cw[r, cluster_of[j]]
    zero = out.sum(axis=0) == 0
    if np.any(zero):
        if sinks != "self_loop":
            raise ValueError(f"Columns {np.where(zero)[0].tolist()} are all zero (sink elements); "
                             "pass sinks='self_loop' or add their influences.")
        out[zero, zero] = 1.0
    return out / out.sum(axis=0)


def limit_supermatrix(Wt, tol: float = 1e-9) -> dict:
    """Cesàro limit of a column-stochastic matrix, computed exactly.

    For stochastic matrices the eigenvalue 1 is semisimple, so the Cesàro limit
    (1/N) Σ W^k is the spectral projector onto its eigenspace: L = V (UᵀV)⁻¹ Uᵀ,
    with V and U right and left null spaces of W − I. This also covers cyclic
    (periodic) supermatrices, for which plain powers oscillate.
    """
    Wt = np.asarray(Wt, dtype=float)
    n = Wt.shape[0]
    if not np.allclose(Wt.sum(axis=0), 1.0, atol=1e-9):
        raise ValueError("Weighted supermatrix must be column-stochastic.")
    I = np.eye(n)
    V = null_space(Wt - I, rcond=1e-10)
    U = null_space((Wt - I).T, rcond=1e-10)
    L = V @ np.linalg.solve(U.T @ V, U.T)
    L[np.abs(L) < 1e-15] = 0.0
    eig = np.linalg.eigvals(Wt)
    on_circle = np.sum(np.abs(np.abs(eig) - 1) < 1e-8)
    cyclic = bool(on_circle > V.shape[1])
    ok = np.allclose(Wt @ L, L, atol=tol) and np.allclose(L @ L, L, atol=tol)
    if not ok:
        raise RuntimeError("Limit matrix check failed; inspect the supermatrix for numerical issues.")
    return {"limit": L, "cyclic": cyclic, "multiplicity_of_eigenvalue_1": int(V.shape[1])}


def anp(W, cluster_index: Sequence[np.ndarray], cluster_weights=None, sinks: str = "error",
        reference: int | None = None) -> dict:
    """Full ANP: weighted supermatrix, limit matrix and priorities.

    If all columns of the limit matrix agree (irreducible network) the common
    column is returned. Otherwise pass `reference`, the index of the element whose
    column carries the synthesis (e.g. the goal node of a hierarchy).
    """
    Wt = weighted_supermatrix(W, cluster_index, cluster_weights, sinks)
    lim = limit_supermatrix(Wt)
    L = lim["limit"]
    if reference is None:
        if not np.allclose(L, L[:, [0]], atol=1e-8):
            raise ValueError("Limit-matrix columns differ (reducible network); pass reference=<column index>.")
        pr = L[:, 0]
    else:
        pr = L[:, reference]
    by_cluster = []
    for members in cluster_index:
        s = pr[members].sum()
        by_cluster.append(pr[members] / s if s > 0 else np.zeros(members.size))
    return {"weighted": Wt, "limit": L, "priorities": pr, "priorities_by_cluster": by_cluster,
            "cyclic": lim["cyclic"]}


def danp(Z, groups: Sequence[Sequence[int]], threshold: float | None = None) -> dict:
    """DEMATEL-based ANP (DANP; Ou Yang et al., 2008).

    Z: criteria-level direct-relation matrix (n x n) or expert stack (k, n, n).
    groups: dimensions as lists of criterion indices (a partition of 0..n-1).
    Returns influential weights of criteria and dimensions plus all intermediate matrices.
    """
    dm = dematel(Z, threshold)
    Tc = dm["T"]
    n = Tc.shape[0]
    idx = [np.asarray(g, dtype=int) for g in groups]
    if sorted(np.concatenate(idx).tolist()) != list(range(n)):
        raise ValueError("groups must partition the criteria indices.")
    K = len(idx)
    # T_C^alpha: normalise every row within each dimension block.
    Tc_alpha = np.zeros_like(Tc)
    for gi in idx:
        for gj in idx:
            block = Tc[np.ix_(gi, gj)]
            Tc_alpha[np.ix_(gi, gj)] = block / block.sum(axis=1, keepdims=True)
    Wc = Tc_alpha.T  # unweighted supermatrix
    Td = np.array([[Tc[np.ix_(gi, gj)].mean() for gj in idx] for gi in idx])
    Td_alpha = Td / Td.sum(axis=1, keepdims=True)
    Wa = np.zeros_like(Wc)
    for i, gi in enumerate(idx):
        for j, gj in enumerate(idx):
            # Block (i, j) carries how dimension j influences dimension i.
            Wa[np.ix_(gi, gj)] = Td_alpha[j, i] * Wc[np.ix_(gi, gj)]
    lim = limit_supermatrix(Wa)
    L = lim["limit"]
    w = L[:, 0] / L[:, 0].sum()
    return {"weights": w, "dimension_weights": np.array([w[g].sum() for g in idx]),
            "T_C": Tc, "T_C_alpha": Tc_alpha, "W_C": Wc, "T_D": Td, "T_D_alpha": Td_alpha,
            "W_alpha": Wa, "limit": L, "dematel": dm, "K": K}
