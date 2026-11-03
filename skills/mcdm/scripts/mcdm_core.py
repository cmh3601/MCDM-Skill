"""Reference implementations of common MCDM methods (numpy + scipy only).

Conventions used by every function in this module
-------------------------------------------------
X      decision matrix, shape (m alternatives, n criteria), raw performance values
w      crisp criterion weights, shape (n,), non-negative and summing to 1
types  criterion directions, shape (n,): +1 / "benefit" / "max" or -1 / "cost" / "min"
ranks  integer ranks where 1 is the best alternative; ties share the lowest rank

Ranking methods return a ``Result`` whose ``scores`` keep the method's own
orientation (``higher_is_better`` says which way) and whose ``ranks`` are
always 1 = best. Never derive ranks with ``np.argsort(...) + 1``: that gives
the index of the k-th best alternative, not the rank of alternative i.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np
from scipy import optimize, stats

BENEFIT, COST = 1, -1

# Saaty's random consistency index for n = 1..10 (Saaty, 1980).
SAATY_RI = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}

# Triangular fuzzy numbers for Saaty's 1-9 scale. This is one common mapping,
# not a standard: report whichever scale you use.
SAATY_TFN = {
    1: (1, 1, 1), 2: (1, 2, 3), 3: (2, 3, 4), 4: (3, 4, 5), 5: (4, 5, 6),
    6: (5, 6, 7), 7: (6, 7, 8), 8: (7, 8, 9), 9: (8, 9, 9),
}


@dataclass
class Result:
    method: str
    scores: np.ndarray
    ranks: np.ndarray
    higher_is_better: bool
    details: dict = field(default_factory=dict)

    def to_dict(self, alternatives: Sequence[str] | None = None) -> dict:
        names = list(alternatives) if alternatives is not None else [f"A{i + 1}" for i in range(len(self.scores))]
        return {
            "method": self.method,
            "higher_is_better": self.higher_is_better,
            "table": [
                {"alternative": a, "score": float(s), "rank": int(r)}
                for a, s, r in sorted(zip(names, self.scores, self.ranks), key=lambda t: t[2])
            ],
            "details": _jsonable(self.details),
        }


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


# ---------------------------------------------------------------------------
# Validation and ranking helpers
# ---------------------------------------------------------------------------

def parse_types(types: Iterable, n: int) -> np.ndarray:
    mapping = {"benefit": BENEFIT, "max": BENEFIT, "profit": BENEFIT, "cost": COST, "min": COST}
    out = []
    for t in types:
        if isinstance(t, str):
            if t.lower() not in mapping:
                raise ValueError(f"Unknown criterion type {t!r}; use benefit/cost.")
            out.append(mapping[t.lower()])
        elif t in (1, -1):
            out.append(int(t))
        else:
            raise ValueError(f"Unknown criterion type {t!r}; use +1/-1.")
    out = np.array(out, dtype=int)
    if out.shape != (n,):
        raise ValueError(f"Expected {n} criterion types, got {out.shape[0]}.")
    return out


def check_weights(w, n: int, tol: float = 1e-6) -> np.ndarray:
    w = np.asarray(w, dtype=float)
    if w.shape != (n,):
        raise ValueError(f"Expected {n} weights, got shape {w.shape}.")
    if np.any(w < 0) or not np.all(np.isfinite(w)):
        raise ValueError("Weights must be finite and non-negative.")
    if abs(w.sum() - 1.0) > tol:
        raise ValueError(f"Weights must sum to 1 (got {w.sum():.6f}); normalise them explicitly.")
    return w


def check_inputs(X, w, types):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] < 2:
        raise ValueError("X must be a 2-D matrix with at least two alternatives.")
    if not np.all(np.isfinite(X)):
        raise ValueError("X contains NaN or infinite values; impute or remove them first.")
    n = X.shape[1]
    return X, check_weights(w, n), parse_types(types, n)


def rank(scores, higher_is_better: bool = True) -> np.ndarray:
    """Ranks with 1 = best. Ties share the smallest rank ("min" method)."""
    s = np.asarray(scores, dtype=float)
    return stats.rankdata(-s if higher_is_better else s, method="min").astype(int)


# ---------------------------------------------------------------------------
# Normalisation (benefit-oriented: larger normalised value is always better)
# ---------------------------------------------------------------------------

def normalize(X, types, method: str = "minmax") -> np.ndarray:
    X = np.asarray(X, dtype=float)
    t = parse_types(types, X.shape[1])
    R = np.empty_like(X)
    for j in range(X.shape[1]):
        x, cost = X[:, j], t[j] == COST
        if method == "vector":
            norm = np.sqrt(np.sum(x ** 2))
            if norm == 0:
                raise ValueError(f"Criterion {j} is all zeros; vector normalisation undefined.")
            # 1 - r for cost keeps Euclidean distances identical to the textbook
            # treatment (cost ideal = min r), because it is an affine flip.
            R[:, j] = 1 - x / norm if cost else x / norm
        elif method == "minmax":
            lo, hi = x.min(), x.max()
            if hi == lo:
                R[:, j] = 1.0  # criterion does not discriminate
            else:
                R[:, j] = (hi - x) / (hi - lo) if cost else (x - lo) / (hi - lo)
        elif method in ("max", "linear"):
            if np.any(x <= 0):
                raise ValueError(f"Criterion {j}: linear (max) normalisation needs positive values.")
            R[:, j] = x.min() / x if cost else x / x.max()
        elif method == "sum":
            if np.any(x <= 0):
                raise ValueError(f"Criterion {j}: sum normalisation needs positive values.")
            R[:, j] = (1 / x) / np.sum(1 / x) if cost else x / x.sum()
        else:
            raise ValueError(f"Unknown normalisation {method!r}.")
    return R


# ---------------------------------------------------------------------------
# Weighting methods
# ---------------------------------------------------------------------------

def pairwise_from_judgments(items: Sequence[str], judgments: dict) -> np.ndarray:
    """Build a reciprocal matrix from {(a, b): value} meaning 'a is value times as important as b'."""
    idx = {name: i for i, name in enumerate(items)}
    A = np.ones((len(items), len(items)))
    for (a, b), v in judgments.items():
        if v <= 0:
            raise ValueError(f"Judgment {(a, b)} must be positive.")
        A[idx[a], idx[b]] = v
        A[idx[b], idx[a]] = 1.0 / v
    return A


def _check_reciprocal(A: np.ndarray, tol: float = 1e-6) -> None:
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Pairwise matrix must be square.")
    if np.any(A <= 0):
        raise ValueError("Pairwise matrix entries must be positive.")
    if not np.allclose(A * A.T, 1.0, atol=tol):
        raise ValueError("Pairwise matrix is not reciprocal (a_ij * a_ji != 1).")


def ahp_weights(A, method: str = "eigen", ri: float | None = None) -> dict:
    """AHP priorities and consistency (Saaty, 1980).

    method: "eigen" (principal right eigenvector) or "geometric" (row geometric mean).
    ri:     random index; defaults to Saaty's table for n <= 10, required for n > 10.
    """
    A = np.asarray(A, dtype=float)
    _check_reciprocal(A)
    n = A.shape[0]
    if method == "eigen":
        vals, vecs = np.linalg.eig(A)
        k = int(np.argmax(vals.real))
        w = np.abs(vecs[:, k].real)
        lam = float(vals[k].real)
    elif method == "geometric":
        w = np.exp(np.mean(np.log(A), axis=1))
        lam = float(np.mean((A @ (w / w.sum())) / (w / w.sum())))
    else:
        raise ValueError("method must be 'eigen' or 'geometric'.")
    w = w / w.sum()
    ci = (lam - n) / (n - 1) if n > 2 else 0.0
    if ri is None:
        if n not in SAATY_RI:
            raise ValueError(f"No built-in random index for n={n}; pass ri= from your chosen RI table.")
        ri = SAATY_RI[n]
    cr = ci / ri if ri > 0 else 0.0
    # Saaty's error matrix: eps_ij = a_ij * w_j / w_i equals 1 for perfectly consistent judgments.
    eps = A * np.outer(1 / w, w)
    iu = np.triu_indices(n, 1)
    dev = np.abs(np.log(eps[iu]))
    order = np.argsort(-dev)
    worst = [
        {"i": int(iu[0][k]), "j": int(iu[1][k]), "given": float(A[iu[0][k], iu[1][k]]),
         "implied": float(w[iu[0][k]] / w[iu[1][k]]), "log_deviation": float(dev[k])}
        for k in order[: min(3, len(order))]
    ]
    return {"weights": w, "lambda_max": lam, "CI": ci, "RI": ri, "CR": cr,
            "consistent": bool(cr < 0.10), "most_inconsistent": worst}


def aggregate_judgments(matrices: Sequence, expert_weights=None) -> np.ndarray:
    """AIJ: element-wise weighted geometric mean of reciprocal matrices (Forman & Peniwati, 1998)."""
    M = np.asarray(matrices, dtype=float)
    k = M.shape[0]
    a = np.full(k, 1 / k) if expert_weights is None else np.asarray(expert_weights, dtype=float)
    a = a / a.sum()
    return np.exp(np.tensordot(a, np.log(M), axes=1))


def aggregate_priorities(weight_vectors: Sequence, expert_weights=None, mean: str = "arithmetic") -> np.ndarray:
    """AIP: combine individual priority vectors (Forman & Peniwati, 1998)."""
    W = np.asarray(weight_vectors, dtype=float)
    k = W.shape[0]
    a = np.full(k, 1 / k) if expert_weights is None else np.asarray(expert_weights, dtype=float)
    a = a / a.sum()
    agg = a @ W if mean == "arithmetic" else np.exp(a @ np.log(W))
    return agg / agg.sum()


def tfn_from_crisp(A, scale: dict = SAATY_TFN) -> np.ndarray:
    """Map a crisp Saaty matrix (integers 1-9 and their reciprocals) to TFNs of shape (n, n, 3)."""
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    F = np.ones((n, n, 3))
    for i in range(n):
        for j in range(n):
            v = A[i, j]
            if v >= 1:
                key = int(round(v))
                if abs(v - key) > 1e-9 or key not in scale:
                    raise ValueError(f"a[{i},{j}]={v} is not on the 1-9 scale.")
                F[i, j] = scale[key]
            else:
                key = int(round(1 / v))
                if abs(1 / v - key) > 1e-9 or key not in scale:
                    raise ValueError(f"a[{i},{j}]={v} is not a reciprocal of the 1-9 scale.")
                l, m, u = scale[key]
                F[i, j] = (1 / u, 1 / m, 1 / l)
    return F


def fuzzy_ahp_buckley(F, ri: float | None = None) -> dict:
    """Fuzzy AHP with Buckley's (1985) geometric-mean method.

    F: (n, n, 3) reciprocal TFN matrix, F[j, i] = (1/u_ij, 1/m_ij, 1/l_ij).
    Returns fuzzy weights, centroid-defuzzified crisp weights, and the CR of the modal matrix.
    Prefer this over Chang's extent analysis, which can assign zero weights
    (Wang, Luo & Hua, 2008).
    """
    F = np.asarray(F, dtype=float)
    n = F.shape[0]
    if F.shape != (n, n, 3):
        raise ValueError("F must have shape (n, n, 3).")
    if np.any(F[..., 0] > F[..., 1]) or np.any(F[..., 1] > F[..., 2]):
        raise ValueError("Each TFN must satisfy l <= m <= u.")
    recip = np.stack([1 / F[..., 2], 1 / F[..., 1], 1 / F[..., 0]], axis=-1).transpose(1, 0, 2)
    if not np.allclose(F, recip, atol=1e-6):
        raise ValueError("TFN matrix is not reciprocal.")
    r = np.exp(np.mean(np.log(F), axis=1))  # (n, 3) fuzzy geometric means of rows
    total = r.sum(axis=0)
    fuzzy_w = np.stack([r[:, 0] / total[2], r[:, 1] / total[1], r[:, 2] / total[0]], axis=1)
    crisp = fuzzy_w.mean(axis=1)
    crisp = crisp / crisp.sum()
    modal = ahp_weights(F[..., 1], method="eigen", ri=ri)
    return {"fuzzy_weights": fuzzy_w, "weights": crisp, "modal_CR": modal["CR"],
            "modal_consistent": modal["consistent"]}


def bwm_weights(best_to_others, others_to_worst, best: int, worst: int) -> dict:
    """Linear Best-Worst Method (Rezaei, 2016).

    best_to_others[j]  = preference of the best criterion over j (a_Bj, 1-9)
    others_to_worst[j] = preference of j over the worst criterion (a_jW, 1-9)
    Returns weights, xi (0 means fully consistent) and the input-based
    consistency ratio CR_I of Liang, Brunelli & Rezaei (2020).
    """
    aB = np.asarray(best_to_others, dtype=float)
    aW = np.asarray(others_to_worst, dtype=float)
    n = aB.size
    if aW.size != n or aB[best] != 1 or aW[worst] != 1:
        raise ValueError("Vectors must have equal length with a_BB = 1 and a_WW = 1.")
    if abs(aB[worst] - aW[best]) > 1e-9:
        raise ValueError("a_BW must be the same in both vectors.")
    # variables: w_0..w_{n-1}, xi ; minimise xi
    c = np.zeros(n + 1)
    c[-1] = 1.0
    A_ub, b_ub = [], []
    for j in range(n):
        for sign in (1, -1):
            row = np.zeros(n + 1)
            row[best] += sign
            row[j] -= sign * aB[j]
            row[-1] = -1
            A_ub.append(row)
            b_ub.append(0.0)
            row = np.zeros(n + 1)
            row[j] += sign
            row[worst] -= sign * aW[j]
            row[-1] = -1
            A_ub.append(row)
            b_ub.append(0.0)
    A_eq = np.zeros((1, n + 1))
    A_eq[0, :n] = 1
    res = optimize.linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), A_eq=A_eq, b_eq=[1.0],
                           bounds=[(0, None)] * (n + 1), method="highs")
    if not res.success:
        raise RuntimeError(f"BWM LP failed: {res.message}")
    w, xi = res.x[:n], float(res.x[-1])
    aBW = aB[worst]
    if aBW > 1:
        cr_j = np.abs(aB * aW - aBW) / (aBW * aBW - aBW)
        cr_input = float(cr_j.max())
    else:
        cr_input = 0.0
    return {"weights": w / w.sum(), "xi": xi, "CR_input": cr_input}


def entropy_weights(X) -> np.ndarray:
    """Shannon-entropy objective weights; uses the convention 0 * ln 0 = 0."""
    X = np.asarray(X, dtype=float)
    if np.any(X < 0):
        raise ValueError("Entropy weights need non-negative data; shift or normalise first.")
    col = X.sum(axis=0)
    if np.any(col == 0):
        raise ValueError("A criterion column sums to zero.")
    P = X / col
    with np.errstate(divide="ignore", invalid="ignore"):
        plogp = np.where(P > 0, P * np.log(P), 0.0)
    e = -plogp.sum(axis=0) / np.log(X.shape[0])
    d = 1 - e
    if d.sum() <= 0:
        raise ValueError("All criteria have maximal entropy (no variation); weights undefined.")
    return d / d.sum()


def critic_weights(X, types) -> np.ndarray:
    """CRITIC weights (Diakoulaki et al., 1995), normalising each criterion by its own direction."""
    X = np.asarray(X, dtype=float)
    R = normalize(X, types, "minmax")
    sd = R.std(axis=0, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(R, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    C = sd * np.sum(1 - corr, axis=0)
    if C.sum() <= 0:
        raise ValueError("No criterion varies; CRITIC weights undefined.")
    return C / C.sum()


def std_weights(X, types) -> np.ndarray:
    R = normalize(np.asarray(X, dtype=float), types, "minmax")
    sd = R.std(axis=0, ddof=1)
    return sd / sd.sum()


def roc_weights(order: Sequence[int]) -> np.ndarray:
    """Rank-order-centroid weights (Barron & Barrett, 1996). order[j] = importance rank of criterion j (1 = most)."""
    order = np.asarray(order, dtype=int)
    n = order.size
    w = np.array([np.sum(1 / np.arange(k, n + 1)) / n for k in order])
    return w / w.sum()


def combine_weights(subjective, objective, alpha: float = 0.5, mode: str = "linear") -> np.ndarray:
    """Hybrid weights: linear alpha*s + (1-alpha)*o, or multiplicative s*o renormalised."""
    s, o = np.asarray(subjective, float), np.asarray(objective, float)
    w = alpha * s + (1 - alpha) * o if mode == "linear" else s * o
    return w / w.sum()


# ---------------------------------------------------------------------------
# Ranking methods
# ---------------------------------------------------------------------------

def wsm(X, w, types, normalization: str = "max") -> Result:
    X, w, t = check_inputs(X, w, types)
    s = normalize(X, t, normalization) @ w
    return Result("WSM", s, rank(s), True, {"normalization": normalization})


def wpm(X, w, types) -> Result:
    X, w, t = check_inputs(X, w, types)
    s = np.prod(normalize(X, t, "max") ** w, axis=1)
    return Result("WPM", s, rank(s), True, {"normalization": "max"})


def waspas(X, w, types, lam: float = 0.5) -> Result:
    """WASPAS (Zavadskas et al., 2012) with linear (max) normalisation."""
    X, w, t = check_inputs(X, w, types)
    R = normalize(X, t, "max")
    q1, q2 = R @ w, np.prod(R ** w, axis=1)
    s = lam * q1 + (1 - lam) * q2
    return Result("WASPAS", s, rank(s), True, {"lambda": lam, "WSM": q1, "WPM": q2})


def topsis(X, w, types, normalization: str = "vector") -> Result:
    """TOPSIS (Hwang & Yoon, 1981). Vector normalisation is the textbook default."""
    X, w, t = check_inputs(X, w, types)
    if normalization == "vector":
        norm = np.sqrt((X ** 2).sum(axis=0))
        if np.any(norm == 0):
            raise ValueError("A criterion is all zeros.")
        V = X / norm * w
        ideal = np.where(t == BENEFIT, V.max(axis=0), V.min(axis=0))
        anti = np.where(t == BENEFIT, V.min(axis=0), V.max(axis=0))
    else:
        V = normalize(X, t, normalization) * w
        ideal, anti = V.max(axis=0), V.min(axis=0)
    d_plus = np.sqrt(((V - ideal) ** 2).sum(axis=1))
    d_minus = np.sqrt(((V - anti) ** 2).sum(axis=1))
    denom = d_plus + d_minus
    cc = np.divide(d_minus, denom, out=np.full_like(denom, 0.5), where=denom > 0)
    return Result("TOPSIS", cc, rank(cc), True,
                  {"normalization": normalization, "d_plus": d_plus, "d_minus": d_minus})


def vikor(X, w, types, v: float = 0.5) -> Result:
    """VIKOR (Opricovic & Tzeng, 2004). Lower Q is better; details hold the compromise set."""
    X, w, t = check_inputs(X, w, types)
    m = X.shape[0]
    f_best = np.where(t == BENEFIT, X.max(axis=0), X.min(axis=0))
    f_worst = np.where(t == BENEFIT, X.min(axis=0), X.max(axis=0))
    span = f_best - f_worst
    if np.any(span == 0):
        bad = np.where(span == 0)[0].tolist()
        raise ValueError(f"Criteria {bad} have identical values for all alternatives; drop them for VIKOR.")
    G = w * (f_best - X) / span
    S, R = G.sum(axis=1), G.max(axis=1)

    def _scaled(a):
        rng = a.max() - a.min()
        return (a - a.min()) / rng if rng > 0 else np.zeros_like(a)

    Q = v * _scaled(S) + (1 - v) * _scaled(R)
    details = {"S": S, "R": R, "v": v}
    details.update(vikor_compromise(Q, S, R))
    return Result("VIKOR", Q, rank(Q, higher_is_better=False), False, details)


def vikor_compromise(Q, S, R) -> dict:
    """Opricovic & Tzeng (2004) conditions C1 (acceptable advantage) and C2 (acceptable stability)."""
    Q, S, R = (np.asarray(a, dtype=float) for a in (Q, S, R))
    order = np.argsort(Q, kind="stable")
    dq = 1 / (Q.size - 1)
    a1, a2 = order[0], order[1]
    c1 = Q[a2] - Q[a1] >= dq
    c2 = S[a1] == S.min() or R[a1] == R.min()
    if c1 and c2:
        compromise = [int(a1)]
    elif c1:
        compromise = [int(a1), int(a2)]
    else:
        compromise = [int(k) for k in order if Q[k] - Q[a1] < dq]
    return {"DQ": dq, "acceptable_advantage": bool(c1), "acceptable_stability": bool(c2),
            "compromise_set": compromise}


def gra(X, w, types, rho: float = 0.5) -> Result:
    """Grey relational analysis (Deng, 1982; Kuo, Yang & Huang, 2008) against the ideal series."""
    X, w, t = check_inputs(X, w, types)
    if not 0 < rho <= 1:
        raise ValueError("Distinguishing coefficient rho must be in (0, 1].")
    Rn = normalize(X, t, "minmax")
    delta = np.abs(1.0 - Rn)
    dmin, dmax = delta.min(), delta.max()
    if dmax == 0:
        coef = np.ones_like(delta)
    else:
        coef = (dmin + rho * dmax) / (delta + rho * dmax)
    grade = coef @ w
    return Result("GRA", grade, rank(grade), True, {"rho": rho, "coefficients": coef})


def preference_degree(d: np.ndarray, kind: str = "usual", q: float = 0.0, p: float = 0.0,
                      s: float = 1.0) -> np.ndarray:
    """PROMETHEE preference functions (Brans & Vincke, 1985) applied to differences d."""
    d = np.asarray(d, dtype=float)
    P = np.zeros_like(d)
    pos = d > 0
    if kind == "usual":
        P[pos] = 1.0
    elif kind == "ushape":
        P[d > q] = 1.0
    elif kind == "vshape":
        if p <= 0:
            raise ValueError("vshape needs p > 0.")
        P = np.clip(d / p, 0.0, 1.0)
    elif kind == "level":
        P[(d > q) & (d <= p)] = 0.5
        P[d > p] = 1.0
    elif kind == "linear":
        if p <= q:
            raise ValueError("linear needs p > q.")
        P = np.clip((d - q) / (p - q), 0.0, 1.0)
    elif kind == "gaussian":
        P[pos] = 1 - np.exp(-(d[pos] ** 2) / (2 * s ** 2))
    else:
        raise ValueError(f"Unknown preference function {kind!r}.")
    return P


def promethee(X, w, types, functions: Sequence[dict] | None = None) -> Result:
    """PROMETHEE I (partial) and II (net flow). functions[j] = {"kind": ..., "q": ..., "p": ..., "s": ...}."""
    X, w, t = check_inputs(X, w, types)
    m, n = X.shape
    functions = functions or [{"kind": "usual"}] * n
    if len(functions) != n:
        raise ValueError("Provide one preference function per criterion.")
    pi = np.zeros((m, m))
    for j in range(n):
        g = X[:, j] * t[j]  # orient so larger is better
        d = g[:, None] - g[None, :]
        f = dict(functions[j])
        pi += w[j] * preference_degree(d, f.pop("kind", "usual"), **f)
    np.fill_diagonal(pi, 0.0)
    phi_plus = pi.sum(axis=1) / (m - 1)
    phi_minus = pi.sum(axis=0) / (m - 1)
    net = phi_plus - phi_minus
    # PROMETHEE I: a outranks b if it is at least as good on both flows and better on one.
    better = (phi_plus[:, None] >= phi_plus[None, :]) & (phi_minus[:, None] <= phi_minus[None, :])
    strict = (phi_plus[:, None] > phi_plus[None, :]) | (phi_minus[:, None] < phi_minus[None, :])
    outranks = better & strict
    incomparable = ~(better | better.T)
    np.fill_diagonal(incomparable, False)
    return Result("PROMETHEE II", net, rank(net), True,
                  {"phi_plus": phi_plus, "phi_minus": phi_minus, "pi": pi,
                   "promethee_I_outranks": outranks, "promethee_I_incomparable": incomparable})


def electre_i(X, w, types, c_hat: float | None = None, d_hat: float | None = None) -> dict:
    """ELECTRE I (Roy, 1968): concordance/discordance outranking and its kernel.

    Discordance divides by each criterion's range. Thresholds default to the mean
    off-diagonal concordance and discordance; state the thresholds you use.
    """
    X, w, t = check_inputs(X, w, types)
    m, n = X.shape
    G = X * t
    rng = G.max(axis=0) - G.min(axis=0)
    rng[rng == 0] = 1.0
    C, D = np.zeros((m, m)), np.zeros((m, m))
    for a in range(m):
        for b in range(m):
            if a == b:
                continue
            diff = G[a] - G[b]
            C[a, b] = w[diff >= 0].sum()
            worse = diff < 0
            D[a, b] = np.max(-diff[worse] / rng[worse]) if worse.any() else 0.0
    off = ~np.eye(m, dtype=bool)
    c_hat = float(C[off].mean()) if c_hat is None else c_hat
    d_hat = float(D[off].mean()) if d_hat is None else d_hat
    S = (C >= c_hat) & (D <= d_hat) & off
    kernel, cyclic = _kernel(S)
    return {"concordance": C, "discordance": D, "c_hat": c_hat, "d_hat": d_hat,
            "outranks": S, "kernel": kernel, "graph_has_cycle": cyclic}


def _kernel(S: np.ndarray) -> tuple[list[int], bool]:
    remaining = set(range(S.shape[0]))
    kernel: list[int] = []
    while remaining:
        sources = [a for a in remaining if not any(S[b, a] for b in remaining if b != a)]
        if not sources:
            return kernel, True  # cycle: kernel not unique; reduce cycles or change thresholds
        kernel.extend(sources)
        dominated = {b for a in sources for b in remaining if S[a, b]}
        remaining -= set(sources) | dominated
    return sorted(kernel), False


def fuzzy_topsis(F, weights, types) -> Result:
    """Fuzzy TOPSIS of Chen (2000) with TFN ratings F (m, n, 3).

    weights: crisp (n,) or fuzzy (n, 3). FPIS = (1,1,1), FNIS = (0,0,0),
    vertex distance d(x, y) = sqrt(mean((x - y)^2)).
    """
    F = np.asarray(F, dtype=float)
    m, n, _ = F.shape
    t = parse_types(types, n)
    W = np.asarray(weights, dtype=float)
    if W.shape == (n,):
        W = np.repeat(W[:, None], 3, axis=1)
    if W.shape != (n, 3):
        raise ValueError("weights must have shape (n,) or (n, 3).")
    R = np.empty_like(F)
    for j in range(n):
        if t[j] == BENEFIT:
            c_star = F[:, j, 2].max()
            R[:, j] = F[:, j] / c_star
        else:
            a_minus = F[:, j, 0].min()
            R[:, j] = a_minus / F[:, j, ::-1]
    V = R * W[None, :, :]
    d_plus = np.sqrt(((V - 1.0) ** 2).mean(axis=2)).sum(axis=1)
    d_minus = np.sqrt((V ** 2).mean(axis=2)).sum(axis=1)
    cc = d_minus / (d_plus + d_minus)
    return Result("Fuzzy TOPSIS (Chen 2000)", cc, rank(cc), True,
                  {"d_plus": d_plus, "d_minus": d_minus, "weighted_fuzzy": V})


def dematel(Z, threshold: float | None = None) -> dict:
    """DEMATEL (Gabus & Fontela, 1972). Z: (n, n) direct-relation matrix or (k, n, n) expert stack."""
    Z = np.asarray(Z, dtype=float)
    if Z.ndim == 3:
        Z = Z.mean(axis=0)
    n = Z.shape[0]
    if Z.shape != (n, n) or np.any(Z < 0):
        raise ValueError("Z must be a non-negative square matrix.")
    if np.any(np.diag(Z) != 0):
        raise ValueError("Diagonal of the direct-relation matrix must be zero.")
    s = max(Z.sum(axis=1).max(), Z.sum(axis=0).max())
    N = Z / s
    I = np.eye(n)
    if np.linalg.cond(I - N) > 1e12:
        raise ValueError("I - N is singular (all row and column sums equal); total relation diverges.")
    T = N @ np.linalg.inv(I - N)
    Dr, Rc = T.sum(axis=1), T.sum(axis=0)
    alpha = float(T.mean()) if threshold is None else threshold
    return {"N": N, "T": T, "D": Dr, "R": Rc, "prominence": Dr + Rc, "relation": Dr - Rc,
            "threshold": alpha, "influence_edges": [(int(i), int(j), float(T[i, j]))
                                                    for i, j in zip(*np.where(T > alpha))]}


# ---------------------------------------------------------------------------
# Robustness and comparison
# ---------------------------------------------------------------------------

MethodFn = Callable[[np.ndarray, np.ndarray, np.ndarray], Result]


def weight_sensitivity(fn: MethodFn, X, w, types, criterion: int, grid=None) -> dict:
    """One-at-a-time sensitivity: set w_c to each grid value, rescale the others proportionally."""
    X, w, t = check_inputs(X, w, types)
    grid = np.linspace(0.0, 1.0, 21) if grid is None else np.asarray(grid, dtype=float)
    base = fn(X, w, t).ranks
    rows = []
    for g in grid:
        others = np.delete(np.arange(w.size), criterion)
        rest = w[others].sum()
        w2 = np.empty_like(w)
        w2[criterion] = g
        w2[others] = (1 - g) * (w[others] / rest if rest > 0 else np.full(others.size, 1 / others.size))
        r = fn(X, w2, t).ranks
        rows.append({"weight": float(g), "ranks": r.tolist(), "best": int(np.argmin(r)),
                     "same_as_base": bool(np.array_equal(r, base))})
    switches = [rows[k]["weight"] for k in range(1, len(rows)) if rows[k]["best"] != rows[k - 1]["best"]]
    return {"criterion": criterion, "base_weight": float(w[criterion]), "base_ranks": base.tolist(),
            "grid": rows, "best_alternative_switches_at": switches}


def rank_reversal_check(fn: MethodFn, X, w, types) -> dict:
    """Remove each alternative in turn and count pairwise order flips among the rest."""
    X, w, t = check_inputs(X, w, types)
    base = fn(X, w, t).ranks
    m = X.shape[0]
    report = []
    for k in range(m):
        keep = np.delete(np.arange(m), k)
        if keep.size < 2:
            continue
        r = fn(X[keep], w, t).ranks
        flips = [(int(keep[a]), int(keep[b])) for a in range(keep.size) for b in range(a + 1, keep.size)
                 if np.sign(base[keep[a]] - base[keep[b]]) != np.sign(r[a] - r[b])]
        report.append({"removed": k, "pair_flips": flips})
    return {"base_ranks": base.tolist(), "removals": report,
            "any_reversal": any(item["pair_flips"] for item in report)}


def weight_acceptability(fn: MethodFn, X, w, types, n_samples: int = 2000, concentration: float = 50.0,
                         seed: int = 0) -> dict:
    """SMAA-style rank acceptability (Lahdelma et al., 1998) with Dirichlet(concentration * w) weights.

    Use concentration -> 0 (e.g. 1.0 * n) for 'no preference information'.
    """
    X, w, t = check_inputs(X, w, types)
    rng = np.random.default_rng(seed)
    m = X.shape[0]
    alpha = np.maximum(concentration * w, 1e-3)
    counts = np.zeros((m, m))
    for _ in range(n_samples):
        r = fn(X, rng.dirichlet(alpha), t).ranks
        counts[np.arange(m), r - 1] += 1
    acc = counts / n_samples
    return {"rank_acceptability": acc, "first_rank_acceptability": acc[:, 0],
            "n_samples": n_samples, "concentration": concentration, "seed": seed}


def weighted_spearman(x, y) -> float:
    """r_w of Pinto da Costa & Soares (2005); emphasises agreement at the top ranks."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    N = x.size
    num = 6 * np.sum((x - y) ** 2 * ((N - x + 1) + (N - y + 1)))
    return float(1 - num / (N ** 4 + N ** 3 - N ** 2 - N))


def ws_coefficient(reference, other) -> float:
    """WS similarity of Sałabun & Urbaniak (2020); asymmetric, reference ranking first."""
    x, y = np.asarray(reference, float), np.asarray(other, float)
    N = x.size
    return float(1 - np.sum(2.0 ** (-x) * np.abs(x - y) / np.maximum(np.abs(x - 1), np.abs(x - N))))


def compare_rankings(rankings: dict) -> dict:
    """Pairwise Spearman, Kendall tau-b, weighted Spearman and WS for {name: ranks}."""
    names = list(rankings)
    out = {"names": names}
    for key in ("spearman", "kendall", "weighted_spearman", "ws"):
        out[key] = np.eye(len(names))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i == j:
                continue
            ra, rb = np.asarray(rankings[a]), np.asarray(rankings[b])
            out["spearman"][i, j] = stats.spearmanr(ra, rb).statistic
            out["kendall"][i, j] = stats.kendalltau(ra, rb).statistic
            out["weighted_spearman"][i, j] = weighted_spearman(ra, rb)
            out["ws"][i, j] = ws_coefficient(ra, rb)
    return out


RANKING_METHODS = {"wsm": wsm, "wpm": wpm, "waspas": waspas, "topsis": topsis, "vikor": vikor,
                   "promethee": promethee, "gra": gra}
