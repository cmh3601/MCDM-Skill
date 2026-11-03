"""ELECTRE II, III, IV and Tri-B (ELECTRE I is in mcdm_core).

Definitions follow Figueira, Mousseau & Roy (2005) "ELECTRE methods", in
Multiple Criteria Decision Analysis: State of the Art Surveys. Criteria are
oriented internally (cost criteria multiplied by -1); thresholds q <= p <= v
are positive constants in each criterion's own units.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from mcdm_core import check_inputs, parse_types


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _thresholds(n: int, q, p, v=None):
    q = np.broadcast_to(np.asarray(q, dtype=float), (n,)).copy()
    p = np.broadcast_to(np.asarray(p, dtype=float), (n,)).copy()
    if v is None:
        v = np.full(n, np.inf)
    else:
        v = np.array([np.inf if x is None else float(x) for x in np.broadcast_to(np.asarray(v, dtype=object), (n,))])
    if np.any(q < 0) or np.any(p < q) or np.any(v < p):
        raise ValueError("Thresholds must satisfy 0 <= q <= p <= v for every criterion.")
    return q, p, v


def _credibility(Ga: np.ndarray, Gb: np.ndarray, w, q, p, v) -> dict:
    """σ(a, b) for every a in Ga against every b in Gb (rows: a, cols: b)."""
    diff = Gb[None, :, :] - Ga[:, None, :]  # g_j(b) - g_j(a), shape (ma, mb, n)
    qb, pb, vb = (np.broadcast_to(t, diff.shape) for t in (q, p, v))
    c = np.ones_like(diff)
    c[diff >= pb] = 0.0
    mid = (diff > qb) & (diff < pb)
    c[mid] = (pb[mid] - diff[mid]) / (pb[mid] - qb[mid])
    C = c @ w
    d = np.zeros_like(diff)
    fin = np.isfinite(vb)
    d[(diff >= vb) & fin] = 1.0
    mid = (diff > pb) & (diff < vb) & fin
    d[mid] = (diff[mid] - pb[mid]) / (vb[mid] - pb[mid])
    S = C.copy()
    for a in range(diff.shape[0]):
        for b in range(diff.shape[1]):
            worse = d[a, b] > C[a, b]
            if worse.any():
                S[a, b] = C[a, b] * np.prod((1 - d[a, b, worse]) / (1 - C[a, b]))
    return {"concordance": C, "partial_concordance": c, "partial_discordance": d, "credibility": S}


def _qualification(S, D, lam, alpha, beta):
    sub = S[np.ix_(D, D)]
    T = (sub > lam) & (sub - sub.T > alpha + beta * sub)
    np.fill_diagonal(T, False)
    return T.sum(axis=1) - T.sum(axis=0)


def _distillate(S, A: list[int], descending: bool, alpha: float, beta: float) -> list[int]:
    """One distillate: the best (descending) or worst (ascending) subset of A."""
    if len(A) == 1:
        return A
    sub = S[np.ix_(A, A)]
    off = ~np.eye(len(A), dtype=bool)
    lam = float(sub[off].max())
    D = list(A)
    while True:
        sub = S[np.ix_(D, D)]
        off = ~np.eye(len(D), dtype=bool)
        s = alpha + beta * lam
        below = sub[off][sub[off] < lam - s]
        lam_next = float(below.max()) if below.size else 0.0
        q = _qualification(S, D, lam_next, alpha, beta)
        target = q.max() if descending else q.min()
        D_new = [D[i] for i in np.where(q == target)[0]]
        if len(D_new) == 1 or lam_next == 0.0:
            return D_new
        D, lam = D_new, lam_next


def distillation(S, alpha: float = 0.3, beta: float = -0.15) -> dict:
    """Descending and ascending distillation (Roy, 1978) with s(λ) = α + βλ.

    Returns classes (best first) for each pre-order, the ranks they imply and the
    final partial pre-order (intersection) as a relation matrix with entries
    'P+' (row preferred), 'P-' (column preferred), 'I' (indifferent), 'R' (incomparable).
    """
    S = np.asarray(S, dtype=float)
    m = S.shape[0]
    out = {}
    for name, descending in (("descending", True), ("ascending", False)):
        A, classes = list(range(m)), []
        while A:
            D = _distillate(S, A, descending, alpha, beta)
            classes.append(sorted(D))
            A = [a for a in A if a not in D]
        if not descending:
            classes = classes[::-1]
        ranks = np.empty(m, dtype=int)
        for k, cls in enumerate(classes, start=1):
            ranks[cls] = k
        out[name] = {"classes": classes, "ranks": ranks}
    rd, ra = out["descending"]["ranks"], out["ascending"]["ranks"]
    rel = np.full((m, m), "R", dtype=object)
    for a in range(m):
        for b in range(m):
            if a == b or (rd[a] == rd[b] and ra[a] == ra[b]):
                rel[a, b] = "I"
            elif rd[a] <= rd[b] and ra[a] <= ra[b]:
                rel[a, b] = "P+"
            elif rd[a] >= rd[b] and ra[a] >= ra[b]:
                rel[a, b] = "P-"
    # Convenience complete order (not Roy's): mean of the two ranks.
    avg = (rd + ra) / 2.0
    out.update({"relation": rel, "average_rank": avg,
                "rank_by_average": np.array([1 + int(np.sum(avg < x)) for x in avg]),
                "alpha": alpha, "beta": beta})
    return out


def _graph_classes(strong: np.ndarray, weak: np.ndarray, A: list[int]) -> list[int]:
    """Alternatives of A not strongly outranked inside A (cycles condensed), refined by the weak graph."""
    def top(G, nodes):
        nodes = list(nodes)
        sub = G[np.ix_(nodes, nodes)].astype(bool)
        # reachability for cycle condensation
        R = sub.copy()
        for k in range(len(nodes)):
            R = R | (R[:, [k]] & R[[k], :])
        keep = []
        for i in range(len(nodes)):
            beaten = any(sub[j, i] and not R[i, j] for j in range(len(nodes)) if j != i)
            if not beaten:
                keep.append(nodes[i])
        return keep
    first = top(strong, A)
    return top(weak, first) or first


# ---------------------------------------------------------------------------
# ELECTRE II
# ---------------------------------------------------------------------------

def electre_ii(X, w, types, c_minus: float = 0.65, c_zero: float = 0.75, c_plus: float = 0.85,
               d_minus: float = 0.25, d_plus: float = 0.50, discordance: str = "criterion_range") -> dict:
    """ELECTRE II (Roy & Bertier, 1973): strong and weak outranking, two complete pre-orders.

    a S^F b: C(a,b) >= C(b,a) and [(C >= c+ and D <= d+) or (C >= c0 and D <= d-)]
    a S^f b: C(a,b) >= C(b,a) and [(C >= c0 and D <= d+) or (C >= c- and D <= d-)]
    discordance: "criterion_range" divides by each criterion's range (as in electre_i);
                 "global_range" divides by the largest range over all criteria.
    """
    X, w, t = check_inputs(X, w, types)
    if not (c_minus <= c_zero <= c_plus and d_minus <= d_plus):
        raise ValueError("Need c- <= c0 <= c+ and d- <= d+.")
    G = X * t
    m = G.shape[0]
    rng = G.max(axis=0) - G.min(axis=0)
    if discordance == "global_range":
        rng = np.full_like(rng, rng.max() if rng.max() > 0 else 1.0)
    rng[rng == 0] = 1.0
    C, D = np.zeros((m, m)), np.zeros((m, m))
    for a in range(m):
        for b in range(m):
            if a != b:
                diff = G[a] - G[b]
                C[a, b] = w[diff >= 0].sum()
                D[a, b] = max(0.0, np.max(-diff / rng))
    base = (C >= C.T) & ~np.eye(m, dtype=bool)
    strong = base & (((C >= c_plus) & (D <= d_plus)) | ((C >= c_zero) & (D <= d_minus)))
    weak = base & (((C >= c_zero) & (D <= d_plus)) | ((C >= c_minus) & (D <= d_minus)))
    out = {"concordance": C, "discordance": D, "strong": strong, "weak": weak}
    for name, (S_, W_) in (("descending", (strong, weak)), ("ascending", (strong.T, weak.T))):
        A, classes = list(range(m)), []
        while A:
            cls = _graph_classes(S_, W_, A)
            classes.append(sorted(cls))
            A = [a for a in A if a not in cls]
        if name == "ascending":
            classes = classes[::-1]
        ranks = np.empty(m, dtype=int)
        for k, cls in enumerate(classes, start=1):
            ranks[cls] = k
        out[name] = {"classes": classes, "ranks": ranks}
    rd, ra = out["descending"]["ranks"], out["ascending"]["ranks"]
    avg = (rd + ra) / 2.0
    out["average_rank"] = avg
    out["rank_by_average"] = np.array([1 + int(np.sum(avg < x)) for x in avg])
    return out


# ---------------------------------------------------------------------------
# ELECTRE III
# ---------------------------------------------------------------------------

def electre_iii(X, w, types, q, p, v=None, alpha: float = 0.3, beta: float = -0.15) -> dict:
    """ELECTRE III (Roy, 1978): fuzzy outranking credibility and distillation."""
    X, w, t = check_inputs(X, w, types)
    q, p, v = _thresholds(X.shape[1], q, p, v)
    G = X * t
    cred = _credibility(G, G, w, q, p, v)
    S = cred["credibility"]
    np.fill_diagonal(S, 0.0)
    out = {"concordance": cred["concordance"], "credibility": S}
    out.update(distillation(S, alpha, beta))
    return out


# ---------------------------------------------------------------------------
# ELECTRE IV
# ---------------------------------------------------------------------------

def electre_iv_counts(X, types, q, p, v=None) -> dict:
    """Pairwise criterion counts m_p, m_q, m_i (b over a) and veto flags."""
    X = np.asarray(X, dtype=float)
    t = parse_types(types, X.shape[1])
    q, p, v = _thresholds(X.shape[1], q, p, v)
    G = X * t
    diff = G[None, :, :] - G[:, None, :]  # [a, b, j] = g_j(b) - g_j(a)
    mp = (diff > p).sum(axis=2)
    mq = ((diff > q) & (diff <= p)).sum(axis=2)
    mi = ((diff > 0) & (diff <= q)).sum(axis=2)
    veto = (diff >= v).any(axis=2)
    return {"mp_ba": mp, "mq_ba": mq, "mi_ba": mi, "veto_ba": veto, "n": X.shape[1]}


def electre_iv(X, types, q, p, v=None, alpha: float = 0.3, beta: float = -0.15,
               levels: Sequence[float] = (1.0, 0.8, 0.6, 0.4, 0.2)) -> dict:
    """ELECTRE IV (Roy & Hugonnard, 1982): no criterion weights.

    With m_x(b,a) counting criteria where b beats a strictly (p), weakly (q) or
    within indifference (i), a outranks b by
      quasi-dominance  S_q: m_p(b,a)+m_q(b,a)=0 and m_i(b,a) <= 1+m_i(a,b)+m_q(a,b)+m_p(a,b)
      canonical        S_c: m_p(b,a)=0, m_q(b,a) <= m_p(a,b), m_q(b,a)+m_i(b,a) <= 1+m_i(a,b)+m_q(a,b)+m_p(a,b)
      pseudo           S_p: m_p(b,a)=0 and m_q(b,a) <= m_q(a,b)+m_p(a,b)
      sub-dominance    S_s: m_p(b,a)=0
      veto-dominance   S_v: m_p(b,a)=0, or m_p(b,a)=1 without veto and m_p(a,b) >= n/2
    mapped to credibility `levels` and ranked by ELECTRE III distillation.
    """
    cnt = electre_iv_counts(X, types, q, p, v)
    mp, mq, mi, veto, n = cnt["mp_ba"], cnt["mq_ba"], cnt["mi_ba"], cnt["veto_ba"], cnt["n"]
    mpT, mqT, miT = mp.T, mq.T, mi.T  # counts of a over b
    Sq = (mp + mq == 0) & (mi <= 1 + miT + mqT + mpT)
    Sc = (mp == 0) & (mq <= mpT) & (mq + mi <= 1 + miT + mqT + mpT)
    Sp = (mp == 0) & (mq <= mqT + mpT)
    Ss = mp == 0
    Sv = (mp == 0) | ((mp == 1) & ~veto & (mpT >= n / 2))
    S = np.zeros(mp.shape)
    for rel, lvl in zip((Sv, Ss, Sp, Sc, Sq), levels[::-1]):
        S[rel] = lvl
    np.fill_diagonal(S, 0.0)
    out = {"credibility": S, "relations": {"S_q": Sq, "S_c": Sc, "S_p": Sp, "S_s": Ss, "S_v": Sv}}
    out.update(cnt)
    out.update(distillation(S, alpha, beta))
    return out


# ---------------------------------------------------------------------------
# ELECTRE Tri-B
# ---------------------------------------------------------------------------

def electre_tri_b(X, w, types, profiles, q, p, v=None, cut: float = 0.75) -> dict:
    """ELECTRE Tri-B sorting (Yu, 1992; Mousseau, Słowiński & Zielniewicz, 2000).

    profiles: (k-1, n) boundary profiles from the worst/best boundary upward, each
    at least as good as the previous on every criterion. Categories are numbered
    1 (worst) .. k (best). cut is the λ cutting level, usually in [0.5, 1].
    """
    X, w, t = check_inputs(X, w, types)
    B = np.atleast_2d(np.asarray(profiles, dtype=float))
    n = X.shape[1]
    if B.shape[1] != n:
        raise ValueError("Profiles need one value per criterion.")
    q, p, v = _thresholds(n, q, p, v)
    G, GB = X * t, B * t
    if np.any(np.diff(GB, axis=0) < 0):
        raise ValueError("Profiles must be ordered from worst to best on every criterion.")
    s_ab = _credibility(G, GB, w, q, p, v)["credibility"]   # σ(a, b_h)
    s_ba = _credibility(GB, G, w, q, p, v)["credibility"].T  # σ(b_h, a)
    a_S_b = s_ab >= cut
    b_S_a = s_ba >= cut
    k = B.shape[0] + 1
    pess = np.ones(X.shape[0], dtype=int)
    opt = np.full(X.shape[0], k, dtype=int)
    for i in range(X.shape[0]):
        for h in range(k - 2, -1, -1):  # from the best profile downward
            if a_S_b[i, h]:
                pess[i] = h + 2
                break
        for h in range(k - 1):  # from the worst profile upward
            if b_S_a[i, h] and not a_S_b[i, h]:
                opt[i] = h + 1
                break
    rel = np.where(a_S_b & b_S_a, "I", np.where(a_S_b, "P+", np.where(b_S_a, "P-", "R")))
    return {"sigma_a_b": s_ab, "sigma_b_a": s_ba, "pessimistic": pess, "optimistic": opt,
            "relation_to_profiles": rel, "categories": k, "cut": cut}
