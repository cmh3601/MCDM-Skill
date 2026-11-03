"""Run a reproducible MCDM study from a JSON spec.

Usage:
    python mcdm_cli.py spec.json [--out results.json]

See example_spec.json for the format. Weights come from one of:
given | equal | entropy | critic | std | roc | ahp | fuzzy_ahp | bwm, optionally
blended with an objective method via "combine_with". Each entry in "methods"
is run on the same weights; rankings are compared and, if requested, a
one-at-a-time weight sensitivity, rank-reversal check and SMAA-style
acceptability analysis are added.
"""

from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mcdm_core as mc  # noqa: E402


def judgments(rows) -> np.ndarray:
    """Pairwise matrix from JSON; entries may be numbers or fraction strings such as "1/3"."""
    return np.array([[float(Fraction(v)) if isinstance(v, str) else float(v) for v in row] for row in rows])


def derive_weights(spec: dict, X: np.ndarray, types: np.ndarray) -> tuple[np.ndarray, dict]:
    cfg = spec.get("weights", {"method": "given"})
    method = cfg["method"]
    info: dict = {"method": method}
    if method == "given":
        w = np.array([c["weight"] for c in spec["criteria"]], dtype=float)
    elif method == "equal":
        w = np.full(X.shape[1], 1 / X.shape[1])
    elif method == "entropy":
        w = mc.entropy_weights(X)
    elif method == "critic":
        w = mc.critic_weights(X, types)
    elif method == "std":
        w = mc.std_weights(X, types)
    elif method == "roc":
        w = mc.roc_weights(cfg["order"])
    elif method == "ahp":
        res = mc.ahp_weights(judgments(cfg["pairwise"]), cfg.get("eigen_method", "eigen"),
                             cfg.get("ri"))
        w = res["weights"]
        info.update({k: res[k] for k in ("lambda_max", "CI", "RI", "CR", "consistent", "most_inconsistent")})
    elif method == "fuzzy_ahp":
        F = np.array(cfg["pairwise_tfn"], dtype=float) if "pairwise_tfn" in cfg else \
            mc.tfn_from_crisp(judgments(cfg["pairwise"]))
        res = mc.fuzzy_ahp_buckley(F, cfg.get("ri"))
        w = res["weights"]
        info.update({"fuzzy_weights": res["fuzzy_weights"], "modal_CR": res["modal_CR"]})
    elif method == "bwm":
        res = mc.bwm_weights(cfg["best_to_others"], cfg["others_to_worst"], cfg["best"], cfg["worst"])
        w = res["weights"]
        info.update({"xi": res["xi"], "CR_input": res["CR_input"]})
    else:
        raise ValueError(f"Unknown weight method {method!r}.")
    if "combine_with" in cfg:
        obj = {"entropy": lambda: mc.entropy_weights(X), "critic": lambda: mc.critic_weights(X, types),
               "std": lambda: mc.std_weights(X, types)}[cfg["combine_with"]]()
        info["objective_weights"] = obj
        info["subjective_weights"] = w
        w = mc.combine_weights(w, obj, cfg.get("alpha", 0.5), cfg.get("combine_mode", "linear"))
    w = w / w.sum()
    return w, info


def method_fn(entry: dict):
    name = entry["name"].lower()
    params = {k: v for k, v in entry.items() if k != "name"}
    base = mc.RANKING_METHODS[name]
    return lambda X, w, t: base(X, w, t, **params)


def run(spec: dict) -> dict:
    alts = spec["alternatives"]
    crit = [c["name"] for c in spec["criteria"]]
    X = np.array(spec["matrix"], dtype=float)
    types = mc.parse_types([c["type"] for c in spec["criteria"]], len(crit))
    w, winfo = derive_weights(spec, X, types)
    out = {"alternatives": alts, "criteria": crit, "weights": dict(zip(crit, w.tolist())),
           "weight_info": winfo, "results": {}}
    rankings = {}
    for entry in spec.get("methods", [{"name": "topsis"}]):
        label = entry.get("label", entry["name"])
        res = method_fn(entry)(X, w, types)
        out["results"][label] = res.to_dict(alts)
        rankings[label] = res.ranks
    if len(rankings) > 1:
        out["ranking_agreement"] = mc.compare_rankings(rankings)
    analysis = spec.get("analysis", {})
    primary_entry = spec.get("methods", [{"name": "topsis"}])[0]
    fn = method_fn(primary_entry)
    if analysis.get("sensitivity"):
        out["sensitivity"] = {crit[j]: mc.weight_sensitivity(fn, X, w, types, j) for j in range(len(crit))}
    if analysis.get("rank_reversal"):
        out["rank_reversal"] = mc.rank_reversal_check(fn, X, w, types)
    if analysis.get("acceptability"):
        cfg = analysis["acceptability"] if isinstance(analysis["acceptability"], dict) else {}
        out["acceptability"] = mc.weight_acceptability(fn, X, w, types, **cfg)
    if spec.get("electre_i") not in (None, False):
        cfg = spec["electre_i"] if isinstance(spec["electre_i"], dict) else {}
        out["electre_i"] = mc.electre_i(X, w, types, **cfg)
    return mc._jsonable(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spec", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    result = run(json.loads(args.spec.read_text(encoding="utf-8")))
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
