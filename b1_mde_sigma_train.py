#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
b1_mde_sigma_train.py - can the planned campaign resolve its own question?

Pre-registration H2 asks whether sigma_train(RLF) of the INVERTED assignment exceeds
that of the proposed assignment, with separated bounds. This script answers, BEFORE
any training is run, what ratio of sigma_train is detectable at (k, R).

GATE FIRST: the estimator must reproduce the PUBLISHED sigma_train values from the
existing R=8 grids. If it cannot, nothing downstream is trustworthy and we stop.

Evaluation-only. Reads existing CSVs. Trains nothing. Writes nothing but its own output.
"""
import os, sys, glob, csv, math
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
RNG  = np.random.default_rng(20260808)

# published sigma_train for RLF (Table 1), used as the gate
PUBLISHED = {
    "flat":        3.89,
    "incremental": 2.30,
    "binary":      6.22,
    "guarded":     None,   # reported as a bound < 0.17
}
GUARDED_BOUND = 0.17


def load_grid(pattern, metric="RLF"):
    """Return dict seed -> list of per-repeat metric values."""
    by_seed = {}
    for path in sorted(glob.glob(os.path.join(ROOT, pattern))):
        with open(path, newline="", encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                try:
                    s = int(row["seed"]); v = float(row[metric])
                except (KeyError, ValueError):
                    continue
                by_seed.setdefault(s, []).append(v)
    return by_seed


def sigma_train(grid):
    """Two-way crossed random-effects: sigma_train = sqrt(max(MS_A - MS_E, 0)/R).

    Rows = trainings (k), columns = CRN repeats (R), one observation per cell.
    """
    seeds = sorted(grid)
    R = min(len(grid[s]) for s in seeds)
    Y = np.array([grid[s][:R] for s in seeds], dtype=float)   # k x R
    k = Y.shape[0]
    if k < 2 or R < 2:
        return None, None, k, R
    gm   = Y.mean()
    MS_A = R * ((Y.mean(axis=1) - gm) ** 2).sum() / (k - 1)
    MS_B = k * ((Y.mean(axis=0) - gm) ** 2).sum() / (R - 1)
    resid = Y - Y.mean(axis=1, keepdims=True) - Y.mean(axis=0, keepdims=True) + gm
    MS_E = (resid ** 2).sum() / ((k - 1) * (R - 1))
    var  = (MS_A - MS_E) / R
    return (math.sqrt(var) if var > 0 else 0.0), MS_E, k, R


def simulate_power(sigma_a, sigma_b, sigma_e, k, R, n=4000, alpha=0.05):
    """Power to declare sigma_train(B) > sigma_train(A) via non-overlapping
    one-sided bounds at joint coverage (1-alpha), by parametric bootstrap."""
    wins = 0
    for _ in range(n):
        est = {}
        for name, s_t in (("a", sigma_a), ("b", sigma_b)):
            a_eff = RNG.normal(0.0, s_t, size=k)[:, None]
            e     = RNG.normal(0.0, sigma_e, size=(k, R))
            Y     = a_eff + e
            gm    = Y.mean()
            MS_A  = R * ((Y.mean(axis=1) - gm) ** 2).sum() / (k - 1)
            resid = Y - Y.mean(axis=1, keepdims=True) - Y.mean(axis=0, keepdims=True) + gm
            MS_E  = (resid ** 2).sum() / ((k - 1) * (R - 1))
            # one-sided chi-square bounds on the variance component
            df    = k - 1
            lo    = max((MS_A * df / CHI2_HI[df] - MS_E) / R, 0.0)
            hi    = max((MS_A * df / CHI2_LO[df] - MS_E) / R, 0.0)
            est[name] = (math.sqrt(lo), math.sqrt(hi))
        if est["b"][0] > est["a"][1]:
            wins += 1
    return wins / n


# chi-square quantiles at 0.025 / 0.975 for small df, keyed by df
from math import inf
try:
    from scipy.stats import chi2
    CHI2_LO = {d: chi2.ppf(0.025, d) for d in range(1, 40)}
    CHI2_HI = {d: chi2.ppf(0.975, d) for d in range(1, 40)}
except ImportError:
    print("scipy not available - install it or the bound calculation cannot run")
    sys.exit(2)


def main():
    print("=" * 78)
    print("GATE - reproduce published sigma_train from the existing R=8 grids")
    print("=" * 78)
    grids, sigmas, resid = {}, {}, {}
    ok = True
    # NOTE: the branched-incremental arm was RETRAINED on 2026-08-03 to remove the C4
    # weight-provenance confound (bdq5_s0..s4, origin retrained-20260803) - which is what
    # Table 1's footnote records. The superseded pre-retrain grid under
    # results/tablevii_r8/unit_incremental_* yields 5.88, the PUBLISHED PRE-RETRAIN value,
    # and must not be used. Published post-retrain sigma_train is 2.30.
    SOURCE = {
        "flat":        "results/tablevii_r8/unit_flat_s*_laptop.csv",
        "incremental": "results/bdq5/bdq5.csv",
        "binary":      "results/tablevii_r8/unit_binary_s*_laptop.csv",
        "guarded":     "results/tablevii_r8/unit_guarded_s*_laptop.csv",
    }
    for arm in ("flat", "incremental", "binary", "guarded"):
        g = load_grid(SOURCE[arm])
        if not g:
            print(f"  {arm:12} NO DATA FOUND"); ok = False; continue
        s, mse, k, R = sigma_train(g)
        grids[arm], sigmas[arm], resid[arm] = g, s, math.sqrt(mse)
        pub = PUBLISHED[arm]
        if pub is None:
            verdict = "OK (bound)" if s < GUARDED_BOUND else f"MISMATCH (bound {GUARDED_BOUND})"
        else:
            verdict = "MATCH" if abs(s - pub) < 0.06 else f"MISMATCH (published {pub})"
        if "MISMATCH" in verdict:
            ok = False
        print(f"  {arm:12} k={k} R={R}  sigma_train={s:6.3f}  sigma_resid={resid[arm]:6.3f}   {verdict}")

    if not ok:
        print("\nGATE FAILED - estimator does not reproduce the published values.")
        print("Stopping. Nothing downstream would be trustworthy.")
        sys.exit(1)
    print("\nGATE PASSED\n")

    # residual scale to assume for the new arms: use the guarded arm's (most similar design)
    s_e = resid["guarded"]
    s_A = sigmas["guarded"] if sigmas["guarded"] > 0 else 0.10
    print("=" * 78)
    print("MDE - smallest sigma_train(inverted) detectable against the proposed arm")
    print(f"assuming proposed sigma_train = {s_A:.3f}, residual sigma = {s_e:.3f}")
    print("=" * 78)
    print(f"{'k':>4} {'R':>4} | minimum detectable sigma_train(inverted) at 80% power")
    print("-" * 78)
    for k, R in ((5, 8), (5, 32), (10, 8), (10, 32), (20, 8)):
        found = None
        for cand in np.arange(0.2, 12.01, 0.2):
            if simulate_power(s_A, float(cand), s_e, k, R, n=1200) >= 0.80:
                found = float(cand); break
        if found is None:
            print(f"{k:>4} {R:>4} | not reachable below 12.0")
        else:
            print(f"{k:>4} {R:>4} | {found:5.1f}   (ratio {found/max(s_A,1e-9):5.1f}x)")

    print()
    print("REFERENCE - observed sigma_train(RLF) in existing arms:")
    for a in ("flat", "incremental", "binary", "guarded"):
        print(f"   {a:12} {sigmas[a]:6.3f}")
    print()
    print("READ THIS AS: if the inverted arm's true sigma_train is BELOW the number")
    print("in the table for your chosen (k,R), the campaign CANNOT resolve H2 and")
    print("the result will be 'inconclusive' by the pre-registration's own rule.")


if __name__ == "__main__":
    main()
