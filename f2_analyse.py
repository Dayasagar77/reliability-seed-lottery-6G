#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
f2_analyse.py - does the guard remove sensitivity to the LEARNING RATE, not just to seeds?

Pre-registered hypothesis (f2_guard_lr_sweep.py, before the run):
  H-F2: the guarded controller's sustained RLF is insensitive to the learning rate -
        mean RLF within ~1 point and sigma_train below 0.5 across lr in {1e-4, 3e-4, 1e-3},
        while the flat arm swings 4.62 -> 16.13.

  HOLDS -> the cliff converts ANY source of training variation into reliability variation,
           and the guard removes sensitivity to the source in general. Strictly stronger
           than the current claim.
  FAILS -> the guard does not do what the paper says. The claim must be rewritten.
"""
import os, csv, glob, math

METRICS = ["HO", "Power", "Spec", "RLF", "Tput"]

FLAT = [
    ("flat lr 1e-4", "results/f1_lr_eval",        "unit_lr1em04_best_s*_*.csv"),
    ("flat lr 3e-4", "results/a1_selection_eval", "unit_flat_best_s*_*.csv"),
    ("flat lr 1e-3", "results/f1_lr_eval",        "unit_lr1em03_best_s*_*.csv"),
]
GUARD = [
    ("guard lr 1e-4", "results/f2_guard_lr_eval",  "unit_g1em04_best_s*_*.csv"),
    ("guard lr 3e-4", "results/a1_selection_eval", "unit_guard_best_s*_*.csv"),
    ("guard lr 1e-3", "results/f2_guard_lr_eval",  "unit_g1em03_best_s*_*.csv"),
]

MEAN_TOL = 1.0     # pre-registered: guarded mean RLF within ~1 point across rates
SIGMA_TOL = 0.5    # pre-registered: guarded sigma_train below 0.5 at every rate


def load(d, pat):
    by = {}
    for p in sorted(glob.glob(os.path.join(d, pat))):
        for row in csv.DictReader(open(p, newline="", encoding="utf-8-sig")):
            try:
                s = int(row["seed"])
            except (KeyError, ValueError):
                continue
            by.setdefault(s, []).append({m: float(row[m]) for m in METRICS})
    return by


def decomp(by, metric):
    seeds = sorted(by)
    if len(seeds) < 2:
        return None
    R = min(len(by[s]) for s in seeds)
    if R < 2:
        return None
    Y = [[by[s][r][metric] for r in range(R)] for s in seeds]
    k = len(Y)
    gm = sum(sum(r) for r in Y) / (k * R)
    rm = [sum(r) / R for r in Y]
    cm = [sum(Y[i][j] for i in range(k)) / k for j in range(R)]
    MS_A = R * sum((m - gm) ** 2 for m in rm) / (k - 1)
    ss_e = sum((Y[i][j] - rm[i] - cm[j] + gm) ** 2 for i in range(k) for j in range(R))
    MS_E = ss_e / ((k - 1) * (R - 1))
    v = (MS_A - MS_E) / R
    return {"k": k, "R": R, "mean": gm, "sigma": math.sqrt(v) if v > 0 else 0.0, "per_seed": rm}


def table(rows, title):
    print(f"\n{title}")
    print(f"  {'arm':16} {'k':>2} {'R':>2} {'mean RLF':>10} {'sigma_train':>12}  per-seed RLF")
    out = {}
    for name, d, pat in rows:
        by = load(d, pat)
        if len(by) < 5:
            print(f"  {name:16}  incomplete - seeds {sorted(by)}")
            continue
        r = decomp(by, "RLF")
        if r is None:
            reps = min(len(v) for v in by.values())
            print(f"  {name:16}  incomplete - only {reps} repeat(s) so far, need >= 2")
            continue
        out[name] = r
        ps = " ".join(f"{v:6.2f}" for v in r["per_seed"])
        print(f"  {name:16} {r['k']:>2} {r['R']:>2} {r['mean']:10.3f} {r['sigma']:12.3f}  {ps}")
    return out


def main():
    print("=" * 94)
    print("F2  -  DOES THE GUARD REMOVE LEARNING-RATE SENSITIVITY, NOT JUST SEED SENSITIVITY?")
    print("=" * 94)
    f = table(FLAT, "FLAT controller (best-checkpoint)")
    g = table(GUARD, "GUARDED controller (best-checkpoint)")

    if len(g) < 3:
        print("\n(waiting for the guarded arms)")
        return

    gm = [v["mean"] for v in g.values()]
    gs = [v["sigma"] for v in g.values()]
    fm = [v["mean"] for v in f.values()] if len(f) == 3 else []

    print()
    print("=" * 94)
    print("VERDICT")
    print("=" * 94)
    print(f"  guarded mean RLF spans {min(gm):.3f} to {max(gm):.3f}  (range {max(gm)-min(gm):.3f} points)")
    if fm:
        print(f"  flat    mean RLF spans {min(fm):.3f} to {max(fm):.3f}  (range {max(fm)-min(fm):.3f} points)")
    print(f"  guarded sigma_train max across rates: {max(gs):.3f}")
    print()
    mean_ok = (max(gm) - min(gm)) <= MEAN_TOL
    sig_ok = max(gs) <= SIGMA_TOL
    if mean_ok and sig_ok:
        print("  -> H-F2 HOLDS.")
        print("     The guarded controller is insensitive to the learning rate on BOTH level and")
        print("     dispersion, while the flat controller is not.")
        print("     CLAIM EARNED: the cliff converts any source of training variation into")
        print("     reliability variation; the guard removes sensitivity to the source in general.")
        if fm:
            print(f"     Headline contrast: flat moves {max(fm)-min(fm):.2f} points across learning rates,")
            print(f"     guarded moves {max(gm)-min(gm):.2f}.")
    elif sig_ok and not mean_ok:
        print("  -> PARTIAL. Dispersion is controlled at every rate, but the LEVEL moves")
        print(f"     ({max(gm)-min(gm):.2f} points > {MEAN_TOL}). Claim the variance result only;")
        print("     do not claim level-invariance.")
    else:
        print("  -> H-F2 FAILS. The guarded arm is itself learning-rate sensitive")
        print(f"     (max sigma_train {max(gs):.3f} > {SIGMA_TOL}).")
        print("     The guard does not remove configuration sensitivity. Report this and")
        print("     rewrite the claim - do not present the guard as configuration-invariant.")

    print()
    print("ALL METRICS, sigma_train (guarded arms)")
    print(f"  {'metric':8}" + "".join(f"{n:>16}" for n, _, _ in GUARD))
    for m in METRICS:
        line = f"  {m:8}"
        for n, d, pat in GUARD:
            r = decomp(load(d, pat), m)
            line += f"{(r['sigma'] if r else float('nan')):16.3f}" if r else f"{'--':>16}"
        print(line)


if __name__ == "__main__":
    main()
