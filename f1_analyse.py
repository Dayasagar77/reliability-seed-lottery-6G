#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
f1_analyse.py - is the retraining lottery a learning-rate artefact?

Compares sigma_train(RLF) of the flat controller at lr in {1e-4, 3e-4, 1e-3}.
The 3e-4 arm is the published setting, measured today in the A1 campaign under the
identical pipeline, so it is the correct comparator.

PRE-COMMITTED READING (fixed before the run, see f1_lr_sweep.py docstring):
  * large at every lr      -> not a tuning artefact; the premise holds
  * materially smaller at some lr -> the claim NARROWS to "at a standard setting"
  * collapses to guarded levels   -> the premise dissolves
"""
import os, csv, glob, math

METRICS = ["HO", "Power", "Spec", "RLF", "Tput"]
SETS = [
    ("lr 1e-4  best",  "results/f1_lr_eval",         "unit_lr1em04_best_s*_*.csv"),
    ("lr 3e-4  best",  "results/a1_selection_eval",  "unit_flat_best_s*_*.csv"),
    ("lr 1e-3  best",  "results/f1_lr_eval",         "unit_lr1em03_best_s*_*.csv"),
    ("lr 1e-4  final", "results/f1_lr_eval",         "unit_lr1em04_final_s*_*.csv"),
    ("lr 3e-4  final", "results/a1_selection_eval",  "unit_flat_final_s*_*.csv"),
    ("lr 1e-3  final", "results/f1_lr_eval",         "unit_lr1em03_final_s*_*.csv"),
]
GUARDED_REF = 0.085   # guarded arm, final-step, measured in A1


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
    Y = [[by[s][r][metric] for r in range(R)] for s in seeds]
    k = len(Y)
    gm = sum(sum(r) for r in Y) / (k * R)
    rm = [sum(r) / R for r in Y]
    cm = [sum(Y[i][j] for i in range(k)) / k for j in range(R)]
    MS_A = R * sum((m - gm) ** 2 for m in rm) / (k - 1)
    ss_e = sum((Y[i][j] - rm[i] - cm[j] + gm) ** 2 for i in range(k) for j in range(R))
    MS_E = ss_e / ((k - 1) * (R - 1))
    v = (MS_A - MS_E) / R
    return {"k": k, "R": R, "mean": gm, "sigma": math.sqrt(v) if v > 0 else 0.0,
            "per_seed": rm, "resid": math.sqrt(MS_E)}


def main():
    print("=" * 92)
    print("F1  -  IS THE RETRAINING LOTTERY A LEARNING-RATE ARTEFACT?")
    print("=" * 92)
    print(f"{'arm':18} {'k':>2} {'R':>2} {'mean RLF':>10} {'sigma_train':>12}  per-seed RLF")
    print("-" * 92)
    got = {}
    for name, d, pat in SETS:
        by = load(d, pat)
        if len(by) < 5:
            print(f"{name:18}  incomplete - seeds present {sorted(by)}")
            continue
        r = decomp(by, "RLF")
        got[name] = r
        ps = " ".join(f"{v:6.2f}" for v in r["per_seed"])
        print(f"{name:18} {r['k']:>2} {r['R']:>2} {r['mean']:10.3f} {r['sigma']:12.3f}  {ps}")

    best = {k: v for k, v in got.items() if k.endswith("best")}
    if len(best) < 3:
        print("\n(waiting for all three learning rates)")
        return
    print()
    print("VERDICT  (best-checkpoint arms, the published selection rule)")
    sigmas = {k: v["sigma"] for k, v in best.items()}
    lo_k = min(sigmas, key=sigmas.get)
    lo, hi = sigmas[lo_k], max(sigmas.values())
    print(f"  sigma_train spans {lo:.3f} ({lo_k.strip()}) to {hi:.3f} across the three rates.")
    print(f"  guarded arm reference: {GUARDED_REF:.3f}")
    print()
    if lo <= GUARDED_REF * 3:
        print("  -> THE LOTTERY COLLAPSES at some learning rate.")
        print("     The premise is a tuning artefact. This must be reported and the claim withdrawn")
        print("     or re-derived.")
    elif lo < 0.5 * hi:
        print("  -> The lottery is MATERIALLY SMALLER at one rate, though still far above the")
        print("     guarded arm. The claim NARROWS to 'at a standard hyperparameter setting'.")
        print("     Report the sweep; state the dependence.")
    else:
        print("  -> The lottery PERSISTS at every learning rate tested, and stays far above the")
        print(f"     guarded arm ({GUARDED_REF:.3f}). It is NOT a tuning artefact.")
        print("     Report as a pre-registered robustness check: the obvious alternative")
        print("     explanation was tested and excluded.")

    print()
    print("ALL METRICS, sigma_train (best-checkpoint arms)")
    hdr = [k for k in SETS if k[0] in best]
    print(f"  {'metric':8}" + "".join(f"{n:>16}" for n, _, _ in hdr))
    for m in METRICS:
        line = f"  {m:8}"
        for n, d, pat in hdr:
            r = decomp(load(d, pat), m)
            line += f"{r['sigma']:16.3f}"
        print(line)


if __name__ == "__main__":
    main()
