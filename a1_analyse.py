#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
a1_analyse.py - did the best-checkpoint save rule create the between-training variance?

Compares sigma_train(RLF) for each arm under two selection rules measured on the SAME
training runs:
    best   the published rule - keep the checkpoint with the best 20-episode MA reward
    final  no selection at all - keep the weights at the last training step

If the flat-vs-guarded sigma_train GAP survives under 'final', the reported variance
difference is a property of the controller, not of the save rule.
"""
import os, csv, glob, math, statistics as st

D = "results/a1_selection_eval"
METRICS = ["HO", "Power", "Spec", "RLF", "Tput"]
ARMS = ["flat_best", "flat_final", "guard_best", "guard_final"]


def load(arm):
    by_seed = {}
    for p in sorted(glob.glob(os.path.join(D, f"unit_{arm}_s*_*.csv"))):
        for row in csv.DictReader(open(p, newline="", encoding="utf-8-sig")):
            try:
                s = int(row["seed"])
            except (KeyError, ValueError):
                continue
            by_seed.setdefault(s, []).append({m: float(row[m]) for m in METRICS})
    return by_seed


def decomp(by_seed, metric):
    seeds = sorted(by_seed)
    if len(seeds) < 2:
        return None
    R = min(len(by_seed[s]) for s in seeds)
    Y = [[by_seed[s][r][metric] for r in range(R)] for s in seeds]
    k = len(Y)
    gm = sum(sum(row) for row in Y) / (k * R)
    rows_m = [sum(row) / R for row in Y]
    cols_m = [sum(Y[i][j] for i in range(k)) / k for j in range(R)]
    MS_A = R * sum((m - gm) ** 2 for m in rows_m) / (k - 1)
    ss_e = sum((Y[i][j] - rows_m[i] - cols_m[j] + gm) ** 2 for i in range(k) for j in range(R))
    MS_E = ss_e / ((k - 1) * (R - 1))
    var = (MS_A - MS_E) / R
    return {"k": k, "R": R, "mean": gm, "sigma_train": math.sqrt(var) if var > 0 else 0.0,
            "sigma_resid": math.sqrt(MS_E), "per_seed": rows_m}


def main():
    data = {a: load(a) for a in ARMS}
    missing = [a for a in ARMS if len(data[a]) < 5]
    if missing:
        print("INCOMPLETE - still evaluating or failed:", missing)
        for a in ARMS:
            print(f"  {a:12} seeds present: {sorted(data[a])}")
        return

    print("=" * 90)
    print("A1  -  IS THE BETWEEN-TRAINING VARIANCE AN ARTEFACT OF THE SAVE RULE?")
    print("=" * 90)
    print(f"{'arm':14} {'k':>2} {'R':>2} {'mean RLF':>10} {'sigma_train':>12} {'sigma_resid':>12}")
    print("-" * 90)
    res = {}
    for a in ARMS:
        d = decomp(data[a], "RLF")
        res[a] = d
        print(f"{a:14} {d['k']:>2} {d['R']:>2} {d['mean']:10.3f} {d['sigma_train']:12.3f} {d['sigma_resid']:12.3f}")

    fb, ff = res["flat_best"]["sigma_train"], res["flat_final"]["sigma_train"]
    gb, gf = res["guard_best"]["sigma_train"], res["guard_final"]["sigma_train"]
    print()
    print("THE COMPARISON THAT MATTERS")
    print(f"  best-checkpoint  : flat {fb:.3f}   guarded {gb:.3f}   gap {fb-gb:+.3f}"
          f"   ratio {(fb/gb if gb>1e-9 else float('inf')):.1f}x")
    print(f"  final-step       : flat {ff:.3f}   guarded {gf:.3f}   gap {ff-gf:+.3f}"
          f"   ratio {(ff/gf if gf>1e-9 else float('inf')):.1f}x")
    print()
    gap_b, gap_f = fb - gb, ff - gf
    print("VERDICT")
    if gap_b <= 1e-9:
        print("  No gap under the published rule in this replication - inconclusive.")
    else:
        retained = gap_f / gap_b
        print(f"  The flat-vs-guarded gap retains {retained*100:.0f}% of its size when selection is removed.")
        if retained >= 0.70:
            print("  -> The gap SURVIVES without checkpoint selection.")
            print("     The save rule is NOT what produces the reported between-training variance.")
            print("     Report this as a positive control: the confound was tested and excluded.")
        elif retained >= 0.35:
            print("  -> The gap is PARTLY reduced. Selection contributes but does not explain it.")
            print("     Disclose the effect size and report both rules.")
        else:
            print("  -> The gap LARGELY DISAPPEARS without selection.")
            print("     A substantial part of the headline is a selection artefact.")
            print("     The claim must be re-derived under final-checkpoint saving.")

    print()
    print("PER-SEED RLF (shows what the save rule actually changed)")
    for a in ARMS:
        vals = " ".join(f"{v:7.3f}" for v in res[a]["per_seed"])
        print(f"  {a:14} {vals}")

    print()
    print("ALL METRICS, sigma_train")
    print(f"  {'metric':8}" + "".join(f"{a:>14}" for a in ARMS))
    for m in METRICS:
        line = f"  {m:8}"
        for a in ARMS:
            d = decomp(data[a], m)
            line += f"{d['sigma_train']:14.3f}"
        print(line)


if __name__ == "__main__":
    main()
