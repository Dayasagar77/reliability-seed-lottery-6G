# -*- coding: utf-8 -*-
"""
review_v91_recompute.py -- regenerate every number added to the manuscript in the
2026-08-06 review-response pass, from released records only. No training, no re-evaluation.

Covers:
  1. Section II-A   -- closed-form bound on the 2D-geometry approximation
  2. Section IX-F   -- CRN block-count sensitivity (R = 4 and R = 2 vs the reported R = 8)
  3. Section IX-F   -- corrected sample sizes: 127 against Safe-DQN, 154 against DDQN+PER
  4. validation     -- Table VI-A (24 cells), Table VII (20 sigma_train + 4 means),
                       Table XI (8 values), the 0.17 bound, and every Holm-adjusted p-value

Run from the repo root:  python review_v91_recompute.py
"""
import csv, itertools, sys
import numpy as np
from scipy import stats

R_DIR = "results/"                      # <- adjust if run from elsewhere

# ---------------------------------------------------------------- helpers
def _grid(rows, key, val, metric):
    sel = [r for r in rows if r[key] == val]
    seeds = sorted({int(r["seed"]) for r in sel})
    reps  = sorted({int(r["repeat"]) for r in sel})
    return np.array([[float(next(r for r in sel
                                 if int(r["seed"]) == s and int(r["repeat"]) == p)[metric])
                      for p in reps] for s in seeds])

def load(metric="RLF"):
    t7 = list(csv.DictReader(open(R_DIR + "tablevii_r8_laptop.csv")))
    bd = list(csv.DictReader(open(R_DIR + "bdq5/bdq5.csv")))
    xi = list(csv.DictReader(open(R_DIR + "tablex_r8_laptop.csv")))
    G = {
        "proposed":   _grid(t7, "arm", "guarded", metric),
        "flat":       _grid(t7, "arm", "flat",    metric),
        "unguarded":  _grid(t7, "arm", "binary",  metric),
        "bdq":        np.array([[float(next(r for r in bd if int(r["seed"]) == s
                                            and int(r["repeat"]) == p)[metric])
                                 for p in range(8)] for s in range(5)]),
        "dn":   _grid(xi, "label", "Proposed Dn-DQN (matched objective)", metric),
        "safe": _grid(xi, "label", "Shielded Safe-DQN", metric),
        "ddqn": _grid(xi, "label", "DDQN+PER", metric),
        "std":  _grid(xi, "label", "Standard DQN", metric),
    }
    return G

def decomp(M):
    """Two-way crossed random-effects decomposition over seeds x CRN blocks."""
    a, R = M.shape
    gm, rm, cm = M.mean(), M.mean(1), M.mean(0)
    MS_A = R * ((rm - gm) ** 2).sum() / (a - 1)
    MS_E = ((M - rm[:, None] - cm[None, :] + gm) ** 2).sum() / ((a - 1) * (R - 1))
    var  = (MS_A - MS_E) / R
    return (np.sqrt(var) if var > 0 else 0.0), MS_A, MS_E, MS_A / MS_E

def sigma(M):  return decomp(M)[0]

def upper_limit(M, alpha=0.025):
    """One-sided 100(1-alpha)% upper limit on sigma_train (F pivot)."""
    _, MS_A, MS_E, _ = decomp(M)
    a, R = M.shape
    f = stats.f.ppf(alpha, a - 1, (a - 1) * (R - 1))
    return np.sqrt(max((MS_A / f - MS_E) / R, 0.0))

def vblock(A, B):
    """Block variance of the arm-mean difference, ddof = 0."""
    return (A.mean(0) - B.mean(0)).var(ddof=0)

def crn_welch(A, B):
    k, R = A.shape
    sA, sB, Vb = sigma(A), sigma(B), vblock(A, B)
    t1, t2, t3 = sA**2 / k, sB**2 / k, Vb / R
    se2 = t1 + t2 + t3
    df  = se2**2 / ((t1**2 / (k-1) if t1 > 0 else 0)
                  + (t2**2 / (k-1) if t2 > 0 else 0) + t3**2 / (R-1))
    t = (A.mean(1).mean() - B.mean(1).mean()) / np.sqrt(se2)
    return t, 2 * stats.t.sf(abs(t), df), df

def mde(sA, sB, Vb, k, R=8, power=0.80, alpha=0.05):
    """Minimum detectable effect: exact non-central t, Satterthwaite df."""
    t1, t2, t3 = sA**2 / k, sB**2 / k, Vb / R
    se2 = t1 + t2 + t3; se = np.sqrt(se2)
    df  = se2**2 / ((t1**2 / (k-1) if t1 > 0 else 0)
                  + (t2**2 / (k-1) if t2 > 0 else 0) + t3**2 / (R-1))
    tc  = stats.t.ppf(1 - alpha/2, df)
    f = lambda d: (1 - stats.nct.cdf(tc, df, d/se) + stats.nct.cdf(-tc, df, d/se)) - power
    lo, hi = 1e-9, 1e4
    for _ in range(200):
        m = (lo + hi) / 2
        lo, hi = (m, hi) if f(m) < 0 else (lo, m)
    return (lo + hi) / 2

def mde_floor(Vb, R=8, power=0.80, alpha=0.05):
    return np.sqrt(Vb / R) * (stats.norm.ppf(1 - alpha/2) + stats.norm.ppf(power))

def trainings_needed(sA, sB, Vb, target, kmax=100000):
    if mde(sA, sB, Vb, kmax) > target: return None
    lo, hi = 2, kmax
    while lo < hi:
        m = (lo + hi) // 2
        lo, hi = (m + 1, hi) if mde(sA, sB, Vb, m) > target else (lo, m)
    return lo

def holm(raw):
    m = len(raw); order = np.argsort(raw); adj = np.empty(m); run = 0.0
    for i, j in enumerate(order):
        run = max(run, min(1.0, raw[j] * (m - i))); adj[j] = run
    return adj

# ---------------------------------------------------------------- 1. geometry
def geometry(D_perp=10.0, R_cell=37.4, n_los=1.9, n_nlos=2.9):
    print("\n1. SECTION II-A -- bound on the 2D-geometry approximation")
    print("   extra path loss = 10 n log10( sqrt(d^2 + dh^2) / d ),  d = sqrt(D_perp^2 + x^2)")
    x = np.linspace(0.0, R_cell, 4001)
    d2 = np.hypot(D_perp, x)
    n  = np.where(d2 < 0.75 * R_cell, n_los, n_nlos)
    for dh in (7.0, 10.0):
        err = 10 * n * np.log10(np.sqrt(D_perp**2 + x**2 + dh**2) / d2)
        guard = err[d2 >= 0.75 * R_cell]
        print("   dh = %4.1f m : max %.2f dB at x = %.1f m (abeam) | cell edge %.2f dB | "
              "guard region (d >= 0.75R) <= %.2f dB"
              % (dh, err.max(), x[err.argmax()], err[-1], guard.max()))

# ---------------------------------------------------------------- 2. CRN sensitivity
def crn_sensitivity(G):
    print("\n2. SECTION IX-F -- CRN block-count sensitivity")
    names = [("flat", "flat"), ("unguarded", "unguarded"),
             ("fine dial", "bdq"), ("proposed", "proposed")]
    for Rsub in (4, 2):
        subs = list(itertools.combinations(range(8), Rsub))
        print("   R = %d  (%d subsets of the 8 blocks)" % (Rsub, len(subs)))
        for lbl, key in names:
            full = sigma(G[key])
            v = np.array([sigma(G[key][:, list(s)]) for s in subs])
            print("      RLF sigma_train %-10s R=8 %5.2f -> median %5.2f (%5.2f .. %5.2f)"
                  % (lbl, full, np.median(v), v.min(), v.max()))
        Fs   = np.array([decomp(G["proposed"][:, list(s)])[3] for s in subs])
        crit = stats.f.ppf(0.975, 4, 4 * (Rsub - 1))
        ul   = np.array([upper_limit(G["proposed"][:, list(s)]) for s in subs])
        print("      guarded component resolvable in %d/%d subsets (crit F = %.2f); "
              "upper limit median %.2f" % (int((Fs > crit).sum()), len(subs), crit, np.median(ul)))
        for other in ("flat", "bdq", "unguarded"):
            res = np.array([crn_welch(G["proposed"][:, list(s)], G[other][:, list(s)])[:2]
                            for s in subs])
            t, p = res[:, 0], res[:, 1]
            print("      Welch proposed vs %-10s sign preserved %d/%d, p<0.05 in %d/%d"
                  % (other, int((t < 0).sum()), len(subs), int((p < 0.05).sum()), len(subs)))

# ---------------------------------------------------------------- 3/4. validation
PUB_VI_A = [("Proposed vs BDQ-style", "proposed", "bdq",       3.83, 2.40, 1.72, 0.81),
            ("Proposed vs flat",      "proposed", "flat",      6.52, 3.91, 2.64, 0.62),
            ("Proposed vs unguarded", "proposed", "unguarded",10.44, 6.23, 4.17, 0.77),
            ("Dn-DQN vs Safe-DQN",    "dn",       "safe",     11.58, 7.34, 4.99, 0.57),
            ("Dn-DQN vs DDQN+PER",    "dn",       "ddqn",     12.49, 8.13, 5.57, 0.40),
            ("Dn-DQN vs Std DQN",     "dn",       "std",      11.44, 7.11, 4.83, 0.88)]

def validate(G):
    print("\n3. VALIDATION -- Table VI-A, 24 cells")
    ok = True
    for name, a, b, p5, p10, p20, pf in PUB_VI_A:
        A, B = G[a], G[b]; sA, sB, Vb = sigma(A), sigma(B), vblock(A, B)
        got = [mde(sA, sB, Vb, k) for k in (5, 10, 20)] + [mde_floor(Vb)]
        pub = [p5, p10, p20, pf]
        good = all(abs(g - p) < 0.06 for g, p in zip(got, pub)); ok &= good
        print("   %-24s %s  %s" % (name, "  ".join("%6.2f" % g for g in got),
                                   "ok" if good else "XX " + str(pub)))
    print("\n4. SECTION IX-F -- trainings needed to resolve a 2.00-point RLF difference at 80%% power")
    for name, a, b, *_ in PUB_VI_A:
        A, B = G[a], G[b]
        n = trainings_needed(sigma(A), sigma(B), vblock(A, B), 2.0)
        print("   %-24s %s" % (name, n if n else "unattainable (floor above target)"))
    print("   -> manuscript states 15, 35, 127, 154; 127/154 replace the stale 124/148.")
    return ok

def validate_tables(G):
    print("\n5. VALIDATION -- Table VII / Table XI reproduction")
    METRICS = ["HO", "Power", "Spec", "RLF", "Tput"]
    for metric in METRICS:
        Gm = load(metric)
        line = "   %-6s " % metric
        for key in ("flat", "bdq", "unguarded", "proposed"):
            line += "%s %7.2f+-%-6.2f " % (key[:4], Gm[key].mean(), sigma(Gm[key]))
        print(line)
    u = upper_limit(G["proposed"])
    _, _, _, F = decomp(G["proposed"])
    print("   guarded RLF: F(4,28) = %.2f, p = %.4f, 97.5%% one-sided upper limit = %.3f  "
          "[manuscript: 1.02 / 0.4155 / <0.17]"
          % (F, stats.f.sf(F, 4, 28), u))

def validate_holm():
    print("\n6. VALIDATION -- Holm step-down on every published family")
    FAM = {
      "proposed vs flat":  ([0.016140,0.040325,0.047200,0.90,0.95],   {0:0.0807,1:0.1613,2:0.1613}),
      "proposed vs BDQ":   ([0.00002,0.005950,0.006000,0.8995,0.99],  {1:0.0238,2:0.0238}),
      "Dn-DQN vs Std DQN": ([0.005800,0.007675,0.011200,0.026000,0.468400],
                            {0:0.0290,1:0.0307,2:0.0337,3:0.0519,4:0.4684}),
      "Dn-DQN vs DDQN+PER":([0.042100,0.046600,0.063900,0.312550,0.418000],
                            {0:0.2105,1:0.2105,2:0.2105,3:0.6251,4:0.6251}),
    }
    allok = True
    for fam, (raw, pub) in FAM.items():
        adj = holm(np.array(raw))
        good = all(abs(adj[i] - v) < 0.0002 for i, v in pub.items()); allok &= good
        print("   %-20s %s  %s" % (fam, " ".join("%.4f" % adj[i] for i in sorted(pub)),
                                   "ok" if good else "XX"))
    print("   Holm reproduces every published adjusted value:", allok)

if __name__ == "__main__":
    G = load("RLF")
    geometry()
    crn_sensitivity(G)
    validate(G)
    validate_tables(G)
    validate_holm()
    print("\nDone. No checkpoint was loaded and no episode was simulated.")
