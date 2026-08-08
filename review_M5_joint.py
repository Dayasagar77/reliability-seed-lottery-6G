#!/usr/bin/env python3
"""
review_M5_joint.py — EVAL ONLY. Measures the four JOINT corners of the array-gain x LOS-rule grid.

Answers guide comment M5: review_M2_sweep.py sweeps one factor at a time, so Table S5 bounds each
assumption alone and says nothing about whether they compose. This adds
(12 dBi, 0.65), (12 dBi, 0.85), (18 dBi, 0.65), (18 dBi, 0.85).

NOTHING IS TRAINED. The same five frozen br_gst_s*.pt checkpoints, the same R = 8 seeded CRN grid,
the same evaluate_branched call. Rather than reimplement any of that, this script writes a patched
copy of review_M2_sweep.py — `review_M2_sweep_joint.py` — with three exact textual substitutions,
prints them, and runs it. Diff the two files to see precisely what changed.

  1. the setting list        one-at-a-time  ->  the four joint corners
  2. a hardcoded cloud path  /home/claude/work/results/...  ->  results/...
     (review_M2_sweep.py was written to run in a cloud container; as shipped its no-op gate
      cannot open that path on Windows and the script stops before sweeping anything)
  3. the venue stamp         'cloud-hw-torch2.5.1'  ->  'laptop-M5joint'
     so the new rows are distinguishable from the existing ones in m2sweep.csv

review_M2_sweep.py is resumable — it keys completed work on (gain, los_frac, seed, repeat) read
back from results/m2sweep/m2sweep.csv — so the five settings already measured are skipped and only
the four corners run: 4 x 5 seeds x 8 repeats = 160 evaluations.

The inherited no-op gate still runs first and still has to pass: it re-evaluates the guarded arm at
the published defaults and requires 200/200 cells bit-exact against results/tablevii_r8_laptop.csv.
If the knobs are not inert at 15.0 dBi / 0.75 the script stops rather than writing anything.

    python review_M5_joint.py            # patch, run, then report
    python review_M5_joint.py --report   # report only, from the CSV already on disk
"""
import csv, os, subprocess, sys
import numpy as np

SRC, DST = 'review_M2_sweep.py', 'review_M2_sweep_joint.py'
BASE_GAIN, BASE_FRAC = 15.0, 0.75
CORNERS = [(12.0, 0.65), (12.0, 0.85), (18.0, 0.65), (18.0, 0.85)]

SUBS = [
    ("    settings = ([(g, BASE_FRAC) for g in ARRAY_GAINS]\n"
     "                + [(BASE_GAIN, f) for f in LOS_FRACS if f != BASE_FRAC])",
     "    settings = [(12.0, 0.65), (12.0, 0.85), (18.0, 0.65), (18.0, 0.85)]   # M5 joint corners"),
    ("'/home/claude/work/results/tablevii_r8_laptop.csv'",
     "os.path.join('results', 'tablevii_r8_laptop.csv')"),
    ("'venue': 'cloud-hw-torch2.5.1'", "'venue': 'laptop-M5joint'"),
]


def patch():
    if not os.path.exists(SRC):
        sys.exit(f"ABORT — {SRC} not found. Run this from the folder that holds it.")
    src = open(SRC, encoding='utf-8', errors='replace').read()
    for i, (old, new) in enumerate(SUBS, 1):
        n = src.count(old)
        if n != 1:
            sys.exit(f"ABORT — substitution {i} matched {n} times, expected exactly 1.\n"
                     f"        {SRC} has changed since this patcher was written; nothing written.")
        src = src.replace(old, new)
        print(f"  [{i}] ok  {old.strip().splitlines()[0][:72]}")
    open(DST, 'w', encoding='utf-8', newline='\n').write(src)
    print(f"\n  -> wrote {DST}  (diff it against {SRC})\n")


def report():
    path = os.path.join('results', 'm2sweep', 'm2sweep.csv')
    rows = list(csv.DictReader(open(path, encoding='utf-8', errors='replace')))

    def cell(g, f):
        sub = [r for r in rows if float(r['array_gain_dbi']) == g and float(r['los_frac']) == f]
        if not sub:
            return None
        seeds = sorted({r['seed'] for r in sub}, key=int)
        return float(np.mean([np.mean([float(x['RLF']) for x in sub if x['seed'] == s]) for s in seeds]))

    base = cell(BASE_GAIN, BASE_FRAC)
    if base is None:
        sys.exit("ABORT — the baseline cell (15.0 dBi / 0.75) is not in m2sweep.csv.")
    print(f"\n  baseline 15 dBi / 0.75 : {base:.2f} % sustained RLF\n")
    print(f"  {'one-at-a-time':>22s} {'RLF %':>7s} {'main effect':>12s}")
    for g, f in [(12.0, BASE_FRAC), (18.0, BASE_FRAC), (BASE_GAIN, 0.65), (BASE_GAIN, 0.85)]:
        v = cell(g, f)
        if v is not None:
            print(f"  {f'{g:.0f} dBi / {f}':>22s} {v:7.2f} {v - base:+12.2f}")

    print(f"\n  {'joint corner':>22s} {'additive':>9s} {'measured':>9s} {'interaction':>12s}")
    inter = []
    for g, f in CORNERS:
        meas = cell(g, f)
        if meas is None:
            print(f"  {f'{g:.0f} dBi / {f}':>22s} {'—':>9s} {'not measured':>9s}")
            continue
        pred = base + (cell(g, BASE_FRAC) - base) + (cell(BASE_GAIN, f) - base)
        inter.append(meas - pred)
        print(f"  {f'{g:.0f} dBi / {f}':>22s} {pred:9.2f} {meas:9.2f} {meas - pred:+12.2f}")

    if inter:
        w = max(abs(x) for x in inter)
        print(f"\n  largest |interaction| = {w:.2f} pp")
        print("  Read: small against the main effects means the two assumptions compose additively,")
        print("  and Table S5's individual bounds may be added. A large positive value means the")
        print("  compounded worst case exceeds the sum and must be reported as such.")


if __name__ == "__main__":
    if '--report' in sys.argv:
        report()
    else:
        print("[M5] patching review_M2_sweep.py -> review_M2_sweep_joint.py")
        patch()
        print("[M5] running the joint sweep (no-op gate first, then 4 corners x 5 seeds x R=8)\n")
        rc = subprocess.call([sys.executable, '-u', DST])
        if rc:
            sys.exit(f"[M5] {DST} exited {rc} — nothing to report.")
        report()
