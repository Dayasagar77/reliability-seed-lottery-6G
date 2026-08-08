#!/usr/bin/env python3
"""
frontier_reeval.py — EVAL ONLY. Re-evaluates the five FROZEN reward-tilt checkpoints
(fr_C1_eff.pt … fr_C5_rel.pt) under the channel-seeded single-pass protocol, so that
Table III stops carrying numbers drawn from OS entropy.

NOTHING IS TRAINED. run_frontier.py is never called: it skips any config already present
in results/frontier_nyu.csv, which is why its own (already seeded) evaluate() never re-ran.

Protocol is inherited, not re-implemented: this imports eval_method from fix_paper1_evals,
the exact function that produced results/comparative_matched_nyu.csv. Same env construction
(seed = seed*1000 + episode), same 5 seeds x 150 episodes, same slow+med pool, same metric code.

BUILT-IN GATE: fr_C4.pt and fr_C5_rel.pt were already re-evaluated by fix_paper1_evals.py
(rows "Proposed Dn-DQN @C4" / "@C5" of comparative_matched_nyu.csv). This script re-evaluates
them again and asserts the numbers reproduce. If they do not, the run is not trustworthy and
the script says so instead of writing a file.

    python frontier_reeval.py

Writes results/frontier_nyu_reseeded.csv. The original results/frontier_nyu.csv is NOT touched.
"""
import csv, os, sys

import train_phase2 as tp
from dndqn import load_dndqn_checkpoint
from fix_paper1_evals import eval_method          # identical protocol to comparative_matched

CONFIGS = [                                        # label, checkpoint, W(mob/pwr/spec), RLF penalty
    ('C1 (eff.)', 'fr_C1_eff.pt', '0.60/0.20/0.20',    -100),
    ('C2',        'fr_C2.pt',     '0.65/0.175/0.175',  -150),
    ('C3',        'fr_C3_mid.pt', '0.70/0.15/0.15',    -200),
    ('C4',        'fr_C4.pt',     '0.75/0.125/0.125',  -250),
    ('C5 (rel.)', 'fr_C5_rel.pt', '0.80/0.10/0.10',    -300),
]
# The gate reads comparative_matched_nyu.csv directly rather than hard-coding values, and
# compares at the precision that file actually stores: fix_paper1_evals.py writes HO/Power/Spec/RLF
# with "%.2f" and Tput with "%.1f". Comparing a full-precision mean against a 1-decimal field with a
# fixed absolute tolerance is wrong (482.16 is stored as "482.2"), so each metric is rounded to its
# own field width and matched exactly.
GATE_SRC = 'results/comparative_matched_nyu.csv'
GATE_ROW = {'fr_C4.pt': 'Proposed Dn-DQN @C4', 'fr_C5_rel.pt': 'Proposed Dn-DQN @C5'}
PREC     = {'HO': 2, 'Power': 2, 'Spec': 2, 'RLF': 2, 'Tput': 1}


def load_gate():
    if not os.path.exists(GATE_SRC):
        sys.exit(f"ABORT — {GATE_SRC} not found; cannot verify this run against the published one.")
    with open(GATE_SRC) as f:
        pub = {r['method']: r for r in csv.DictReader(f)}
    return {ck: pub[label] for ck, label in GATE_ROW.items() if label in pub}


def main():
    slow, med, fast, skip = tp.load_trajectories(tp.CSV_PATH)
    pool = slow + med
    print(f"[data] pool={len(pool)} (slow {len(slow)} + med {len(med)}) | held-out fast={len(fast)}", flush=True)

    missing = [c for _, c, _, _ in CONFIGS if not os.path.exists(c)]
    if missing:
        sys.exit(f"ABORT — checkpoints not found: {missing}")

    gate = load_gate()
    print(f"[gate] checking against {GATE_SRC}: {', '.join(sorted(gate))}", flush=True)

    rows, failures = [], []
    for label, ck, w, pen in CONFIGS:
        net = load_dndqn_checkpoint(ck)[0]
        m = eval_method(net, None, pool)
        rows.append({'config': label, 'ckpt': ck, 'W': w, 'rlf_pen': pen,
                     'HO': f"{m['HO'][0]:.2f}",    'HO_sd': f"{m['HO'][1]:.2f}",
                     'Power': f"{m['Power'][0]:.2f}", 'Power_sd': f"{m['Power'][1]:.2f}",
                     'Spec': f"{m['Spec'][0]:.2f}",   'Spec_sd': f"{m['Spec'][1]:.2f}",
                     'RLF': f"{m['RLF'][0]:.2f}",     'RLF_sd': f"{m['RLF'][1]:.2f}",
                     'Tput': f"{m['Tput'][0]:.1f}"})
        print(f"[eval] {label:11s} HO {m['HO'][0]:6.2f}  Power {m['Power'][0]:6.2f}  "
              f"Spec {m['Spec'][0]:6.2f}  RLF {m['RLF'][0]:5.2f}  Tput {m['Tput'][0]:6.1f}", flush=True)
        if ck in gate:
            for k, nd in PREC.items():
                got_s, want_s = f"{m[k][0]:.{nd}f}", gate[ck][k]
                if got_s != want_s:
                    failures.append(f"{ck} {k}: this run {got_s}, {GATE_SRC} has {want_s} "
                                    f"(full precision here: {m[k][0]!r})")

    print()
    if failures:
        print("GATE FAILED — this run does not reproduce the already-published re-evaluation:")
        for f in failures:
            print("   " + f)
        sys.exit("nothing written. Do not use these numbers.")
    print("GATE PASSED — C4 and C5 reproduce comparative_matched_nyu.csv exactly,")
    print("             at the precision that file stores (HO/Power/Spec/RLF 2 dp, Tput 1 dp).")

    out = 'results/frontier_nyu_reseeded.csv'
    os.makedirs('results', exist_ok=True)
    with open(out, 'w', newline='') as f:
        wtr = csv.DictWriter(f, fieldnames=['config', 'ckpt', 'W', 'rlf_pen', 'HO', 'HO_sd',
                                            'Power', 'Power_sd', 'Spec', 'Spec_sd',
                                            'RLF', 'RLF_sd', 'Tput'])
        wtr.writeheader(); wtr.writerows(rows)
    print(f"  -> {out}   (results/frontier_nyu.csv left untouched)")


if __name__ == "__main__":
    main()
