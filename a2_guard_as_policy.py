#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
a2_guard_as_policy.py - what does the LEARNING add over the safety layer alone?

THE QUESTION. Every arm in the paper that reaches ~4.9% sustained RLF is a trained controller
carrying a non-learned 9 dB state guard. Nobody has run the guard as the ENTIRE policy, with no
learned network at all. If a two-line threshold rule reaches comparable reliability and power
saving, the learned controller is dominated and the paper's contribution narrows sharply.

F1 made this sharper, not softer: a flat controller at lr = 1e-4 already reaches 4.62% without any
guard. So "what is the learning for?" now has two independent reasons to be asked, and it is much
better answered by the authors than by a referee.

THE RULE (no parameters fitted, nothing trained):
    reliability branch : HANDOVER if reported RSRP < PROACTIVE_DBM, else STAY_SAVE
    spectrum branch    : RB_HOLD always (the allocation never moves from its initial value)
The environment's own state guard then does the protecting: below the -76 dBm save line it
restores PT_MAX whatever the rule pressed. That is precisely "the guard is the controller".

THREE ARMS, all evaluation-only:
    rule_guard    SAVE-always + 9 dB state guard   <- the arm that could dominate the paper
    rule_noguard  SAVE-always, guard disabled      <- isolates what the guard contributes
    rule_protect  PROTECT-always                   <- the always-max-power floor, a sanity anchor

PROVENANCE. The evaluation loop, the CRN channel-seed patch, the eval seeds and the metric
definitions are the published objects, imported from tablevii_r8 and train_branched_nyu. Only the
policy is replaced, by an object exposing the same .act(obs) interface as the learned network.
"""
import os, sys, csv, time, json

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

PROACTIVE_DBM = -75.0          # the paper's proactive handover trigger
OUTDIR = "results/a2_guard_as_policy"

_saved = sys.argv
sys.argv = [_saved[0]]
import tablevii_r8 as T
sys.argv = _saved

from dndqn_branched import R_STAY_PROTECT, R_STAY_SAVE, R_HANDOVER, S_HOLD


class RulePolicy:
    """The guard as the entire controller. No learned parameters."""

    def __init__(self, mode):
        self.mode = mode          # 'save' | 'protect'
        self.n = 0
        self.n_ho = 0

    def act(self, obs):
        rsrp = float(obs[0]) * 100.0          # obs[0] = rsrp_norm = RSRP/100
        self.n += 1
        if rsrp < PROACTIVE_DBM:
            self.n_ho += 1
            return [R_HANDOVER, S_HOLD]
        a = R_STAY_PROTECT if self.mode == "protect" else R_STAY_SAVE
        return [a, S_HOLD]


ARMS = {
    "rule_guard":   ("Guard as policy (SAVE-always + 9 dB state guard)", "save",    9.0),
    "rule_noguard": ("SAVE-always, no guard",                            "save",    None),
    "rule_protect": ("PROTECT-always (max power floor)",                 "protect", None),
}

METRICS = ["HO", "Power", "Spec", "RLF", "Tput"]


def main():
    S = T.import_stack()
    T.install_patch(S["env_6g_v5"])
    B, tp = S["B"], S["tp"]
    slow, med, fast, skip = tp.load_trajectories(tp.CSV_PATH)
    pool = slow + med
    os.makedirs(OUTDIR, exist_ok=True)
    print("pool=%d | R=%d repeats | %d eval seeds x %d episodes"
          % (len(pool), T.R, len(B.EVAL_SEEDS), B.EVAL_EP))
    print("proactive trigger %.0f dBm | save line -76 dBm (=-92 + 7 + 9)\n" % PROACTIVE_DBM)

    summary = {}
    for arm, (label, mode, gdb) in ARMS.items():
        rows = []
        t0 = time.time()
        for r in range(T.R):
            pol = RulePolicy(mode)
            with T.seeded(r):
                m = B.evaluate_branched(pol, pool, True, guard_db=gdb, guard_mode="state")
            row = {"arm": arm, "label": label, "repeat": r,
                   "chan_seed_first": T.BASE + 100000 * r,
                   "guard_db": gdb, "mode": mode,
                   "ho_intent_share": round(pol.n_ho / max(pol.n, 1), 4)}
            for k in METRICS:
                row[k] = round(float(m[k]), 4)
            rows.append(row)
            print("  [%s] r%d | " % (arm, r) + " ".join("%s %8.3f" % (k, row[k]) for k in METRICS))
        with open(os.path.join(OUTDIR, "unit_%s.csv" % arm), "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        mean = {k: sum(x[k] for x in rows) / len(rows) for k in METRICS}
        sd = {k: (sum((x[k] - mean[k]) ** 2 for x in rows) / len(rows)) ** 0.5 for k in METRICS}
        summary[arm] = {"label": label, "mean": mean, "sd_eval": sd,
                        "ho_intent_share": rows[0]["ho_intent_share"]}
        print("  [%s] done in %.0fs -> " % (arm, time.time() - t0)
              + " ".join("%s %.2f+-%.2f" % (k, mean[k], sd[k]) for k in METRICS) + "\n")

    json.dump(summary, open(os.path.join(OUTDIR, "a2_summary.json"), "w"), indent=2)

    print("=" * 92)
    print("A2  -  WHAT DOES THE LEARNING ADD OVER THE GUARD ALONE?")
    print("=" * 92)
    print("%-46s %8s %8s %8s %8s %9s" % ("arm", "RLF", "Power", "Spec", "HO", "Tput"))
    print("-" * 92)
    for arm in ARMS:
        m = summary[arm]["mean"]
        print("%-46s %8.2f %8.2f %8.2f %8.2f %9.1f"
              % (summary[arm]["label"], m["RLF"], m["Power"], m["Spec"], m["HO"], m["Tput"]))
    print("%-46s %8.2f %8.2f %8.2f %8.2f %9.1f"
          % ("LEARNED guarded controller (Table 1, k=5)", 4.93, 63.48, 80.72, 94.71, 504.0))
    print("%-46s %8.2f" % ("LEARNED flat, lr=1e-4 (F1)", 4.62))
    print("%-46s %8.2f %8.2f %8.2f %8.2f %9.1f"
          % ("3GPP A3 rule (Table 1)", 3.73, 0.00, 0.00, 97.50, 1919.7))
    print()
    g = summary["rule_guard"]["mean"]
    print("READING")
    print("  The rule reaches RLF %.2f%% at %.2f%% power saving," % (g["RLF"], g["Power"]))
    print("  against the learned guarded controller's 4.93%% at 63.48%%.")
    if g["RLF"] <= 5.5 and g["Power"] >= 55.0:
        print("  -> THE RULE IS COMPETITIVE. The learned controller is largely dominated on")
        print("     reliability and power. The paper must say so and re-state what the learning")
        print("     contributes - most plausibly the spectrum axis, which the rule never moves.")
    elif g["RLF"] <= 5.5:
        print("  -> The rule matches on RELIABILITY but not on efficiency. The honest claim is")
        print("     that the guard buys the reliability and the learner buys the resource saving.")
    else:
        print("  -> The rule does NOT reach the learned operating point. The learning is doing")
        print("     work the guard alone cannot, and this arm answers the referee directly.")
    print("\n  Spectrum: the rule holds its initial allocation, saving %.2f%%," % g["Spec"])
    print("  against the learned controller's 80.72%. That gap is not attributable to the guard.")


if __name__ == "__main__":
    main()
