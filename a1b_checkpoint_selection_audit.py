#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
a1b_checkpoint_selection_audit.py

THE QUESTION. Every training saves the checkpoint with the best moving-average reward,
computed on an UNSEEDED channel. Selecting an argmax over a noisy criterion injects a
run-specific positive bias whose size scales with how much that criterion fluctuates.

If the arms that report LARGE between-training variance (flat, binary-unguarded) also
fluctuate more during training than the arm that reports almost NONE (guarded), then
the reported variance difference could be an artefact of the SELECTION RULE rather
than a property of the controller.

WHAT THIS MEASURES, per training run, from the logs alone:
  * post-exploration volatility of mean_rew          (how noisy the criterion is)
  * the step at which 'best' last improved           (late = converged, early = lucky spike)
  * SELECTION PREMIUM = (best - mean(mean_rew)) / sd(mean_rew)
      how many standard deviations above typical the SELECTED point sat.
      A large premium on the volatile arms and a small one on the guarded arm is the
      signature of the confound.

Read-only. Parses logs. Trains nothing, writes nothing except its own report.
"""
import os, re, glob, math, statistics as st
from collections import defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))
ROW = re.compile(
    r"\[\s*(?P<arm>[A-Za-z0-9_]+)\s+s(?P<seed>\d+)\]\s*"
    r"\[\s*(?P<step>\d+)/\s*(?P<total>\d+)\]\s+\S+\s+"
    r"mean_rew=\s*(?P<rew>-?[\d.]+)\s+eps=(?P<eps>[\d.]+)\s+best=\s*(?P<best>-?[\d.eE+]+)"
)
EXPLORE_DONE = 0.06          # eps at/below this = exploration finished
SENTINEL = -1e17             # the "no checkpoint yet" placeholder


def parse():
    runs = defaultdict(list)
    for path in glob.glob(os.path.join(ROOT, "logs", "*.log")) + glob.glob(os.path.join(ROOT, "*.log")):
        try:
            txt = open(path, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        for m in ROW.finditer(txt):
            runs[(m.group("arm"), int(m.group("seed")))].append(
                (int(m.group("step")), float(m.group("rew")),
                 float(m.group("eps")), float(m.group("best")))
            )
    return runs


def analyse(rows):
    rows = sorted(set(rows))
    post = [(s, r, b) for (s, r, e, b) in rows if e <= EXPLORE_DONE]
    if len(post) < 5:
        return None
    rew = [r for (_, r, _) in post]
    mu, sd = st.mean(rew), (st.pstdev(rew) or 1e-9)
    final_best = max(b for (_, _, b) in post if b > SENTINEL) if any(b > SENTINEL for (_, _, b) in post) else None
    if final_best is None:
        return None
    # last step at which best improved
    last_imp, prev = None, None
    for (s, _, b) in post:
        if b <= SENTINEL:
            continue
        if prev is None or b > prev + 1e-9:
            last_imp, prev = s, b
    total = max(s for (s, _, _) in post)
    return {
        "sd_rew": sd,
        "mean_rew": mu,
        "best": final_best,
        "premium": (final_best - mu) / sd,
        "sel_frac": (last_imp / total) if (last_imp and total) else float("nan"),
        "n": len(post),
    }


def main():
    runs = parse()
    if not runs:
        print("No training logs matched. Nothing to audit."); return
    by_arm = defaultdict(list)
    for (arm, seed), rows in runs.items():
        a = analyse(rows)
        if a:
            a["seed"] = seed
            by_arm[arm].append(a)

    print("=" * 96)
    print("CHECKPOINT-SELECTION AUDIT  -  is the reported variance an artefact of the save rule?")
    print("=" * 96)
    print(f"{'arm':22} {'n':>2} {'sd(mean_rew)':>13} {'premium (sd)':>13} {'sel. position':>14}")
    print("-" * 96)
    summary = {}
    for arm in sorted(by_arm):
        A = by_arm[arm]
        if len(A) < 2:
            continue
        sd  = st.mean(a["sd_rew"] for a in A)
        pr  = st.mean(a["premium"] for a in A)
        sf  = st.mean(a["sel_frac"] for a in A if not math.isnan(a["sel_frac"]))
        summary[arm] = (sd, pr, sf, len(A))
        print(f"{arm:22} {len(A):>2} {sd:13.1f} {pr:13.2f} {sf*100:13.0f}%")

    print()
    print("HOW TO READ THIS")
    print("  sd(mean_rew)  - how noisy the selection criterion was during training.")
    print("  premium       - how many sd ABOVE its own typical value the saved checkpoint sat.")
    print("                  Large premium = the save rule caught a lucky spike.")
    print("  sel. position - where in training the saved checkpoint came from.")
    print("                  Near 100% = late and converged (harmless).")
    print("                  Well below = a mid-training spike was kept (suspicious).")
    print()
    print("VERDICT")
    if len(summary) < 2:
        print("  Not enough arms parsed to compare.")
        return
    prem = {a: v[1] for a, v in summary.items()}
    sds  = {a: v[0] for a, v in summary.items()}
    lo_a = min(prem, key=prem.get); hi_a = max(prem, key=prem.get)
    spread = prem[hi_a] - prem[lo_a]
    print(f"  Lowest  selection premium: {lo_a} ({prem[lo_a]:.2f} sd)")
    print(f"  Highest selection premium: {hi_a} ({prem[hi_a]:.2f} sd)")
    print(f"  Spread across arms: {spread:.2f} sd")
    print(f"  Criterion noise ranges {min(sds.values()):.0f} to {max(sds.values()):.0f} across arms.")
    print()
    if spread < 0.75:
        print("  -> Selection premium is SIMILAR across arms. The save rule is not")
        print("     differentially favouring the noisy arms. The confound is small,")
        print("     and this becomes a positive control worth reporting.")
    else:
        print("  -> Selection premium DIFFERS MATERIALLY across arms. The save rule may be")
        print("     contributing to the reported between-training variance difference.")
        print("     Escalate to the full test: retrain flat + guarded with FINAL-checkpoint")
        print("     saving and recompute sigma_train.")
    print()
    print("PER-RUN DETAIL")
    for arm in sorted(by_arm):
        for a in sorted(by_arm[arm], key=lambda x: x["seed"]):
            print(f"  {arm:22} s{a['seed']} | sd={a['sd_rew']:8.1f} best={a['best']:9.1f} "
                  f"premium={a['premium']:5.2f} sel@{a['sel_frac']*100:3.0f}% pts={a['n']}")


if __name__ == "__main__":
    main()
