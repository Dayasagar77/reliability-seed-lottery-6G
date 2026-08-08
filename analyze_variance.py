#!/usr/bin/env python3
"""
analyze_variance.py — formal test of the paper's CENTRAL claim, which is a claim about VARIANCE.

THE CLAIM
---------
Identical trainings of the flat controller land in different reliability regimes (the "seed
lottery"); the measurement-guarded branched controller does not. In the published 5-seed data that
is sigma_RLF = 4.18 against 0.92 -- a 4.5x spread ratio.

WHAT THE AE SAID, AND WHY IT LANDS
----------------------------------
Results in Engineering AE (RINENG-D-26-14706, 1 Aug 2026): "the evidence base is limited to five
training seeds". At n = 5 a spread can be DISPLAYED but not TESTED: the sampling distribution of a
variance estimate is wide, and a 4.5x ratio drawn from two n=5 samples is not, by itself, evidence
that the underlying variances differ. This script does the test the claim actually needs, at n = 10
per arm.

WHAT IS COMPUTED
----------------
1. Brown-Forsythe (Levene, MEDIAN-centred) -- PRIMARY test of H0: equal variances. Median centring
   is used because it is the variant that keeps its nominal error rate under non-normality, which
   is exactly the regime here (RLF is a bounded rate, and the flat arm is visibly skewed).
2. Levene (MEAN-centred) and Fligner-Killeen -- secondary, reported for transparency, never used to
   pick the answer after the fact.
   Bartlett's test and the classical F-ratio of variances are DELIBERATELY NOT USED as the primary:
   both are severely non-robust to departures from normality and would overstate significance here.
   The F-ratio is printed only as a descriptive quantity, explicitly labelled as such.
3. Percentile bootstrap CI on the SIGMA RATIO sigma_flat / sigma_guard, resampling seeds within each
   arm independently. This is what converts "we observed a 4.5x spread" into "the spread ratio is
   4.5x, 95% CI [a, b]" -- an interval a reader can argue with.
4. Bootstrap CI on each arm's sigma separately.
5. Welch t-test on the MEANS (unequal variances; arms are different architectures, so this is an
   unpaired comparison -- the seed index does not pair anything meaningful once the action space
   differs. Paired tests are reserved, as elsewhere in the paper, for same-architecture ablations).

SIGMA CONVENTION
----------------
The manuscript states verbatim: "Reported +/- is the standard deviation across seeds (population
form)". All reported sigma here is population form (ddof=0), matching. The test statistics
themselves use whatever scipy defines internally; that is a different quantity and is not affected.

PIPELINE HYGIENE  (revised 2026-08-01 -- see REVISION NOTE)
-----------------------------------------------------------
Only same-pipeline data enters a test. Everything else is printed as an independent cross-pipeline
reproduction check and is never pooled into the tested sample.

REVISION NOTE (2026-08-01)
--------------------------
The previous version hardcoded the CLOUD filenames as the tested sample and labelled its output
"pipeline = cloud single-pass". That was written when the cloud was the only campaign. It is now
the REPLICATION arm: the reporting pipeline is the laptop (torch 2.5.1), because every number
already in the manuscript came off that toolchain, and mixing torch 2.13 rows into the same table
would be a quieter version of the very inconsistency the AE rejected the paper for. Run as written,
the old script would have found no laptop data at all or -- worse -- silently reported the cloud
arm as the paper's result.

Three things changed. Nothing statistical: BOOT_SEED, BOOT_N, CI and every test are byte-identical,
so a cloud run of this file reproduces the old numbers exactly.

  1. --venue selects the pipeline (default: laptop). Paths come from campaign_units.MERGE_TARGETS,
     not from literals, so the plan stays the single source of truth.
  2. PROVENANCE IS THE pipeline COLUMN, NOT THE FILENAME. Only `guard` shards carry a venue suffix
     in their names; a lottery shard has the SAME filename in both venues. Filtering by filename
     alone would therefore have been unsafe the moment laptop and cloud results sat in one folder.
     Every row is checked against the requested venue's tag and mismatches are dropped out loud.
  3. Output is results/variance_tests_<venue>.json, so analysing one arm cannot overwrite the
     recorded result of the other.

Usage:  python -u analyze_variance.py                  # the reporting pipeline (laptop)
        python -u analyze_variance.py --venue cloud    # the independent replication arm
"""
import argparse, os, csv, json
import numpy as np
from scipy import stats

import campaign_units as CU

BOOT_SEED = 20260801        # fixed so the reported CI is exactly reproducible
BOOT_N = 20_000
CI = 95.0

# Pre-instrumentation cloud ledgers: written before PIPELINE_TAG existed, so their rows carry no
# pipeline column. Their provenance is known from the campaign that wrote them, but it is NOT
# recorded IN them, and that distinction is printed every time they are used.
LEGACY_CLOUD = {
    'lottery': ['results/lottery_flat_10seed_800k.csv'],
    'guard':   ['results/lottery_guard_800k_s0_4.csv', 'results/lottery_guard_800k_s5_9.csv'],
    'matched': ['results/matched_rel_5seed_800k.csv'],
    'safedqn': ['results/safedqn_5seed_800k.csv'],
}


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def col(rows, key):
    return np.array([float(r[key]) for r in rows], dtype=float)


def psd(x):
    """population-form sigma, the manuscript's stated convention"""
    return float(np.std(np.asarray(x, dtype=float)))


# --------------------------------------------------------------------------------------------
# Sourcing. Which rows are the tested sample, and how do we know they are all one toolchain?
# --------------------------------------------------------------------------------------------
def gather(family, venue, steps=CU.STEPS):
    """Return (rows, sources, notes) for one arm of one venue.

    Tier order, and it matters -- whichever tier answers first is the WHOLE sample; tiers are
    never mixed, because a half-merged half-shard sample is exactly the "some rows from here,
    some from there" defect this campaign exists to remove:

      1. the merged ledger merge_shards.py writes for this venue (the audited artefact)
      2. this venue's per-seed shards, when the merge has not run yet (lets us look at a
         campaign while it is still in flight, without pretending the merge happened)
      3. cloud only: the legacy pre-instrumentation filenames the cloud campaign actually uses

    Rows are then filtered by the pipeline TAG, not the filename (see REVISION NOTE 2).
    """
    notes = []
    merged = CU.MERGE_TARGETS[family].format(k=steps // 1000, venue=venue)
    shard_paths = sorted(u['rows_path'] for u in CU.units(venue=venue, steps=steps)
                         if u['family'] == family)
    legacy_paths = LEGACY_CLOUD.get(family, []) if venue == 'cloud' else []
    rows, sources, tier = read_csv(merged), [merged], 'merged ledger'

    if not rows:
        got = [(p, rs) for p, rs in ((q, read_csv(q)) for q in shard_paths) if rs]
        if got:
            rows = [r for _, rs in got for r in rs]
            sources, tier = [p for p, _ in got], 'per-seed shards (merge not yet run)'
            notes.append(f"{len(got)}/{len(shard_paths)} shards present — this is a LOOK AT A "
                         f"CAMPAIGN IN FLIGHT, not a merged result; do not report from it")

    if not rows and legacy_paths:
        got = [(p, rs) for p, rs in ((q, read_csv(q)) for q in legacy_paths) if rs]
        if got:
            rows = [r for _, rs in got for r in rs]
            sources, tier = [p for p, _ in got], 'legacy pre-instrumentation cloud ledger'
            notes.append("these rows carry NO pipeline column; their venue is known from the "
                         "campaign that wrote them, not recorded inside them")

    if not rows:
        # Say what was looked for. "n=0" next to a path that does not exist reads like a defect;
        # it is usually just a campaign that has not landed yet, and the difference matters.
        tier = 'NO DATA — none of these exist yet'
        sources = [merged] + shard_paths[:2] + (['...'] if len(shard_paths) > 2 else []) \
            + legacy_paths

    # --- provenance filter: the tag decides, never the filename -------------------------------
    kept, wrong, blank = [], {}, 0
    legacy = tier.startswith('legacy')
    for r in rows:
        tag = (r.get('pipeline') or '').strip()
        if not tag:
            if legacy:
                kept.append(r)                 # provenance asserted by tier 3, and printed as such
            else:
                blank += 1
            continue
        if tag.split('-')[0] == venue:
            kept.append(r)
        else:
            wrong[tag] = wrong.get(tag, 0) + 1

    if blank:
        notes.append(f"DROPPED {blank} row(s) with no pipeline tag — provenance unknown, "
                     f"do not report")
    for tag, n in sorted(wrong.items()):
        notes.append(f"DROPPED {n} row(s) tagged '{tag}' — not this venue")

    resumed = sorted({(r.get('pipeline') or '').strip() for r in kept
                      if (r.get('pipeline') or '').strip()
                      and (r.get('pipeline') or '').strip() != f'{venue}-singlepass'})
    if resumed:
        n_res = sum(1 for r in kept if (r.get('pipeline') or '').strip() != f'{venue}-singlepass')
        notes.append(f"{n_res}/{len(kept)} row(s) carry {resumed} (mid-seed restart) rather than "
                     f"'{venue}-singlepass'. Same toolchain, interrupted run. DISCLOSE.")

    if 'seed' in (kept[0] if kept else {}):
        kept.sort(key=lambda r: int(r['seed']))
        seeds = [int(r['seed']) for r in kept]
        if len(seeds) != len(set(seeds)):
            dup = sorted({s for s in seeds if seeds.count(s) > 1})
            raise SystemExit(f"FATAL: duplicate seed(s) {dup} in {family}/{venue} across {sources}")

    print(f"\n  {family.upper()} <- {tier}")
    for p in sources:
        print(f"      {p}")
    for n in notes:
        print(f"      ! {n}")
    return kept, sources, notes


def boot_sigma_ratio(a, b, n=BOOT_N, seed=BOOT_SEED):
    """Percentile bootstrap on sigma(a)/sigma(b), resampling seeds within each arm."""
    rng = np.random.default_rng(seed)
    a = np.asarray(a, float); b = np.asarray(b, float)
    ra = rng.integers(0, len(a), size=(n, len(a)))
    rb = rng.integers(0, len(b), size=(n, len(b)))
    sa = a[ra].std(axis=1)            # population form, matches the paper
    sb = b[rb].std(axis=1)
    keep = sb > 0
    ratio = sa[keep] / sb[keep]
    lo, hi = np.percentile(ratio, [(100 - CI) / 2, 100 - (100 - CI) / 2])
    return float(lo), float(hi), int(keep.sum()), float(np.median(ratio))


def boot_sigma(a, n=BOOT_N, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    a = np.asarray(a, float)
    s = a[rng.integers(0, len(a), size=(n, len(a)))].std(axis=1)
    lo, hi = np.percentile(s, [(100 - CI) / 2, 100 - (100 - CI) / 2])
    return float(lo), float(hi)


def describe(name, x):
    x = np.asarray(x, float)
    if len(x) == 0:
        print(f"  {name:<34} n=0   (no data yet)")
        return
    lo, hi = boot_sigma(x) if len(x) >= 3 else (float('nan'), float('nan'))
    print(f"  {name:<34} n={len(x):<3} mean {x.mean():6.2f}  sigma {psd(x):5.2f} "
          f"[boot {CI:.0f}% {lo:.2f}, {hi:.2f}]  span {x.min():5.2f}-{x.max():5.2f}")


def variance_tests(flat, guard, label_a='flat', label_b='guarded'):
    out = {}
    print()
    print("-" * 96)
    print(f"  VARIANCE TESTS  H0: sigma^2({label_a}) == sigma^2({label_b})   "
          f"n={len(flat)} vs n={len(guard)}")
    print("-" * 96)
    if len(flat) < 3 or len(guard) < 3:
        print("  insufficient data (need n>=3 per arm)")
        return out

    bf = stats.levene(flat, guard, center='median')      # Brown-Forsythe
    lv = stats.levene(flat, guard, center='mean')        # classical Levene
    fk = stats.fligner(flat, guard)                      # Fligner-Killeen
    print(f"  Brown-Forsythe (median-centred)  W = {bf.statistic:8.4f}   p = {bf.pvalue:.5f}   <-- PRIMARY")
    print(f"  Levene (mean-centred)            W = {lv.statistic:8.4f}   p = {lv.pvalue:.5f}")
    print(f"  Fligner-Killeen (rank-based)     X2= {fk.statistic:8.4f}   p = {fk.pvalue:.5f}")
    out['brown_forsythe'] = {'W': float(bf.statistic), 'p': float(bf.pvalue)}
    out['levene_mean'] = {'W': float(lv.statistic), 'p': float(lv.pvalue)}
    out['fligner'] = {'X2': float(fk.statistic), 'p': float(fk.pvalue)}

    sa, sb = psd(flat), psd(guard)
    lo, hi, nb, med = boot_sigma_ratio(flat, guard)
    print()
    print(f"  sigma({label_a}) = {sa:.3f}   sigma({label_b}) = {sb:.3f}   "
          f"point ratio = {sa / sb:.2f}x")
    print(f"  bootstrap {CI:.0f}% CI on the sigma RATIO: [{lo:.2f}, {hi:.2f}]   "
          f"(median {med:.2f}x, {BOOT_N:,} resamples, {nb:,} usable, rng seed {BOOT_SEED})")
    print(f"  -> CI excludes 1.0: {'YES' if lo > 1.0 else 'NO'}")
    out['sigma'] = {label_a: sa, label_b: sb, 'ratio': sa / sb,
                    'ci_lo': lo, 'ci_hi': hi, 'excludes_1': bool(lo > 1.0),
                    'boot_n': BOOT_N, 'boot_seed': BOOT_SEED}

    # descriptive only -- NOT the primary test (assumes normality, non-robust)
    va, vb = np.var(flat, ddof=1), np.var(guard, ddof=1)
    F = va / vb
    pF = 2 * min(stats.f.cdf(F, len(flat) - 1, len(guard) - 1),
                 1 - stats.f.cdf(F, len(flat) - 1, len(guard) - 1))
    print(f"  [descriptive only, non-robust] F = {F:.2f}  (p = {pF:.5f}) -- NOT used as the claim")
    out['f_ratio_descriptive'] = {'F': float(F), 'p': float(pF)}

    t = stats.ttest_ind(flat, guard, equal_var=False)    # Welch on the means
    pooled = np.sqrt((np.var(flat, ddof=1) + np.var(guard, ddof=1)) / 2)
    d = (np.mean(flat) - np.mean(guard)) / pooled if pooled > 0 else float('nan')
    print()
    print(f"  MEANS (Welch, unpaired): delta = {np.mean(flat) - np.mean(guard):+.2f} pp   "
          f"t = {t.statistic:+.3f}   p = {t.pvalue:.5f}   Cohen d = {d:+.2f}")
    out['means_welch'] = {'delta': float(np.mean(flat) - np.mean(guard)),
                          't': float(t.statistic), 'p': float(t.pvalue), 'd': float(d)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--venue', default='laptop',
                    help="pipeline to TEST. laptop = the reporting arm (torch 2.5.1, the paper's "
                         "toolchain); cloud = the independent replication arm (torch 2.13)")
    ap.add_argument('--steps', type=int, default=CU.STEPS)
    ap.add_argument('--metric', default='RLF')
    a = ap.parse_args()

    other = 'cloud' if a.venue == 'laptop' else 'laptop'
    role = ("the REPORTING pipeline — this is what the paper states"
            if a.venue == 'laptop' else
            "the REPLICATION arm — context only, NOT the paper's reported numbers")

    print("=" * 96)
    print(f"  SEED-LOTTERY VARIANCE ANALYSIS — sustained {a.metric} (%) across TRAINING seeds")
    print(f"  venue = {a.venue}  ({role})")
    print("=" * 96)
    print("\n  SOURCES")

    flat_rows, flat_src, _ = gather('lottery', a.venue, a.steps)
    guard_rows, guard_src, _ = gather('guard', a.venue, a.steps)

    flat = col(flat_rows, a.metric) if flat_rows else np.array([])
    guard = col(guard_rows, a.metric) if guard_rows else np.array([])

    print(f"\n  TESTED SAMPLE  (pipeline = {a.venue} single-pass, both arms, {a.steps:,} steps)")
    describe("flat reliability-tilted Dn-DQN", flat)
    describe("branched + 9 dB measured guard", guard)
    for arm, rows, n_exp in (('flat', flat_rows, CU.EXPECTED_N['lottery']),
                             ('guarded', guard_rows, CU.EXPECTED_N['guard'])):
        if 0 < len(rows) < n_exp:
            print(f"    INCOMPLETE: {arm} has {len(rows)}/{n_exp} seeds. Any sigma below is a "
                  f"preview of a running campaign, NOT a reportable result.")

    res = variance_tests(flat, guard) if len(flat) >= 3 and len(guard) >= 3 else {}

    # ---------- independent cross-pipeline reproduction checks (NEVER pooled) ----------
    print()
    print("-" * 96)
    print("  CROSS-PIPELINE REPRODUCTION CHECKS — reported for context, NOT pooled into any test")
    print("-" * 96)
    o_flat, _, _ = gather('lottery', other, a.steps)
    o_guard, _, _ = gather('guard', other, a.steps)
    print()
    describe(f"flat, {other} pipeline", col(o_flat, a.metric) if o_flat else [])
    describe(f"guarded, {other} pipeline", col(o_guard, a.metric) if o_guard else [])
    pub = read_csv('results/multiseed_rel_nyu.csv')
    describe("flat, laptop 5-seed (published)", col(pub, a.metric) if pub else [])
    pg = read_csv('results/srdndqn_postguard.csv')
    describe("guarded, cloud CHUNKED (Sec IX-K)", col(pg, a.metric) if pg else [])
    print("  manuscript published values: flat 11.47 +/- 4.18 (laptop) | "
          "branched-guarded 4.61 +/- 0.92 (laptop)")

    if res:
        os.makedirs('results', exist_ok=True)
        out = f'results/variance_tests_{a.venue}.json'
        payload = {'venue': a.venue, 'steps': a.steps, 'metric': a.metric,
                   'sources': {'flat': flat_src, 'guard': guard_src},
                   'pipeline_tags': sorted({(r.get('pipeline') or '') or '(untagged)'
                                            for r in flat_rows + guard_rows}),
                   'flat_rlf': [float(x) for x in flat], 'guard_rlf': [float(x) for x in guard],
                   'n_flat': int(len(flat)), 'n_guard': int(len(guard)),
                   'complete': bool(len(flat) == CU.EXPECTED_N['lottery']
                                    and len(guard) == CU.EXPECTED_N['guard']),
                   'tests': res}
        with open(out, 'w') as f:
            json.dump(payload, f, indent=2)
        print(f"\n  -> {out}"
              + ("" if payload['complete'] else "   (marked complete=false — campaign still running)"))


if __name__ == "__main__":
    main()
