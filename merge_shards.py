#!/usr/bin/env python3
"""
merge_shards.py — concatenate the per-seed shard CSVs into one ledger per family, per venue.

WHY THIS IS NOT JUST `cat`
--------------------------
This campaign exists because the Results in Engineering AE found the paper's comparisons
"methodologically inconsistent": some rows carried training-seed sigma and others carried
evaluation-seed sigma. A careless merge would reintroduce exactly that class of defect in a new
form, so the merge REFUSES rather than guesses:

  * refuses if a family is incomplete — a sigma over 8 of 10 seeds silently reported as "n=10" is
    the same species of error the AE caught
  * refuses if two rows claim the same (method, seed, steps) with different metrics — that means
    two processes wrote the same unit and one of them is not what it says it is
  * refuses if rows inside one merged file disagree about the venue — that is pipeline mixing
    inside a single reported +/-, which is disqualifying in a paper whose headline number IS a
    variance
  * never writes a filename an in-flight cloud campaign is using; the venue suffix guarantees it

Population-form sigma (np.std, ddof=0) is used everywhere, matching the manuscript's stated
convention: "Reported +/- is the standard deviation across seeds (population form)".

Usage:
    python -u merge_shards.py --venue laptop
    python -u merge_shards.py --venue laptop --allow-partial   # summary only, writes nothing
"""
import argparse, csv, glob, os, sys
import numpy as np

import campaign_units as CU

METRICS = ['HO', 'Power', 'Spec', 'RLF', 'Tput']


def read_rows(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def unit_key(family, r):
    if family == 'matched':
        return (r['method'], int(r['seed']), int(r['steps']))
    return (int(r['seed']), int(r['steps']))


def merge_family(family, venue, steps, allow_partial):
    units = [u for u in CU.units(venue=venue, steps=steps) if u['family'] == family]
    found, missing, rows, problems = [], [], [], []

    for u in units:
        if os.path.exists(u['rows_path']):
            rs = read_rows(u['rows_path'])
            if not rs:
                missing.append(u['name'] + ' (file present but empty)')
                continue
            if len(rs) > 1:
                problems.append(f"{u['rows_path']} holds {len(rs)} rows; a shard must hold exactly 1")
            found.append(u['rows_path'])
            rows.extend(rs)
        else:
            missing.append(u['name'])

    # Duplicate units with conflicting numbers => two processes wrote the same training.
    seen = {}
    for r in rows:
        k = unit_key(family, r)
        sig = tuple(r[m] for m in METRICS)
        if k in seen and seen[k] != sig:
            problems.append(f"CONFLICT {k}: {seen[k]} vs {sig}")
        seen[k] = sig

    # Venue consistency: every row in one merged ledger must come from one toolchain.
    venues = sorted({(r.get('pipeline') or '').split('-')[0] for r in rows if r.get('pipeline')})
    if len(venues) > 1:
        problems.append(f"MIXED VENUES in one ledger: {venues}")
    blank = sum(1 for r in rows if not (r.get('pipeline') or ''))
    if blank:
        problems.append(f"{blank} row(s) carry no pipeline tag — provenance unknown, do not report")

    print(f"\n{'=' * 96}\n  {family.upper()} | {len(rows)}/{CU.EXPECTED_N[family]} units | "
          f"venue(s) {venues or ['(none recorded)']}\n{'=' * 96}")
    if missing:
        print(f"  MISSING ({len(missing)}): {', '.join(missing)}")
    for p in problems:
        print(f"  PROBLEM: {p}")

    ok = not missing and not problems and len(rows) == CU.EXPECTED_N[family]
    if rows:
        groups = {}
        for r in rows:
            groups.setdefault(r.get('method', family), []).append(r)
        for label, rs in groups.items():
            rs.sort(key=lambda r: int(r['seed']))
            cells = []
            for m in METRICS:
                v = [float(r[m]) for r in rs]
                cells.append(f"{m} {np.mean(v):7.2f}+/-{np.std(v):5.2f}")
            print(f"  {label:>16} (n={len(rs):2d}) | " + " | ".join(cells))

    if not ok and not allow_partial:
        print(f"  -> NOT WRITTEN. {family} is incomplete or inconsistent; fix before merging.")
        return False
    if allow_partial and not ok:
        print("  -> --allow-partial: summary printed, nothing written.")
        return False

    out = CU.MERGE_TARGETS[family].format(k=steps // 1000, venue=venue)
    fields = list(rows[0].keys())
    rows.sort(key=lambda r: (r.get('method', ''), int(r['seed'])))
    tmp = out + '.tmp'
    with open(tmp, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, out)
    print(f"  -> WROTE {out} ({len(rows)} rows from {len(found)} shards)")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--venue', default='laptop')
    ap.add_argument('--steps', type=int, default=CU.STEPS)
    ap.add_argument('--allow-partial', action='store_true')
    a = ap.parse_args()

    # Hard guard: a merge must never be able to clobber a ledger another campaign is writing.
    live = set(glob.glob('results/*_5seed_*.csv')) | set(glob.glob('results/*_10seed_*.csv'))
    targets = {CU.MERGE_TARGETS[f].format(k=a.steps // 1000, venue=a.venue) for f in CU.EXPECTED_N}
    clash = {t for t in targets if t in live and a.venue not in os.path.basename(t)}
    if clash:
        sys.exit(f"refusing to run: merge target(s) collide with existing ledgers {sorted(clash)}")

    oks = [merge_family(f, a.venue, a.steps, a.allow_partial) for f in
           ['matched', 'lottery', 'guard', 'safedqn']]
    print(f"\n{sum(oks)}/4 families merged.")
    if all(oks):
        print("All four ledgers written. Next: python -u analyze_variance.py")
    else:
        print("Some families are not ready — nothing was written for those.")


if __name__ == '__main__':
    main()
