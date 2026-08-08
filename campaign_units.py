#!/usr/bin/env python3
"""
campaign_units.py — the ONE definition of what the revision campaign is: 40 trainings.

WHY A SEPARATE MODULE
---------------------
Three things need to agree on the same 40 units and the same 40 output paths:

    laptop_runner.py         decides what to launch and when a unit is finished
    merge_shards.py          decides which shard files to concatenate, and notices missing ones
    verify_shard_parsing.py  cross-checks these paths against what the DRIVERS actually compute

If the plan lived in the runner, a typo in a path would look like "that seed never finished" and
would be silently retrained forever, or worse, silently dropped from the merge. Keeping it here,
with the verifier asserting driver-vs-plan agreement for all 40, makes that class of error loud.

THE CAMPAIGN (all at 800,000 steps, the budget of every headline result in the paper)

    matched  3 methods x 5 seeds = 15   every learned row gets TRAINING-seed sigma (AE point b)
    lottery  1 arm    x 10 seeds = 10   flat arm of the variance claim at n=10 (AE point a)
    guard    1 arm    x 10 seeds = 10   guarded arm of the variance claim at n=10
    safedqn  1 arm    x  5 seeds =  5   the shielded baseline, retrained at the common budget
                                   --
                                   40

est_min is a SCHEDULING HINT ONLY — derived from single-job cloud throughput measurements
(branched ~290 steps/s, flat ~511, ddqn ~869) purely to run the long units first so the tail of the
campaign is not one 45-minute job finishing alone. It is never reported anywhere and no result
depends on it.
"""

STEPS = 800_000

# (family, driver script, method key or None, seed, est_min at single-job cloud throughput)
_PLAN = (
    [('matched', 'campaign_matched.py', 'dndqn', s, 26) for s in range(5)]
    + [('matched', 'campaign_matched.py', 'ddqn', s, 16) for s in range(5)]
    + [('matched', 'campaign_matched.py', 'std', s, 14) for s in range(5)]
    + [('lottery', 'campaign_lottery.py', None, s, 26) for s in range(10)]
    + [('guard', 'campaign_guard.py', None, s, 46) for s in range(10)]
    + [('safedqn', 'campaign_safedqn.py', None, s, 30) for s in range(5)]
)


def rows_path(family, method, seed, venue='laptop', steps=STEPS):
    """Must match, character for character, what each driver computes for the same argv+env.

    verify_shard_parsing.py asserts this for all 40 units by exec'ing the drivers themselves.
    """
    k = steps // 1000
    if family == 'matched':
        return f'results/matched_rel_{k}k_{method}_s{seed}_{seed}.csv'
    if family == 'lottery':
        return f'results/lottery_flat_{k}k_s{seed}_{seed}.csv'
    if family == 'safedqn':
        return f'results/safedqn_{k}k_s{seed}_{seed}.csv'
    if family == 'guard':
        vsuf = '' if venue == 'cloud' else f'_{venue}'
        return f'results/lottery_guard_{k}k_s{seed}_{seed}{vsuf}.csv'
    raise ValueError(family)


def units(venue='laptop', steps=STEPS):
    """The 40 units, longest first. Each is one process, one seed, one output file."""
    out = []
    for family, script, method, seed, est in _PLAN:
        arg = f'{method}:{seed}-{seed}' if family == 'matched' else f'{seed}-{seed}'
        out.append({
            'name': f'{family}_{method}_s{seed}' if method else f'{family}_s{seed}',
            'family': family,
            'script': script,
            'argv': [arg, str(steps)],
            'method': method,
            'seed': seed,
            'steps': steps,
            'rows_path': rows_path(family, method, seed, venue, steps),
            'est_min': est,
        })
    out.sort(key=lambda u: -u['est_min'])          # long jobs first: shortens the tail
    return out


# Merge targets. The cloud ledgers (matched_rel_5seed_800k.csv, lottery_flat_10seed_800k.csv,
# safedqn_5seed_800k.csv, lottery_guard_800k_s0_4.csv, lottery_guard_800k_s5_9.csv) are NEVER
# written by the merge — the venue suffix guarantees a different filename, so a laptop merge can
# not overwrite an in-flight cloud result.
MERGE_TARGETS = {
    'matched': 'results/matched_rel_5seed_{k}k_{venue}.csv',
    'lottery': 'results/lottery_flat_10seed_{k}k_{venue}.csv',
    'guard':   'results/lottery_guard_10seed_{k}k_{venue}.csv',
    'safedqn': 'results/safedqn_5seed_{k}k_{venue}.csv',
}
EXPECTED_N = {'matched': 15, 'lottery': 10, 'guard': 10, 'safedqn': 5}

if __name__ == '__main__':
    us = units()
    print(f"{len(us)} units, {sum(u['est_min'] for u in us)} single-job minutes of work "
          f"({sum(u['est_min'] for u in us) / 60:.1f} h serial)")
    for u in us:
        print(f"  {u['est_min']:3d}m  {u['name']:<22} {u['script']} {' '.join(u['argv'])}"
              f"  -> {u['rows_path']}")
