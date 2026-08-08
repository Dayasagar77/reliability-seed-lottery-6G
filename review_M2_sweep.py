#!/usr/bin/env python3
"""
review_M2_sweep.py -- M2: bound the two stated modelling assumptions the paper does not sweep.

THE REVIEWER'S POINT. The 15 dBi combined array gain is "a stated modelling assumption calibrated
for scenario coherence, not a measured value", and the LOS/NLOS transition is a geometric rule
(d < 0.75R) rather than a physical blockage model. Since the guard constant G = 9 dB and the save
line (-76 dBm) are derived from the RSRP geometry, a 3 dB change in assumed array gain shifts the
entire RSRP landscape and could move the cliff. The alpha-mu and blockage sensitivities are
bounded in IX-G and IX-I; these two are not bounded anywhere.

TWO KNOBS, TWO MECHANISMS.

  ARRAY GAIN is an instance attribute, self.array_gain_dbi, used directly in
  env_6g_nyu._compute_rsrp. It looks patchable via the module constant nyu.ARRAY_GAIN_DBI -- but
  it is NOT: BranchedEnv6G.__init__ hard-codes array_gain_dbi=15.0 as its own default rather than
  reading the module. Patching the constant would have run six identical sweeps and reported a
  flat line as a robustness result. It is injected by wrapping the constructor instead.

  LOS FRACTION is the literal 0.75 inside SubTHzEnv6G_NYU._compute_rsrp. Re-implementing that
  method to parameterise it would violate the project rule that published code is imported and
  never re-implemented. Instead its SOURCE is transformed -- one literal replaced by a knob --
  and the transform is PROVEN behaviour-preserving: with the knob at 0.75 the patched method must
  reproduce the laptop grid bit-exactly before any swept value is trusted.

THE NO-OP GATE. Before sweeping anything, the fully patched stack is evaluated at the published
defaults (15.0 dBi, 0.75) and required to reproduce the laptop Table VII guarded row to four
decimals on all 40 cells. If the machinery perturbs the baseline, every swept number is
measuring the patch rather than the assumption, and the run stops.
"""
import csv
import hashlib
import inspect
import os
import sys
import textwrap
import time

import numpy as np

ARRAY_GAINS = [12.0, 15.0, 18.0]
LOS_FRACS = [0.65, 0.75, 0.85]
BASE_GAIN, BASE_FRAC = 15.0, 0.75
SEEDS = [0, 1, 2, 3, 4]
R = 8
OUT = 'results/m2sweep'
METRICS = ['HO', 'Power', 'Spec', 'RLF', 'Tput']
LOS_FRAC = [BASE_FRAC]                       # mutable knob the transformed method closes over


def log(m):
    print(time.strftime('%Y-%m-%d %H:%M:%S') + ' | ' + m, flush=True)


def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def install_knobs():
    """Wrap the branched constructor for array gain; transform _compute_rsrp for the LOS rule."""
    import env_6g_nyu as EN
    import env_6g_branched as EB

    # --- array gain: inject through the constructor, since the default is hard-coded ---
    if not hasattr(EB.BranchedEnv6G, '_orig_init'):
        EB.BranchedEnv6G._orig_init = EB.BranchedEnv6G.__init__
        GAIN = [BASE_GAIN]

        def _init(self, traj, *a, **kw):
            kw['array_gain_dbi'] = GAIN[0]
            return EB.BranchedEnv6G._orig_init(self, traj, *a, **kw)

        EB.BranchedEnv6G.__init__ = _init
        EB.BranchedEnv6G._gain_knob = GAIN

    # --- LOS fraction: source transform, one literal, proven below ---
    if not hasattr(EN.SubTHzEnv6G_NYU, '_orig_compute_rsrp'):
        src = textwrap.dedent(inspect.getsource(EN.SubTHzEnv6G_NYU._compute_rsrp))
        n = src.count('CELL_RADIUS * 0.75')
        if n != 1:
            raise SystemExit('LOS literal appears %d times in _compute_rsrp, expected exactly 1 '
                             '-- refusing to transform source I do not understand.' % n)
        new = src.replace('CELL_RADIUS * 0.75', 'CELL_RADIUS * _LOS_FRAC[0]')
        g = dict(vars(sys.modules['env_6g_nyu']))
        g['_LOS_FRAC'] = LOS_FRAC
        ns = {}
        exec(new, g, ns)
        EN.SubTHzEnv6G_NYU._orig_compute_rsrp = EN.SubTHzEnv6G_NYU._compute_rsrp
        EN.SubTHzEnv6G_NYU._compute_rsrp = ns['_compute_rsrp']
    return EB.BranchedEnv6G._gain_knob


def main():
    saved = sys.argv
    sys.argv = [saved[0]]
    import torch
    torch.set_num_threads(1)
    import env_6g_v5
    import train_phase2 as tp
    import train_branched_nyu as B
    from dndqn_branched import load_branched
    from dndqn import OBS_DIM
    import tablevii_r8 as T7
    sys.argv = saved

    T7.install_patch(env_6g_v5)
    GAIN = install_knobs()
    slow, med, fast, skip = tp.load_trajectories(tp.CSV_PATH)
    pool = slow + med
    os.makedirs(OUT, exist_ok=True)

    def run(gain, frac, seed):
        GAIN[0], LOS_FRAC[0] = gain, frac
        net = load_branched('br_gst_s%d.pt' % seed, OBS_DIM)[0]
        out = []
        for r in range(R):
            with T7.seeded(r):
                m = B.evaluate_branched(net, pool, True, guard_db=9.0, guard_mode='state')
            out.append({k: float(m[k]) for k in METRICS})
        return out

    # ---------------------------------------------------------------- the no-op gate
    log('NO-OP GATE: patched stack at the published defaults must reproduce the laptop grid')
    lap = {}
    for r in csv.DictReader(open('/home/claude/work/results/tablevii_r8_laptop.csv')):
        if r['arm'] == 'guarded':
            lap[(int(r['seed']), int(r['repeat']))] = {k: float(r[k]) for k in METRICS}
    bad = 0
    for s in SEEDS:
        got = run(BASE_GAIN, BASE_FRAC, s)
        for r in range(R):
            for k in METRICS:
                if abs(got[r][k] - lap[(s, r)][k]) > 1e-4:
                    bad += 1
                    if bad < 6:
                        log('  MISMATCH s%d r%d %s: %.4f vs laptop %.4f'
                            % (s, r, k, got[r][k], lap[(s, r)][k]))
        log('  seed %d: %d/%d cells match' % (s, len(METRICS) * R - bad, len(METRICS) * R))
    if bad:
        log('*** %d cell(s) differ at the defaults. The patch is not a no-op. STOPPING. ***' % bad)
        return 3
    log('  200/200 cells bit-exact — the knobs are inert at 15.0 dBi / 0.75')

    # ---------------------------------------------------------------- the sweeps
    path = os.path.join(OUT, 'm2sweep.csv')
    rows = list(csv.DictReader(open(path))) if os.path.exists(path) else []
    have = {(float(r['array_gain_dbi']), float(r['los_frac']), int(r['seed']), int(r['repeat']))
            for r in rows}
    settings = ([(g, BASE_FRAC) for g in ARRAY_GAINS]
                + [(BASE_GAIN, f) for f in LOS_FRACS if f != BASE_FRAC])
    log('sweeping %d setting(s) x %d seeds x R=%d' % (len(settings), len(SEEDS), R))
    for gain, frac in settings:
        for s in SEEDS:
            if all((gain, frac, s, r) in have for r in range(R)):
                continue
            t0 = time.time()
            got = run(gain, frac, s)
            for r in range(R):
                row = {'array_gain_dbi': gain, 'los_frac': frac, 'seed': s, 'repeat': r,
                       'ckpt': 'br_gst_s%d.pt' % s, 'chan_seed_first': T7.BASE + 100000 * r,
                       'venue': 'cloud-hw-torch2.5.1', 'torch': torch.__version__,
                       'measured_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
                for k in METRICS:
                    row[k] = '%.4f' % got[r][k]
                rows.append(row)
            with open(path + '.tmp', 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            os.replace(path + '.tmp', path)
            log('  gain=%.0f frac=%.2f s%d done (%.0fs)' % (gain, frac, s, time.time() - t0))

    # ---------------------------------------------------------------- report
    sys.path.insert(0, '/home/claude/work')
    import tablevii_decompose as D
    print('\n' + '=' * 92)
    print('M2 -- SENSITIVITY TO THE TWO UNSWEPT MODELLING ASSUMPTIONS (guarded arm, k=5, R=8)')
    print('=' * 92)
    for label, key, grid in (('array gain (dBi)', 'array_gain_dbi', ARRAY_GAINS),
                             ('LOS fraction of R', 'los_frac', LOS_FRACS)):
        print('\n  %-18s %20s %20s %20s' % (label, 'sustained RLF', 'power saving', 'spectrum'))
        for v in grid:
            cell = {}
            for x in rows:
                if float(x[key]) == v and (float(x['los_frac']) == BASE_FRAC
                                           if key == 'array_gain_dbi'
                                           else float(x['array_gain_dbi']) == BASE_GAIN):
                    cell[(int(x['seed']), int(x['repeat']))] = x
            if len(cell) < len(SEEDS) * R:
                print('  %-18.2f  (incomplete: %d/%d cells)' % (v, len(cell), len(SEEDS) * R))
                continue
            outs = []
            for m in ('RLF', 'Power', 'Spec'):
                X = np.array([[float(cell[(s, r)][m]) for r in range(R)] for s in SEEDS])
                d = D.decompose(X)
                sd = ('%.2f' % d['sd_train']) if d['resolvable'] else '<%.2f' % d['sd_train_hi']
                outs.append('%7.2f +- %-7s' % (d['grand'], sd))
            mark = '   <- published' if v in (BASE_GAIN, BASE_FRAC) else ''
            print('  %-18.2f %20s %20s %20s%s' % (v, outs[0], outs[1], outs[2], mark))
    print('\n' + '=' * 92)
    return 0


if __name__ == '__main__':
    sys.exit(main())
