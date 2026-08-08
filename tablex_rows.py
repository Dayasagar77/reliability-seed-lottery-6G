#!/usr/bin/env python3
"""tablex_rows.py -- recompute Table VII / Table X learned rows from the seeded R=8 CRN grids."""
import csv, os, sys, math, hashlib, collections
BASE = sys.argv[1] if len(sys.argv) > 1 else '.'
T7 = os.path.join(BASE, 'tablevii_r8_laptop.csv')
TX = os.path.join(BASE, 'tablex_r8_laptop.csv')
METRICS = ['HO', 'Power', 'Spec', 'RLF', 'Tput']
FMT = {'HO': '%.2f', 'Power': '%.2f', 'Spec': '%.2f', 'RLF': '%.2f', 'Tput': '%.1f'}
def load(p):
    with open(p, newline='') as fh: return list(csv.DictReader(fh))
def md5(p): return hashlib.md5(open(p,'rb').read()).hexdigest()
def seed_means(rows, arm):
    per = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        if r['arm'] != arm: continue
        for m in METRICS: per[int(r['seed'])][m].append(float(r[m]))
    census = {s: len(per[s]['HO']) for s in per}
    return ({s: {m: sum(per[s][m])/len(per[s][m]) for m in METRICS} for s in sorted(per)}, census)
def msd(v, ddof=0):
    n=len(v); mu=sum(v)/n
    return mu, math.sqrt(sum((x-mu)**2 for x in v)/(n-ddof))
def emit(name, sm, ddof=0):
    cells, stats = [], {}
    for m in METRICS:
        mu, sd = msd([sm[s][m] for s in sorted(sm)], ddof); stats[m]=(mu,sd)
        cells.append((FMT[m]+' +- '+FMT[m]) % (mu,sd))
    print('  %-44s n=%-3d| %s' % (name, len(sm), ' | '.join('%-17s' % c for c in cells)))
    return stats
print('='*138); print('TABLE VII / TABLE X LEARNED ROWS -- seeded R=8 CRN grids'); print('='*138)
for p in (T7,TX): print('  %-26s %7d bytes   md5 %s' % (os.path.basename(p), os.stat(p).st_size, md5(p)))
r7, rx = load(T7), load(TX)
print('  t7 %d rows %s venue %s' % (len(r7), sorted(set(x['arm'] for x in r7)), sorted(set(x['venue'] for x in r7))))
print('  tx %d rows %s venue %s' % (len(rx), sorted(set(x['arm'] for x in rx)), sorted(set(x['venue'] for x in rx))))
print(); print('-'*138); print('  CENSUS'); bad=0
for tag, rows in (('t7',r7),('tx',rx)):
    for arm in sorted(set(x['arm'] for x in rows)):
        sm,c = seed_means(rows,arm); ok = len(sm)==5 and set(c.values())=={8}; bad += 0 if ok else 1
        print('    %-3s %-14s seeds=%d repeats=%-8s %s' % (tag,arm,len(sm),sorted(set(c.values())),'ok' if ok else '!! CHECK'))
print('    CENSUS: %s' % ('PASS' if bad==0 else 'FAIL')); print()
LBL7={'flat':'Flat Dn-DQN (6-action)','incremental':'Branched incremental (BDQ-style)','binary':'Branched binary, unguarded','guarded':'Branched binary + state guard (proposed)'}
LBLX={'m5_dndqn':'Proposed Dn-DQN (matched obj.) [family B]','m5_ddqn':'DDQN+PER','m5_std':'Standard DQN','sd5_safe':'Shielded Safe-DQN'}
for DD in (0,1):
    print('-'*138); print('DDOF = %d   (%s)' % (DD, 'POPULATION sd -- manuscript convention' if DD==0 else 'SAMPLE sd -- for comparison only')); print('-'*138)
    t7s={}; smA=None
    for arm in ('flat','incremental','binary','guarded'):
        sm,_ = seed_means(r7,arm)
        if arm=='flat': smA=sm
        t7s[arm]=emit('T7  '+LBL7[arm], sm, DD)
    txs={}; smB=None
    for arm in ('m5_dndqn','m5_ddqn','m5_std','sd5_safe'):
        sm,_ = seed_means(rx,arm)
        if arm=='m5_dndqn': smB=sm
        txs[arm]=emit('TX  '+LBLX[arm], sm, DD)
    pooled={}
    for s in sorted(smA): pooled[('A',s)]=smA[s]
    for s in sorted(smB): pooled[('B',s)]=smB[s]
    emit('POOLED n=10 Proposed Dn-DQN', pooled, DD)
    print()
