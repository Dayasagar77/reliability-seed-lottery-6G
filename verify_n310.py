import csv, sys
import numpy as np
sys.path.insert(0,'.')
import tablevii_decompose as D
R=8; SEEDS=[0,1,2,3,4]; METRICS=['HO','Power','Spec','RLF','Tput']
for path,label in (('results/n310/n310.csv','GUARDED'),('results/n310/n310_flat.csv','FLAT')):
    rows=list(csv.DictReader(open(path)))
    print('='*90); print(label, path, len(rows),'rows'); print('='*90)
    for n in sorted({int(r['n310']) for r in rows}):
        sub=[r for r in rows if int(r['n310'])==n]
        line='  N310=%d  ' % n
        percell={}
        for m in METRICS:
            cell={(int(r['seed']),int(r['repeat'])):float(r[m]) for r in sub}
            X=np.array([[cell[(s,rr)] for rr in range(R)] for s in SEEDS])
            d=D.decompose(X); percell[m]=d
            sd = ('%.2f'%d['sd_train']) if d['resolvable'] else '<%.2f'%d['sd_train_hi']
            if m in ('RLF','Power','Spec'):
                line += '%s %.2f +- %-7s ' % (m, d['grand'], sd)
        # per-seed means for span/mean
        cell={(int(r['seed']),int(r['repeat'])):float(r['RLF']) for r in sub}
        pers=np.array([[cell[(s,rr)] for rr in range(R)] for s in SEEDS]).mean(axis=1)
        span=pers.max()-pers.min()
        print(line)
        print('        RLF per-seed means %s  span %.4f  mean %.4f  span/mean %.4f'
              % (np.round(pers,3).tolist(), span, pers.mean(), span/pers.mean()))
