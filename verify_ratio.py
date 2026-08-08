import csv, sys
import numpy as np
sys.path.insert(0,'.')
import tablevii_decompose as D
R=8; SEEDS=[0,1,2,3,4]
def dec(path,n,m):
    rows=[r for r in csv.DictReader(open(path)) if int(r['n310'])==n]
    cell={(int(r['seed']),int(r['repeat'])):float(r[m]) for r in rows}
    X=np.array([[cell[(s,rr)] for rr in range(R)] for s in SEEDS])
    return D.decompose(X)
print('%-6s %-28s %-28s %s'%('N310','flat RLF sd_train','guarded RLF sd_train (or hi)','collapse ratio'))
for n in (2,3,5):
    f=dec('results/n310/n310_flat.csv',n,'RLF'); g=dec('results/n310/n310.csv',n,'RLF')
    gv = g['sd_train'] if g['resolvable'] else g['sd_train_hi']
    tag = '' if g['resolvable'] else ' (upper bound -> ratio is a LOWER bound)'
    print('%-6d %-28.6f %-28.6f %.3f%s'%(n,f['sd_train'],gv,f['sd_train']/gv,tag))
    print('       flat resolvable=%s  guarded resolvable=%s'%(f['resolvable'],g['resolvable']))
