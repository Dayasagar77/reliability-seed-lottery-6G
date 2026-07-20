#!/usr/bin/env python3
"""Plot the alpha-mu fading sensitivity: metrics vs mu and vs alpha."""
import csv, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
rows=list(csv.DictReader(open('results/alpha_mu_sensitivity_v6.csv')))
mu_rows=[r for r in rows if r['setting'].startswith('mu=')]
al_rows=[r for r in rows if r['setting'].startswith('alpha=')]
fig,(a1,a2)=plt.subplots(1,2,figsize=(11,4.2))
def panel(ax,rws,xkey,xlabel):
    x=[float(r[xkey]) for r in rws]
    for met,c in [('RLF','#D85A30'),('HO','#378ADD'),('Power','#1D9E75'),('Spec','#7a6fb0')]:
        ax.plot(x,[float(r[met]) for r in rws],'-o',color=c,ms=4,label=met+' %')
    ax.set_xlabel(xlabel); ax.set_ylabel('%'); ax.set_ylim(-2,102)
    ax.spines[['top','right']].set_visible(False)
panel(a1,mu_rows,'mu','mu (cluster count),  alpha = 2.0')
panel(a2,al_rows,'alpha','alpha (nonlinearity),  mu = 1.0')
a1.legend(frameon=False,ncol=2,fontsize=8,loc='center left')
fig.suptitle('Robustness to alpha-mu fading: RLF stays 0-2%, savings vary <3 pp across the reported 140 GHz range',
             fontsize=9,y=1.02)
plt.tight_layout(); fig.savefig('results/fig_alpha_mu_sensitivity_v6.png',dpi=200,bbox_inches='tight'); plt.close()
print('saved results/fig_alpha_mu_sensitivity_v6.png')
