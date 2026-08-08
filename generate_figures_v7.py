#!/usr/bin/env python3
"""generate_figures_v7.py — corrected results figures from verified numbers."""
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
O='/tmp/dndqn-clean/results/'
def bars(ax,x,series,labels,colors,w):
    n=len(series); off=np.linspace(-(n-1)/2,(n-1)/2,n)*w
    for k,(s,l,c) in enumerate(zip(series,labels,colors)):
        ax.bar(x+off[k],s,w,label=l,color=c)
        for i in x: ax.text(i+off[k],s[i]+1,f'{s[i]:.0f}',ha='center',fontsize=7)
    ax.spines[['top','right']].set_visible(False)

# Fig 8 — performance comparison (3 methods)
m=['3GPP A3','PG-RRM','Dn-DQN\n(proposed)']; x=np.arange(3)
fig,ax=plt.subplots(figsize=(7,4.2))
bars(ax,x,[[98.8,98.2,95.3],[0.0,12.7,77.9],[0.0,0.0,79.5]],
     ['HO success','Power saving','Spectrum saving'],['#378ADD','#1D9E75','#D85A30'],0.25)
ax.set_xticks(x); ax.set_xticklabels(m); ax.set_ylabel('%'); ax.set_ylim(0,108)
ax.legend(frameon=False,ncol=3,fontsize=9,loc='upper center')
ax.set_title('Performance comparison (5 seeds × 150 ep, corrected env)',fontsize=10)
plt.tight_layout(); fig.savefig(O+'fig8_performance_v7.png',dpi=200); plt.close()

# Fig 12 — throughput
mth=['3GPP A3','PG-RRM','DDQN+PER','Std DQN','Dn-DQN']; tp=[414.0,368.3,117.5,113.8,78.1]
fig,ax=plt.subplots(figsize=(7,4))
ax.bar(range(5),tp,color=['#8a8a8a','#8a8a8a','#7a6fb0','#7a6fb0','#D85A30'])
for i,v in enumerate(tp): ax.text(i,v+8,f'{v:.0f}',ha='center',fontsize=8)
ax.set_xticks(range(5)); ax.set_xticklabels(mth,fontsize=9); ax.set_ylabel('Per-UE throughput (Mbps)')
ax.set_title('Per-UE Shannon throughput (corrected 380.16 MHz occupied-BW basis)',fontsize=9.5)
ax.spines[['top','right']].set_visible(False)
plt.tight_layout(); fig.savefig(O+'fig12_throughput_v7.png',dpi=200); plt.close()

# Fig 13 — n-step sensitivity
ns=['n=1','n=3','n=5']; x=np.arange(3)
fig,ax=plt.subplots(figsize=(6,4))
bars(ax,x,[[43.3,79.1,73.1],[71.6,78.1,75.7]],['Spectrum saving','Power saving'],
     ['#D85A30','#1D9E75'],0.35)
ax.set_xticks(x); ax.set_xticklabels(ns); ax.set_ylabel('%'); ax.set_ylim(0,100)
ax.legend(frameon=False,fontsize=9)
ax.set_title('N-step sensitivity — n=3 joint optimum (+35.8 pp spectrum vs n=1)',fontsize=9.5)
plt.tight_layout(); fig.savefig(O+'fig13_nstep_v7.png',dpi=200); plt.close()

# Fig 16 — velocity-stratified
grp=['Slow\n5–25 m/s','Medium\n25–40 m/s']; x=np.arange(2)
fig,ax=plt.subplots(figsize=(6,4))
bars(ax,x,[[95.3,95.3],[78.1,76.8],[88.1,78.6]],['HO success','Power saving','Spectrum saving'],
     ['#378ADD','#1D9E75','#D85A30'],0.25)
ax.set_xticks(x); ax.set_xticklabels(grp); ax.set_ylabel('%'); ax.set_ylim(0,108)
ax.legend(frameon=False,ncol=3,fontsize=8,loc='upper center')
ax.set_title('Velocity-stratified — HO invariant; operating point shifts with speed',fontsize=9)
plt.tight_layout(); fig.savefig(O+'fig16_velocity_v7.png',dpi=200); plt.close()
print('generated fig8, fig12, fig13, fig16 (v7)')
