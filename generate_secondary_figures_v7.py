#!/usr/bin/env python3
"""Fig 7 (training convergence, from-scratch under corrected env), Fig 10 (Pareto, corrected
points), Fig 11 (action distribution, v6)."""
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
O='/tmp/dndqn-clean/results/'

# Fig 7 — from-scratch 'none' training curve (corrected env, laptop run, 800k)
steps=np.array([30,60,90,120,150,180,210,240,270,300,330,360,390,420,450,480,510,540,570,600,630,660,690,720,750,780])*1000
rew=np.array([-671.56,-627.38,-320.84,-261.26,-119.12,-13.40,26.84,203.69,213.29,256.91,296.20,283.02,245.43,210.22,276.49,177.67,198.47,186.92,203.17,237.25,245.85,268.12,240.93,247.14,337.27,273.37])
fig,ax=plt.subplots(figsize=(6.5,4))
ax.plot(steps/1000,rew,'-o',color='#378ADD',ms=3,lw=1.4)
ax.axhline(np.mean(rew[-8:]),ls='--',color='#888',lw=1,label=f'plateau ~{np.mean(rew[-8:]):.0f}')
ax.set_xlabel('Training steps (k)'); ax.set_ylabel('Mean episode reward (20-ep window)')
ax.set_title('Training convergence — Dn-DQN, from scratch under corrected env (800k)',fontsize=9)
ax.legend(frameon=False,fontsize=9); ax.spines[['top','right']].set_visible(False)
plt.tight_layout(); fig.savefig(O+'fig7_convergence_v7.png',dpi=200); plt.close()

# Fig 10 — Pareto: (spec, power) and (spec, HO). corrected points
names=['Dn-DQN','PG-RRM','3GPP A3','DDQN+PER','Std DQN']
spec=[79.5,0.0,0.0,39.9,48.3]; powr=[77.9,12.7,0.0,80.7,79.2]; ho=[95.3,98.2,98.8,96.0,95.8]
col=['#D85A30','#1D9E75','#8a8a8a','#7a6fb0','#b07a9e']
fig,(a1,a2)=plt.subplots(1,2,figsize=(10,4))
for i,n in enumerate(names):
    mk='*' if n=='Dn-DQN' else 'o'; sz=260 if n=='Dn-DQN' else 90
    a1.scatter(spec[i],powr[i],s=sz,marker=mk,color=col[i],edgecolor='#333',zorder=3,label=n)
    a2.scatter(spec[i],ho[i],s=sz,marker=mk,color=col[i],edgecolor='#333',zorder=3)
a1.set_xlabel('Spectrum saving (%)'); a1.set_ylabel('Power saving (%)'); a1.set_title('Spectrum vs Power',fontsize=10)
a2.set_xlabel('Spectrum saving (%)'); a2.set_ylabel('HO success (%)'); a2.set_title('Spectrum vs HO success',fontsize=10)
for a in (a1,a2): a.spines[['top','right']].set_visible(False)
a1.legend(frameon=False,fontsize=8,loc='center right')
fig.suptitle('Four-objective Pareto front (corrected env) — Dn-DQN at the high-spectrum/high-power vertex',fontsize=9.5,y=1.02)
plt.tight_layout(); fig.savefig(O+'fig10_pareto_v7.png',dpi=200,bbox_inches='tight'); plt.close()

# Fig 11 — action distribution (v6)
acts=['HANDOVER','RB_DOWN','PWR_DOWN','HOLD','PWR_UP','RB_UP']; sh=[83.9,11.1,4.9,0.0,0.0,0.0]
fig,ax=plt.subplots(figsize=(6.5,4))
ax.bar(range(6),sh,color=['#378ADD','#D85A30','#1D9E75','#aaa','#aaa','#aaa'])
for i,v in enumerate(sh): ax.text(i,v+1,f'{v:.1f}%',ha='center',fontsize=8)
ax.set_xticks(range(6)); ax.set_xticklabels(acts,fontsize=8,rotation=15); ax.set_ylabel('Action share (%)')
ax.set_title('Greedy action distribution (canonical model, v6; mean episode ~60 steps)',fontsize=9)
ax.spines[['top','right']].set_visible(False); ax.set_ylim(0,95)
plt.tight_layout(); fig.savefig(O+'fig11_actiondist_v7.png',dpi=200); plt.close()
print('generated fig7, fig10, fig11 (v7)')
