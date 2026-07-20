#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""rederive_blockage.py <rec> — re-derive a 10 Hz eval trajectory pool WITH real truck-blockage
flags directly from raw HighD (reviewer major comment 2). The original preprocessing script for
6G_Handover_Training_Data_v3.csv was not preserved, so this rebuilds an equivalent pool from the
same recordings and the SAME empirical rate (~0.1 s/step), with per-step 2D ray-blocking computed
from actual Truck-class footprints. Equivalence is validated downstream: the frozen guarded models
must reproduce their published no-blockage operating point on this pool before any blockage claim.

Per ego (Car-class) trajectory:
  - direction-normalized (x increasing), clipped to the 374 m road span (env geometry),
  - linearly interpolated from 25 Hz to a 0.1 s grid (matches the empirical ego-CSV rate),
  - blocked(t)=1 iff at the nearest raw frame a Truck footprint spans the ego's x AND lies
    laterally between the ego and the near-side roadside gNB line (D_perp=10 m off the road edge).

Runs on the user's machine. Usage: python3 rederive_blockage.py 35
Output: mnt/startfromscratch/rederived_blockage_<rec>.csv  (Car_ID, step, x, v_abs, blocked)
"""
import sys
import numpy as np
import pandas as pd

rec = str(sys.argv[1])
DATA = 'mnt/highD-dataset-v1.0/data'
OUT = f'mnt/startfromscratch/rederived_blockage_{rec}.csv'
D_PERP = 10.0
DT = 0.1                     # empirical ego-CSV rate (~102 ms); raw highD = 25 Hz (40 ms)
ROAD_LEN = 374.0             # 5 x 74.8 m (env geometry)
X_MARGIN = 5.0

rm = pd.read_csv(f'{DATA}/{rec}_recordingMeta.csv')
upper = [float(v) for v in str(rm.iloc[0]['upperLaneMarkings']).split(';')]
lower = [float(v) for v in str(rm.iloc[0]['lowerLaneMarkings']).split(';')]
y_mid = (min(upper) + max(lower)) / 2.0
gnb_y_upper = min(upper) - D_PERP
gnb_y_lower = max(lower) + D_PERP

meta = pd.read_csv(f'{DATA}/{rec}_tracksMeta.csv')
truck_ids = set(int(i) for i in meta.loc[meta['class'] == 'Truck', 'id'])
car_meta = meta[meta['class'] == 'Car'][['id', 'drivingDirection']]

tr = pd.read_csv(f'{DATA}/{rec}_tracks.csv', usecols=['frame', 'id', 'x', 'y', 'width', 'height', 'xVelocity'])
x_all_max = float(tr['x'].max())

trucks = tr[tr['id'].isin(truck_ids)]
tbf = {}
for f, g in trucks.groupby('frame'):
    a = np.empty((len(g), 5))
    a[:, 0] = g['x'].values - g['width'].values / 2.0
    a[:, 1] = g['x'].values + g['width'].values / 2.0
    a[:, 2] = g['y'].values - g['height'].values / 2.0
    a[:, 3] = g['y'].values + g['height'].values / 2.0
    a[:, 4] = g['id'].values
    tbf[int(f)] = a

egos = tr[tr['id'].isin(set(int(i) for i in car_meta['id']))]
dirmap = dict(zip(car_meta['id'].astype(int), car_meta['drivingDirection'].astype(int)))

rows = []
n_traj = n_blk_traj = n_blk_steps = n_steps = 0
for cid, g in egos.groupby('id'):
    cid = int(cid)
    g = g.sort_values('frame')
    fr = g['frame'].values.astype(int)
    x = g['x'].values.astype(float)
    y = g['y'].values.astype(float)
    v = np.abs(g['xVelocity'].values.astype(float))
    t = (fr - fr[0]) / 25.0                       # seconds from first frame
    if t[-1] < 1.0:
        continue
    # direction-normalize x (all trajectories drive toward increasing x)
    if dirmap.get(cid, 2) == 1:
        xn = x_all_max - x
    else:
        xn = x.copy()
    xn = xn - xn.min() + X_MARGIN                 # start near road entry
    # 0.1 s grid, clipped to the road span
    tg = np.arange(0.0, t[-1] + 1e-9, DT)
    xg = np.interp(tg, t, xn)
    vg = np.interp(tg, t, v)
    keep = xg <= (ROAD_LEN - X_MARGIN)
    tg, xg, vg = tg[keep], xg[keep], vg[keep]
    if len(tg) < 35:                              # HO_MIN_LEAD(25)+10, same length filter
        continue
    # blockage on the grid: nearest raw frame's truck geometry
    idx = np.clip(np.round(tg * 25.0).astype(int), 0, len(fr) - 1)
    blk = np.zeros(len(tg), dtype=int)
    for k in range(len(tg)):
        f0 = int(fr[idx[k]])
        arr = tbf.get(f0)
        if arr is None:
            continue
        xe, ye = x[idx[k]], y[idx[k]]             # RAW coordinates for the geometric test
        gy = gnb_y_upper if ye < y_mid else gnb_y_lower
        lo, hi = (gy, ye) if gy < ye else (ye, gy)
        hit = (arr[:, 0] <= xe) & (xe <= arr[:, 1]) & (arr[:, 3] > lo) & (arr[:, 2] < hi) & (arr[:, 4] != cid)
        if hit.any():
            blk[k] = 1
    n_traj += 1
    n_steps += len(tg)
    bs = int(blk.sum())
    n_blk_steps += bs
    if bs:
        n_blk_traj += 1
    rows.extend((cid, k, round(float(xg[k]), 3), round(float(vg[k]), 3), int(blk[k])) for k in range(len(tg)))

o = pd.DataFrame(rows, columns=['Car_ID', 'step', 'x', 'v_abs', 'blocked'])
o.insert(0, 'Recording', rec)
o.to_csv(OUT, index=False)
print(f"rec {rec}: trajs={n_traj} steps={n_steps} blocked_steps={n_blk_steps} "
      f"({100*n_blk_steps/max(n_steps,1):.2f}%) trajs_with_blockage={n_blk_traj} "
      f"({100*n_blk_traj/max(n_traj,1):.1f}%) -> {OUT}")
