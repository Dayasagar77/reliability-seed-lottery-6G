#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""blockage_prep.py <rec> — compute REAL per-step 2D ray-blocking flags for the paper's ego cars
in one HighD recording, from actual truck positions (reviewer major comment 2).

Model (the reviewer's 'simple 2D ray-blocking'): the serving gNB is roadside, so the ego->gNB LOS is
~perpendicular to the road. A vehicle of class Truck blocks the link at a step iff its footprint spans
the ego's longitudinal position (x0 <= x_ego <= x1) AND its lateral extent lies between the ego and the
near-side roadside gNB (an intervening lane). Flags are emitted in the SAME per-(Car_ID, Recording) CSV
row order train_phase2.load_trajectories consumes, so they align 1:1 with sim steps.

Runs on the user's machine (pandas). Paths relative to the device_bash cwd (/sessions/<id>).
Usage: python3 blockage_prep.py 35   ->  mnt/startfromscratch/blockage_flags_35.csv
"""
import sys, pandas as pd, numpy as np

rec = str(sys.argv[1])
DATA = 'mnt/highD-dataset-v1.0/data'
EGO_CSV = 'mnt/startfromscratch/6G_Handover_Training_Data_v3.csv'
OUT = f'mnt/startfromscratch/blockage_flags_{rec}.csv'
D_PERP = 10.0

rm = pd.read_csv(f'{DATA}/{rec}_recordingMeta.csv')
upper = [float(v) for v in str(rm.iloc[0]['upperLaneMarkings']).split(';')]
lower = [float(v) for v in str(rm.iloc[0]['lowerLaneMarkings']).split(';')]
y_upper_edge, y_lower_edge = min(upper), max(lower)
y_mid = (min(upper) + max(lower)) / 2.0
gnb_upper, gnb_lower = y_upper_edge - D_PERP, y_lower_edge + D_PERP

meta = pd.read_csv(f'{DATA}/{rec}_tracksMeta.csv')
truck_ids = set(int(i) for i in meta.loc[meta['class'] == 'Truck', 'id'])

tr = pd.read_csv(f'{DATA}/{rec}_tracks.csv', usecols=['frame', 'id', 'x', 'y', 'width', 'height'])
trucks = tr[tr['id'].isin(truck_ids)].copy()
trucks['x0'] = trucks['x'] - trucks['width'] / 2.0
trucks['x1'] = trucks['x'] + trucks['width'] / 2.0
trucks['y0'] = trucks['y'] - trucks['height'] / 2.0
trucks['y1'] = trucks['y'] + trucks['height'] / 2.0
tbf = {int(f): g[['x0', 'x1', 'y0', 'y1', 'id']].values for f, g in trucks.groupby('frame')}

ego_csv = pd.read_csv(EGO_CSV)
ego_csv = ego_csv[ego_csv['Recording'].astype(str) == rec]
ego_ids = set(int(c) for c in ego_csv['Car_ID'].unique())
# precompute frame-sorted (frame,x,y) per ego id ONCE
ego_raw = {int(i): g.sort_values('frame')[['frame', 'x', 'y']].values
           for i, g in tr[tr['id'].isin(ego_ids)].groupby('id')}

out, mismatch = [], 0
for cid, grp in ego_csv.groupby('Car_ID'):
    cid = int(cid)
    xseq = grp['x_position'].values.astype(float)          # CSV row order == sim step order
    raw = ego_raw.get(cid)
    if raw is None or len(raw) != len(xseq) or (len(raw) and np.max(np.abs(raw[:, 1] - xseq)) > 1.0):
        mismatch += 1
        out.extend((rec, cid, i, 0) for i in range(len(xseq)))
        continue
    for i in range(len(xseq)):
        fr, xe, ye = int(raw[i, 0]), raw[i, 1], raw[i, 2]
        gnb_y = gnb_upper if ye < y_mid else gnb_lower
        lo, hi = (gnb_y, ye) if gnb_y < ye else (ye, gnb_y)
        blk = 0
        arr = tbf.get(fr)
        if arr is not None:
            for x0, x1, y0, y1, tid in arr:
                if int(tid) != cid and x0 <= xe <= x1 and y1 > lo and y0 < hi:
                    blk = 1
                    break
        out.append((rec, cid, i, blk))

o = pd.DataFrame(out, columns=['Recording', 'Car_ID', 'step', 'blocked'])
o.to_csv(OUT, index=False)
tot, blk = len(o), int(o['blocked'].sum())
neps = int(o.groupby('Car_ID')['blocked'].max().sum())
print(f"rec {rec}: egos={len(ego_ids)} steps={tot} blocked_steps={blk} "
      f"({100*blk/max(tot,1):.2f}%) egos_with_blockage={neps} mismatch_cars={mismatch} -> {OUT}")
