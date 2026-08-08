"""
nyu_channel_impact_v6.py — physics-level, agent-free comparison of the OLD free-space
channel vs the NEW measurement-calibrated NYU 142 GHz CI channel, over the REAL HighD
geometry. Also calibrates the array-gain so the NEW operating point is comparable to OLD
(realism comes from the CI/LOS-NLOS/shadow structure, not from a gratuitous power shift).

Deterministic (mean large-scale) RSRP is used for a fair, reproducible comparison:
  OLD: RSRP = Pt + 10log10(FSPL_gain(d)) - molecular          (n=2, no array gain, no LOS/NLOS)
  NEW: RSRP = Pt + G_array - CI(d, LOS/NLOS) - molecular       (measured PLE; shadow shown separately)
"""
import numpy as np
import multi_seed_eval as mse
from env_6g_v5 import (
    GNB_X, D_PERP, CELL_RADIUS, A3_THRESH_DBM, RLF_THRESH_DBM,
    MOLECULAR_ABS_DB_PER_KM, PT_MAX_DBM,
)
from channel_model import channel_gain as fs_gain
import channel_model_nyu as nyu

PT = PT_MAX_DBM   # 30 dBm nominal for the physics comparison

# ── Real geometry: distances to nearest gNB over the whole trajectory pool ──
slow, med = mse.load_trajectories(mse.CSV_PATH)
pool = slow + med
ds, los = [], []
for traj in pool:
    x   = traj[:, 0]
    idx = np.argmin(np.abs(x[:, None] - GNB_X[None, :]), axis=1)
    dal = np.abs(x - GNB_X[idx])
    d   = np.sqrt(dal**2 + D_PERP**2)
    ds.append(d)
    los.append(d < CELL_RADIUS * 0.75)
d      = np.concatenate(ds)
is_los = np.concatenate(los)
mol    = MOLECULAR_ABS_DB_PER_KM * (d / 1000.0)
n_los  = int(is_los.sum())
print(f"Real geometry: {d.size} samples over {len(pool)} trajectories | "
      f"LOS {100*n_los/d.size:.1f}% / NLOS {100*(1-n_los/d.size):.1f}% | "
      f"distance {d.min():.1f}-{d.max():.1f} m (median {np.median(d):.1f})\n")

# ── OLD free-space RSRP ─────────────────────────────────────────────────────
rsrp_old = PT + 10*np.log10(fs_gain(d)) - mol

# ── NEW CI RSRP (mean large-scale), parameterised by array gain G ───────────
pl_los  = nyu.ci_pathloss_db(d, True)
pl_nlos = nyu.ci_pathloss_db(d, False)
pl_ci   = np.where(is_los, pl_los, pl_nlos)
def rsrp_new(G):
    return PT + G - pl_ci - mol

# ── Calibrate G so NEW median matches OLD median ────────────────────────────
G_match = float(np.median(rsrp_old) - np.median(rsrp_new(0.0)))
print(f"Array gain that matches OLD median link budget: G = {G_match:.1f} dBi "
      f"(rounded {round(G_match):d} dBi)\n")

def stats(name, r):
    print(f"{name:<28} median={np.median(r):7.1f}  mean={np.mean(r):7.1f}  "
          f"p10={np.percentile(r,10):7.1f}  p90={np.percentile(r,90):7.1f}  "
          f"%<A3={100*np.mean(r<A3_THRESH_DBM):5.1f}  %<RLF={100*np.mean(r<RLF_THRESH_DBM):5.1f}")

print("Deterministic RSRP (dBm) over the real geometry — coverage vs thresholds:")
print(f"  (A3 = {A3_THRESH_DBM:.0f} dBm, RLF = {RLF_THRESH_DBM:.0f} dBm)\n")
stats("OLD free-space (n=2)", rsrp_old)
for G in [0, 5, 10, 15, 20, round(G_match)]:
    stats(f"NEW CI, array gain {G:>2} dBi", rsrp_new(G))

# ── LOS vs NLOS contrast the OLD model could not represent ──────────────────
Gc = round(G_match)
print(f"\nLOS vs NLOS split under NEW CI (array gain {Gc} dBi) — the physics OLD missed:")
rn = rsrp_new(Gc)
for lab, m in [("LOS  links", is_los), ("NLOS links", ~is_los)]:
    if m.sum():
        print(f"  {lab}: median RSRP {np.median(rn[m]):6.1f} dBm | "
              f"%<A3 {100*np.mean(rn[m]<A3_THRESH_DBM):5.1f} | %<RLF {100*np.mean(rn[m]<RLF_THRESH_DBM):5.1f}")

# ── Shadow-fading spread now present (absent in OLD) ────────────────────────
print(f"\nShadow fading now modelled (per-association, log-normal): "
      f"LOS sigma={nyu.SIG_LOS} dB, NLOS sigma={nyu.SIG_NLOS} dB "
      f"(+/-1 sigma swing the OLD model did not have).")
print(f"  => an NLOS link at cell edge can vary by ~{2*nyu.SIG_NLOS:.0f} dB peak-to-peak from shadowing alone.")
