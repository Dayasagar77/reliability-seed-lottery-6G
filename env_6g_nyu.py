"""
env_6g_nyu.py — SINGLE-VARIABLE channel upgrade of the 6G Sub-THz env.

This subclasses SubTHzEnv6G_v6 and overrides ONLY the propagation computations
(_compute_rsrp, _compute_sinr) to use the measurement-calibrated NYU 142 GHz CI model
(channel_model_nyu) + log-normal shadow fading + a directional array (beamforming) gain.

EVERYTHING ELSE IS IDENTICAL to v6: geometry (5 gNBs, ISD 74.8 m, D_PERP 10 m), the 6D
state, the action set, the reward, the sustained (N310) RLF model, episode logic. So a
comparison of results between v6 and this env isolates exactly one variable: the channel.

Modelling choices (all explicit):
  * Large-scale path loss  : NYU CI, measured 142 GHz UMi PLE (LOS 1.9 / NLOS 2.9).
  * Shadow fading          : log-normal, measured sigma (LOS 2.7 / NLOS 8.2 dB), drawn
                             ONCE per (serving-gNB, LOS-state) association and held constant
                             (large-scale shadow varies slowly — NOT redrawn every 40 ms
                             sample; that would be a fast-fading artifact).
  * Small-scale fading     : retained alpha-mu (channel_model.apply_fading), unchanged.
  * Array / beamforming    : serving beam gets ARRAY_GAIN_DBI (boresight, stated assumption);
                             interferers get INTERFERER_ARRAY_GAIN_DBI (sidelobe, misaligned).
  * LOS/NLOS rule          : identical geometric rule to v6 (d < 0.75*CELL_RADIUS => LOS).
                             Real blockage-based LOS/NLOS is Paper-2 work, not changed here.
"""
import numpy as np

from env_6g_v6 import SubTHzEnv6G_v6
from env_6g_v5 import (
    CELL_RADIUS, ISD, MOLECULAR_ABS_DB_PER_KM,
    N_INTERFERERS, INTERFERER_PT_DBM, N0_WATTS,
)
from channel_model import apply_fading, get_interferer_distances
import channel_model_nyu as nyu


class SubTHzEnv6G_NYU(SubTHzEnv6G_v6):
    """v6 environment with the NYU 142 GHz CI + shadow + array-gain channel."""

    def __init__(self, vehicle_trajectory,
                 array_gain_dbi=nyu.ARRAY_GAIN_DBI,
                 interferer_gain_dbi=nyu.INTERFERER_ARRAY_GAIN_DBI,
                 small_scale=True, **kw):
        self.array_gain_dbi     = float(array_gain_dbi)
        self.interferer_gain_dbi = float(interferer_gain_dbi)
        self.small_scale        = bool(small_scale)
        super().__init__(vehicle_trajectory, **kw)

    def _reset_state(self):
        super()._reset_state()
        self._shadow_key = None
        self._shadow_db  = 0.0

    def _shadow(self, is_los, active):
        """Log-normal shadow fading (dB), held constant per (gNB, LOS-state) association."""
        if not active:
            return 0.0
        key = (self.serving_gnb_idx, bool(is_los))
        if self._shadow_key != key:
            self._shadow_key = key
            self._shadow_db  = float(self.rng.normal(0.0, nyu.sigma_sf(is_los)))
        return self._shadow_db

    def _compute_rsrp(self, d_m, apply_fading_flag=True):
        """RSRP under NYU CI + shadow + array gain (+ retained alpha-mu small-scale)."""
        is_los = (d_m < CELL_RADIUS * 0.75)
        pl_db  = nyu.ci_pathloss_db(d_m, is_los)                 # measured CI mean path loss
        sh_db  = self._shadow(is_los, apply_fading_flag)         # per-association shadow
        rsrp_dbm = self.pt_dbm + self.array_gain_dbi - pl_db - sh_db

        if apply_fading_flag and self.small_scale:
            h2 = apply_fading(1.0, is_los=is_los, rng=self.rng)  # alpha-mu power multiplier h^2
            rsrp_dbm += 10.0 * np.log10(max(h2, 1e-20))

        rsrp_dbm -= MOLECULAR_ABS_DB_PER_KM * (d_m / 1000.0)     # ITU-R P.676-13, unchanged
        return rsrp_dbm, is_los

    def _compute_sinr(self, d_m, is_los, velocity_ms=0.0):
        """SINR with CI path loss + beamforming: serving beam aligned, interferers sidelobe."""
        # Serving link (aligned beam)
        pl_s   = nyu.ci_pathloss_db(d_m, is_los)
        g_s    = 10 ** (self.array_gain_dbi / 10.0)
        L_s    = 10 ** (-pl_s / 10.0)
        p_s    = 10 ** ((self.pt_dbm - 30) / 10.0)
        sig    = p_s * g_s * L_s

        # Co-channel interferers (assumed NLOS + beam-misaligned -> sidelobe gain)
        i_dists = get_interferer_distances(self.ue_x, self.ue_y, CELL_RADIUS, 0.0,
                                           isd_m=ISD, n_interferers=N_INTERFERERS)
        g_i    = 10 ** (self.interferer_gain_dbi / 10.0)
        p_i    = 10 ** ((INTERFERER_PT_DBM - 30) / 10.0)
        interference = 0.0
        for di in np.atleast_1d(i_dists):
            interference += p_i * g_i * 10 ** (-nyu.ci_pathloss_db(di, is_los=False) / 10.0)

        # Base SINR (CI path loss + beamforming), THEN the SAME bounded Doppler-ICI adjustment
        # the parent env uses — so the only difference from v6 stays the channel. (The earlier
        # additive form was mis-scaled: the raw Doppler term ~1e-3 W dwarfs the ~1e-9 W received
        # signal and drove SINR to ~-49 dB, zeroing throughput. This bounded form is ~a few dB.)
        sinr_lin = sig / (interference + N0_WATTS)
        sinr_db  = 10.0 * np.log10(max(sinr_lin, 1e-20))
        doppler  = self._compute_doppler_interference(velocity_ms)
        if doppler > 0:
            slv = 10 ** (sinr_db / 10.0)
            signal_power = slv * (N0_WATTS + doppler)
            new_slin = signal_power / (N0_WATTS + doppler + slv * doppler / max(slv, 1e-10))
            sinr_db = 10.0 * np.log10(max(new_slin, 1e-10))
        return float(sinr_db)
