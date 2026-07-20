#!/usr/bin/env python3
"""env_6g_branched.py — branched-action environment for the cliff-aware JOINT controller
(Paper 1 novel mechanism). Each step the agent emits BOTH branches simultaneously:
    a_rel  in {0:STAY_PROTECT, 1:STAY_SAVE, 2:HANDOVER}
    a_spec in {0:RB_HOLD,      1:RB_DOWN,   2:RB_UP}

binary_power=True  -> PROTECT sets power to PT_MAX, SAVE sets power to PT_MIN  (cliff-aware, PROPOSED)
binary_power=False -> PROTECT steps power up, SAVE steps power down            (vanilla-BDQ ablation)

Implementation reuses the base SubTHzEnv6G_NYU transition verbatim: we set pt_dbm / rb_count and pick
the base flat action (HANDOVER when a_rel==HANDOVER, else HOLD), then delegate to super().step().
This guarantees the NYU channel, SINR, sustained-RLF, HO-outcome and info logic are IDENTICAL to the
flat controller — so any performance difference is due to the action-space geometry alone, not a
re-implemented environment. Reliability-tilted reward (0.8/0.1/0.1) + RLF terminal penalty (-300),
matching train_phase2_nyu_rel.
"""
from env_6g_nyu import SubTHzEnv6G_NYU
from env_6g_v6 import N310_DEFAULT
from env_6g_v5 import PT_MIN_DBM, PT_MAX_DBM, PT_STEP_DBM, RB_MIN, RB_MAX, RB_STEP, HANDOVER, HOLD
from dndqn_branched import R_STAY_PROTECT, R_STAY_SAVE, R_HANDOVER, S_HOLD, S_DOWN, S_UP

RLF_TERMINAL_PENALTY = -300.0
# C1/C5 shaping constants — EXACT mirror of train_phase2_nyu_rel (W 0.8/0.1/0.1 reliability reward)
W_MOB_REL, W_PWR_REL, W_SPEC_REL = 0.80, 0.10, 0.10
SAFE_RSRP_THRESH, PWR_DOWN_UNSAFE_PEN = -78.0, -0.20
_PWR_STEP_REW = W_PWR_REL * (PT_STEP_DBM / (PT_MAX_DBM - PT_MIN_DBM)) * 10.0
_RB_STEP_REW = W_SPEC_REL * (PT_STEP_DBM / (PT_MAX_DBM - PT_MIN_DBM)) * 10.0


class BranchedEnv6G(SubTHzEnv6G_NYU):
    """Branched env with the FULL reliability reward ported (v2 — reward-matched to the flat
    Phase2Env_NYU_rel: C1 RSRP-gated SAVE shaping, C5 symmetric RB_DOWN reward, RLF −300).
    v1 lacked C1/C5 (they were keyed to flat action IDs); its agents landed on the no-C1
    operating point (RLF ~21%, Pwr 77-85%) — kept as 'noshape' evidence rows."""
    def __init__(self, traj, binary_power=True, array_gain_dbi=15.0,
                 w_mob=W_MOB_REL, w_pwr=W_PWR_REL, w_spec=W_SPEC_REL, guard_db=None, guard_mode='action'):
        super().__init__(traj, array_gain_dbi=array_gain_dbi, n310=N310_DEFAULT,
                         w_mob=w_mob, w_pwr=w_pwr, w_spec=w_spec)
        self.binary_power = bool(binary_power)
        # guard_db: MEASUREMENT-DERIVED protective guard (oracle sweep). None = unguarded.
        # 9 dB ~= worst-case (NLOS) shadow std 8.2 dB, the conservative end of the measured sweep.
        # guard_mode:
        #   'action' (v3, evidence): only gates SAVE *presses* — EXPLOITABLE: agent presses SAVE once
        #            then idles at PT_MIN via no-op HANDOVER/other actions, bypassing the gate
        #            (observed: guarded-action s0 -> RLF 22.3%, overrides on only 3% of presses).
        #   'state'  (v3.1, PROPOSED): safety layer on the POWER STATE — every step, if last observed
        #            RSRP < RLF_THRESH + Δpower + guard and power is low, power is restored to PT_MAX
        #            regardless of the action pressed (intent penalty unless PROTECT). Exact
        #            controller realization of the oracle-budget definition.
        self.guard_db = guard_db
        self.guard_mode = guard_mode
        self._prev_rsrp = -70.0

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._prev_rsrp = -70.0
        return obs, info

    def step(self, action):
        a_rel, a_spec = int(action[0]), int(action[1])
        rsrp_before = self._prev_rsrp

        # ── measured save-guard: below the headroom line, SAVE physically acts as PROTECT ──
        unsafe_save_intent = False
        if a_rel == R_STAY_SAVE and self.guard_db is not None and self.guard_mode == 'action':
            save_line = -92.0 + (PT_MAX_DBM - PT_MIN_DBM) + self.guard_db   # RLF thresh + Δpower + guard
            if rsrp_before < save_line:
                unsafe_save_intent = True
                a_rel = R_STAY_PROTECT          # physical override (protect the link)

        # ── reliability branch: power (binary or incremental) ──────────────
        if a_rel == R_STAY_PROTECT:
            self.pt_dbm = PT_MAX_DBM if self.binary_power else min(self.pt_dbm + PT_STEP_DBM, PT_MAX_DBM)
        elif a_rel == R_STAY_SAVE:
            self.pt_dbm = PT_MIN_DBM if self.binary_power else max(self.pt_dbm - PT_STEP_DBM, PT_MIN_DBM)
        # (R_HANDOVER handled below via the base flat action)

        # ── spectrum branch: RB level ──────────────────────────────────────
        if a_spec == S_DOWN:
            self.rb_count = max(self.rb_count - RB_STEP, RB_MIN)
        elif a_spec == S_UP:
            self.rb_count = min(self.rb_count + RB_STEP, RB_MAX)

        # ── STATE-mode safety layer: low margin + low power -> restore PT_MAX, whatever was pressed ──
        if self.guard_db is not None and self.guard_mode == 'state':
            save_line = -92.0 + (PT_MAX_DBM - PT_MIN_DBM) + self.guard_db
            if rsrp_before < save_line and self.pt_dbm < PT_MAX_DBM:
                self.pt_dbm = PT_MAX_DBM
                if a_rel != R_STAY_PROTECT:
                    unsafe_save_intent = True   # intent penalty: agent failed to protect on its own

        # ── delegate transition to the base env (identical physics/info) ───
        flat_action = HANDOVER if a_rel == R_HANDOVER else HOLD
        obs, _, terminated, truncated, info = super().step(flat_action)
        self._prev_rsrp = info['rsrp_dbm']

        # ── rebuild reward EXACTLY like Phase2Env_NYU_rel, mapped to branches ──
        r_mob = info.get('r_mobility', 0.0)
        r_pwr = info.get('r_power', 0.0)
        r_spec = info.get('r_spectrum', 0.0)
        reward = self.w_mob * r_mob
        if unsafe_save_intent:                         # guarded: unsafe SAVE intent → penalty, link protected
            reward += PWR_DOWN_UNSAFE_PEN
        elif a_rel == R_STAY_SAVE:                     # C1 gate on the power-saving action
            reward += _PWR_STEP_REW if rsrp_before >= SAFE_RSRP_THRESH else PWR_DOWN_UNSAFE_PEN
        else:
            reward += self.w_pwr * r_pwr
        if a_spec == S_DOWN:                           # C5 symmetric RB-down reward
            reward += _RB_STEP_REW
        else:
            reward += self.w_spec * r_spec
        if terminated and info.get('sustained_rlf'):   # C3 terminal penalty
            reward += RLF_TERMINAL_PENALTY
        return obs, reward, terminated, truncated, info


# Action-space sizes for the branched net (reliability, spectrum)
BRANCH_SIZES = [3, 3]

if __name__ == "__main__":
    import numpy as np, train_phase2 as tp
    slow, med, fast, skip = tp.load_trajectories(tp.CSV_PATH)
    traj = (slow + med)[0]
    for bp in (True, False):
        env = BranchedEnv6G(traj, binary_power=bp)
        obs, _ = env.reset(seed=0)
        assert obs.shape == (6,), obs.shape
        # drive PROTECT then SAVE, check binary power semantics
        env.pt_dbm = 27.0
        env.step((R_STAY_PROTECT, S_HOLD))
        prot = env.pt_dbm
        env.step((R_STAY_SAVE, S_DOWN))
        save = env.pt_dbm
        r_ok = True; steps = 0
        env2 = BranchedEnv6G(traj, binary_power=bp); env2.reset(seed=1); done = False
        while not done and steps < 200:
            o, r, t, tr, info = env2.step((np.random.randint(3), np.random.randint(3)))
            done = t or tr; steps += 1
            if not np.isfinite(r): r_ok = False; break
        print(f"binary_power={bp}: PROTECT->pt={prot:.1f}  SAVE->pt={save:.1f}  "
              f"| ran {steps} steps, reward finite={r_ok}, keys={'pt_dbm' in info and 'rb_count' in info}")
    print("env_6g_branched self-test OK")
