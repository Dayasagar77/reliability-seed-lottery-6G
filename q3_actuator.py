#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
q3_actuator.py - the ONLY mechanism by which the 2x2 assignment ablation changes the action
                 space. No file on disk is edited by this campaign.

WHY THIS FILE EXISTS
--------------------
The pre-registration of 08 Aug 2026 puts one claim under test:

    the MEASUREMENT decides the ASSIGNMENT. Binarising the axis whose measured KPI coupling is
    cliff-like, while leaving fine control on the axis measured as decoupled, is what collapses
    the between-training variance - not binarisation per se, and not the branched architecture.

Every arm in the manuscript that binarises power also leaves spectrum incremental, so the
assignment and the binarisation are confounded. Breaking that confound needs a binary SPECTRUM
actuator, which the published environment does not have.

HOW IT IS OBTAINED WITHOUT TOUCHING PUBLISHED SOURCE
----------------------------------------------------
env_6g_branched.BranchedEnv6G.step() contains, verbatim:

    if a_spec == S_DOWN:
        self.rb_count = max(self.rb_count - RB_STEP, RB_MIN)
    elif a_spec == S_UP:
        self.rb_count = min(self.rb_count + RB_STEP, RB_MAX)

RB_STEP is a module global of env_6g_branched, resolved at CALL time. Rebinding it to
RB_MAX - RB_MIN = 56 makes S_DOWN land on RB_MIN and S_UP land on RB_MAX from any state - which
IS the binary spectrum actuator {10, 66} - with the published step() executing unmodified.

Two facts make this safe, and q3_smoke.py asserts both:
  1. RB_STEP occurs in env_6g_branched.py at exactly two places, lines 80 and 82, both inside the
     spectrum branch. Nothing else in that module reads it.
  2. The C5 reward constant is
         _RB_STEP_REW = W_SPEC_REL * (PT_STEP_DBM / (PT_MAX_DBM - PT_MIN_DBM)) * 10.0
     which is computed from the POWER step and does not involve RB_STEP. The reward is therefore
     byte-identical across all four cells of the 2x2, as the pre-registration requires.

A consequence worth stating rather than hiding: with a binary spectrum actuator one S_DOWN press
returns 56 resource blocks and still earns the same _RB_STEP_REW as an 8-block press does under
the incremental actuator. That is what changing granularity while holding the reward fixed MEANS,
and holding the reward fixed is the correct control for this ablation.

THE 2x2
-------
    arm    power        spectrum     binary_power   RB_STEP   status
    g_ii   incremental  incremental  False          8         to train
    g_bi   binary       incremental  True           8         PUBLISHED - never retrained here
    g_bb   binary       binary       True           56        to train
    g_ib   incremental  binary       False          56        to train   <- the falsifier

The guard is held constant at G = 9 dB, save line -76 dBm, guard_mode 'state', in all four.
"""
import env_6g_branched as E
from env_6g_v5 import RB_MIN, RB_MAX, RB_STEP as RB_STEP_PUBLISHED

RB_STEP_BINARY = RB_MAX - RB_MIN          # 56, so one press spans the whole range
GUARD_DB = 9.0
GUARD_MODE = 'state'

# key -> (label, binary_power, binary_spectrum, trained_here)
ARMS = {
    'g_ii': ('Guarded, incremental power + incremental spectrum', False, False, True),
    'g_bi': ('Guarded, binary power + incremental spectrum (published)', True, False, False),
    'g_bb': ('Guarded, binary power + binary spectrum', True, True, True),
    'g_ib': ('Guarded, incremental power + binary spectrum (inverted)', False, True, True),
}
ARM_ORDER = ['g_ii', 'g_bi', 'g_bb', 'g_ib']
TRAIN_ARMS = [a for a in ARM_ORDER if ARMS[a][3]]


def set_spectrum(binary):
    """Rebind the spectrum step. Returns the value now in force."""
    E.RB_STEP = RB_STEP_BINARY if binary else RB_STEP_PUBLISHED
    return E.RB_STEP


def assert_spectrum(binary):
    want = RB_STEP_BINARY if binary else RB_STEP_PUBLISHED
    if E.RB_STEP != want:
        raise RuntimeError(
            'spectrum actuator mismatch: env_6g_branched.RB_STEP is %r but this arm needs %r. '
            'Training and evaluation would disagree about the action space.' % (E.RB_STEP, want))
    return True


def configure(arm):
    """Set the actuator for `arm` and return (label, binary_power, binary_spectrum)."""
    if arm not in ARMS:
        raise KeyError('unknown arm %r; expected one of %s' % (arm, ARM_ORDER))
    label, bp, bs, _ = ARMS[arm]
    set_spectrum(bs)
    assert_spectrum(bs)
    return label, bp, bs


def describe(arm):
    label, bp, bs, trained = ARMS[arm]
    return ('%s | binary_power=%s binary_spectrum=%s RB_STEP=%d guard=%.1f dB mode=%s'
            % (label, bp, bs, RB_STEP_BINARY if bs else RB_STEP_PUBLISHED, GUARD_DB, GUARD_MODE))


if __name__ == '__main__':
    print('published RB_STEP = %d | binary RB_STEP = %d (RB_MIN=%d RB_MAX=%d)'
          % (RB_STEP_PUBLISHED, RB_STEP_BINARY, RB_MIN, RB_MAX))
    for a in ARM_ORDER:
        print('  %-5s %s%s' % (a, describe(a), '' if ARMS[a][3] else '   [not retrained]'))
