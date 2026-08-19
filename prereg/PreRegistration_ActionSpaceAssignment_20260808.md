# Pre-registration — measurement-derived action-space assignment, and a Rainbow baseline

**Paper 1 · Scientific Reports revision · registered 08 August 2026, before any training run.**

**Registered by:** Daya Sagar G · **Corresponding author:** Dr. Deepa Nivethika S.

> This document is written **before** any of the arms below are trained. Its purpose is to fix the
> hypotheses, the decision rules and the kill criteria in advance, so that the outcome cannot be
> chosen after the data is seen. It follows the same discipline as
> `DnDQN_SRDnDQN_PreRegistration_20260728.md`, whose registered hypothesis was subsequently
> **falsified and reported as a null**.

---

## 1. Background and the claim under test

The manuscript reports that sustained radio-link failure (RLF) is a steep, saturating function of
transmit power over 23–30 dBm (19.73% → 3.73%) and **exactly** independent of resource-block
allocation across 10–66 RBs (9.47 ± 1.86% at every setting). The proposed controller binarises the
power actuator and retains incremental control of spectrum, and attains sustained RLF 4.91% with a
between-training variance component σ_train < 0.17, against 10.11 ± 3.89% for the flat controller.

**The claim now under test is causal and stronger than anything currently in the paper:**

> The *measurement decides the assignment*. Binarising the axis whose measured KPI coupling is
> cliff-like, while leaving fine control on the axis measured as decoupled, is what collapses the
> between-training variance — not binarisation per se, and not the branched architecture.

This claim is currently **asserted by construction and never tested.** Every existing arm that
binarises power also leaves spectrum incremental, so the assignment and the binarisation are
confounded.

## 2. Design — a 2 × 2 on actuator granularity, guard held constant

The guard (G = 9 dB, save line −76 dBm) is **held constant in all four cells**, so the only thing
that varies is which axis receives which actuator type.

| Arm | Power actuator | Spectrum actuator | Status |
|---|---|---|---|
| **G-II** | incremental (3 dBm steps) | incremental (±8 RB) | **to run** unless an existing guarded checkpoint set matches exactly |
| **G-BI** (proposed, published) | **binary** {23, 30} dBm | incremental (±8 RB) | **have** — 5 checkpoints, 4.91%, σ_train < 0.17 |
| **G-BB** (binary everywhere) | **binary** {23, 30} dBm | **binary** {10, 66} RB | **to run** |
| **G-IB** (**inverted**) | incremental (3 dBm steps) | **binary** {10, 66} RB | **to run** |

Plus one comparator arm outside the 2 × 2:

| Arm | Description | Status |
|---|---|---|
| **RBW** | **Rainbow** — the existing dueling + double + PER + n-step trunk, plus noisy linear layers and a distributional C51 head. Flat 6-action space, identical reliability reward and budget | **to run** |

**Total new trainings: 4 arms × 5 seeds = 20**, estimated ~4 h wall-clock on the existing parallel
runner.

## 3. Protocol — identical to the published arms, no exceptions

- **5 independent trainings per arm**, seeds 0–4, random initialisation.
- **8 × 10⁵ training steps**, the same budget as every learned row in the manuscript.
- Identical reliability-tilted reward W = (0.80, 0.10, 0.10), sustained-RLF terminal penalty −300.
- Identical channel (NYU 142 GHz CI), identical HighD slow+medium trajectory pool.
- **Evaluation: R = 8 common-random-number blocks**, 5 evaluation seeds × 150 episodes, **the same
  channel-seed blocks as the published arms**, so all arms meet identical channel realisations.
- σ_train is the between-training variance component of the two-way crossed random-effects
  decomposition over seeds × CRN blocks, √((MS_A − MS_E)/R), as used throughout the manuscript.
- Where a variance component is not separable from zero, a **bound** is reported, never a point
  estimate — as already done for the proposed arm.

**No arm may be re-run, re-seeded or extended after its results are seen.** The 20 trainings are run
once.

## 4. Hypotheses

Stated so that each can fail.

**H1 — assignment beats the alternatives.**
Among the four guarded cells, **G-BI** (measurement-derived) attains the lowest σ_train of
sustained RLF.

**H2 — the inverted assignment is worse. *This is the falsifier.***
σ_train(RLF) for **G-IB** is **strictly greater** than for **G-BI**, with the lower bound of G-IB's
variance component exceeding the upper bound of G-BI's at ≥ 95% joint coverage.

*If H2 fails, the central claim of this campaign is false.*

**H3 — binarisation per se is not the mechanism.**
**G-BB** does not attain lower σ_train(RLF) than **G-BI**. Binarising the decoupled axis buys no
reliability-variance improvement, because that axis does not reach the failure mechanism.

**H4 — Rainbow does not remove the lottery.**
**RBW** does not attain a lower σ_train(RLF) than **G-BI**.

*Note: H4 concerns **variance**, not the mean.* Rainbow attaining a **lower mean RLF** than the
proposed controller would **not** refute the paper's thesis, and will be reported plainly if it
occurs. A Rainbow arm with a better mean but a larger σ_train would in fact *support* the thesis.

## 5. Primary and secondary outcomes

- **Primary:** σ_train of sustained RLF (%), per arm.
- **Secondary, reported for every arm regardless of outcome:** mean sustained RLF, handover success,
  power saving, spectrum saving, per-UE throughput — each with its own σ_train.
- Significance testing on means uses Welch's t with CRN correction and **Holm adjustment within
  family**, as in the manuscript. Minimum detectable effects at k = 5 are recomputed and reported
  **before** any comparison is interpreted, so that a null reads as a bound.

## 6. Rainbow correctness gate — passed before RBW results may be used

A Rainbow implementation that underperforms because of a bug would hand the paper a win it did not
earn. **RBW results are inadmissible until all four checks pass, and the check log is archived.**

1. **Component check.** Noisy layers produce non-zero, resampled noise per forward pass; C51
   projection preserves probability mass (support sums to 1 ± 1e-6 on random inputs).
2. **Reduction check — the strongest one.** With the distributional head and noisy layers disabled,
   the implementation must **reproduce the existing Dn-DQN results** on a matched seed, since it
   then reduces to already-verified code.
3. **Reference-behaviour check.** The full Rainbow reaches known published performance on a standard
   control benchmark (CartPole-v1 to solved threshold) within the expected sample budget.
4. **Learning check.** On the RRM environment, training reward improves monotonically in trend over
   the 8 × 10⁵ steps; a flat curve indicates a broken agent, not a negative result.

If check 2 fails, the arm is not reported at all and the manuscript continues to state that
distributional baselines are absent.

## 7. Decision rules — fixed in advance

| Outcome | Consequence |
|---|---|
| **H2 holds** (inverted is worse, bounds separated) | The measurement-derived-assignment claim is supported. It becomes the paper's algorithmic contribution, stated in the **narrowed** form only (§8) |
| **H2 fails** (inverted is no worse) | **The claim is false.** The algorithmic contribution is dropped. The result is reported as a pre-registered null, as SR-Dn-DQN element B was. The manuscript reverts to its current claim |
| **H2 inconclusive** (bounds overlap) | Reported as a bound, not as support. The claim is **not** made. The MDE is stated so the reader can see what the design could not resolve |
| **H3 fails** (binary-everywhere is better) | The mechanism is binarisation, not assignment. The claim is reframed accordingly and the assignment framing is withdrawn |
| **H4 fails** (Rainbow has lower σ_train) | Reported plainly. The paper's cure is then not the best available cure, and the contribution narrows to the measurement and the attribution |

## 8. The claim, in the only form it may be stated

Following the prior-art gate of 08 Aug 2026, the claim is **domain-specific and ablation-bound**:

> A policy-free, pre-training sensitivity sweep of each controllable radio resource against the
> target KPI is used to assign heterogeneous action-space granularity — a binary actuator to
> dimensions whose measured KPI coupling is cliff-like, incremental control to dimensions measured
> as decoupled — and this measurement-derived assignment is shown, by ablation against
> uniform-granularity action spaces, to reduce between-seed variance of the KPI in a sub-THz
> vehicular RRM controller.

**Explicitly not claimed:** domain generality; novelty of the guard-from-σ rule (this is textbook
link-budget fade margin and is demoted to method detail); novelty of per-dimension sensitivity
sweeping; that bang-bang control is new.

**Must be cited and positioned against:** Seyde et al. NeurIPS 2021 (bang-bang emergence);
Seyde et al. PMLR v242 2024 (adaptive control resolution — which states heterogeneous
discretization "are feasible as well"); Kanervisto et al. IEEE CoG 2020 (action space shaping);
Hussing et al. arXiv:2605.21214 2026 (behavior-consistent RL); Xu et al. 2020 (adaptive
discretization reduces cross-trial variance); Cheng et al. ICML 2019 (control regularisation for
reduced-variance RL).

## 9. What is reported regardless of outcome

All 20 trainings, all five metrics, every arm — including arms that fail their hypothesis, and
including any arm that beats the proposed controller. No arm is dropped for being unflattering. The
per-seed CSVs are committed to the public repository with the rest.

## 10. Standing constraints carried forward

- **Never re-run** `run_frontier.py`, `run_multiseed.py`, `run_multiseed_matched.py`,
  `run_multiseed_rel.py` — they retrain published arms. New arms use new drivers.
- No fabrication, falsification or manipulation. Every reported number traces to a named result file.
- All new prose is swept for the rebuttal register before it ships.
