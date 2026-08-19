# Pre-Registration — SR-Dn-DQN (the new algorithm for Paper 1)

**Frozen:** 2026-07-28, **before** any result of the new algorithm is inspected. Nothing below (metrics,
ablation ladder, success criteria, null outcomes) may be changed after seeing results. Purpose: make the
"new algorithm" claim honest by fixing the bar in advance — no goalpost-moving, no metric added post-hoc.

---

## 1. Motivation & the finding it must cure
Identically-configured trainings of the flat joint controller scatter across a **3.9× reliability range**
(flat Dn-DQN multi-seed: **RLF 11.47 ± 4.18 %**) — the *reliability seed lottery*. SR-Dn-DQN must (i) make
each policy reliable, and (ii) **collapse the cross-seed variance**, as a genuine methodological
contribution (answering Reviewer 4 and the supervisor's request for a new algorithm).

## 2. The algorithm (two novel elements, on the existing branched substrate)
- **Element B — measurement-shielded Bellman backup.** In `branched_td_loss`, mask the unsafe reliability
  action `R_STAY_SAVE` in the **target** max whenever the *next-state* observed serving RSRP is below the
  measured save line (`SAVE_LINE = −92 + Δpower(7) + guard(9) = −76 dBm`). The value backup therefore
  respects the physical shield — the agent *learns* values inside the safe region. This is distinct from
  (a) the existing **post-hoc state guard** (env overrides power after the fact) and (b) **action masking at
  selection** (`train_safe_dqn.py`, changes behaviour only). Element B changes what is *learned*.
- **Element A — cross-seed reliability-variance constraint.** Train a K-member ensemble; add a Lagrangian
  term λ₂ that penalises the **spread of the reliability metric across members** (or, cheap surrogate:
  penalise each new member's deviation from a running reliability target). Drives all members into the
  low-variance, high-reliability regime — i.e. optimises the seed lottery away.
- **Backbone:** `BranchedDuelingQNet` (branches [3,3]), NYU 142 GHz CI channel, reliability-tilt reward
  (0.8/0.1/0.1), sustained-RLF (N310=3). **Optional secondary:** QR-DQN distributional head (tail control) —
  pre-registered as an extension, not required for the primary claim.

## 3. Locked metrics (frozen)
- **Primary:** per-seed **sustained-RLF %**, reported as **mean ± std across training seeds**. The *std* is
  the seed-lottery metric — collapsing it is the whole point.
- **Secondary (paired, same eval seeds):** HO success (over *real* serving-cell changes), power saving %,
  spectrum saving %, throughput (Mbps), link-availability.
- **Eval protocol:** the existing `multi_seed_eval` 5×150 (SEEDS × 150 episodes), identical for every arm.
- No metric may be added or swapped after results are seen.

## 4. Ablation ladder (frozen — this IS the evidence)
| Arm | Config | Purpose |
|---|---|---|
| **A0** | Unguarded branched (no shield, no variance) | floor; expect high & variable RLF |
| **A1** | Post-hoc **state guard** (current RiE proposed, guard_db=9) | the baseline SR-Dn-DQN must match/beat |
| **A2** | **Shielded backup only** (element B) | does moving the shield into learning match/beat A1? |
| **A3** | **Shielded backup + variance constraint** (A+B = full SR-Dn-DQN) | does the variance constraint collapse cross-seed std? |
| **A4** *(opt)* | A3 + distributional QR head | tail control (secondary) |
| **Baselines** | flat Dn-DQN (11.47±4.18), StdDQN (19.29), shielded-action Safe-DQN | context |

## 5. Hypotheses, success criteria, and declared NULL outcomes
- **H1 (element B is real).** *Success:* A2 per-seed RLF ≤ A1 at matched-or-better power/link. *NULL:* if
  A2 RLF exceeds A1 by > 1 pp, element B does **not** help — report it plainly and fall back to the
  post-hoc guard; do not claim the shielded backup.
- **H2 (element A collapses the lottery).** *Success:* A3 **cross-seed RLF std materially below** the flat
  baseline's 4.18 (target < 2.0) at comparable mean RLF. *NULL:* if the std is not reduced, the variance
  constraint fails — report the negative.
- **H3 (a genuine, necessary-component contribution).** *Success:* A3 dominates the flat baseline on the
  reliability-variance frontier **and** removing either element (→ A2 or A1) measurably hurts (each element
  necessary). *NULL:* if a simpler arm matches A3, claim only what survives the ablation.
- **Declared up front:** if **both H1 and H2 fail**, SR-Dn-DQN is not a win. We report the honest negative
  and revert Paper 1 to the finding-plus-methodology framing. No fabrication, no reframing of a null as a
  win.

## 6. Seeds & pairing
- Training seeds `0 … K−1`. **K ≥ 5** for any variance/std claim (H2/H3); **K = 3** permitted for quick
  intermediate ablations, stated explicitly wherever used.
- Eval seeds fixed (`multi_seed_eval.SEEDS`); same eval seeds for every arm (paired comparison).

## 7. Compute & logistics
- Single-threaded CPU (`torch.set_num_threads(1)`), ~800k steps/seed full (reduced steps allowed for
  intermediate ablations, stated). Best-checkpoint, resumable via per-arm CSV.
- **Watchdog** monitors long runs; **all scripts + result CSVs mirrored to the laptop bridge and the cloud
  continuously**, per standing instruction.

## 8. Integrity commitments
Metrics and success bars are fixed by this document. Every reported number will trace to a named result CSV.
The novelty is claimed only as the **defensible conjunction** (the seed-lottery finding + the shielded backup
+ the variance objective), never as a new RL primitive; a full prior-art re-check precedes submission. Any
negative result is reported as-is.

---

*Companion: `DnDQN_Paper1_NewAlgorithm_Proposal_20260728` (design + prior-art scan). Implementation:
`train_srdndqn.py` (to be built on `env_6g_branched.py` / `dndqn_branched.py`).*
