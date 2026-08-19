# Prior-art gate — measurement-keyed risk asymmetry in a branched RRM controller

**Run:** 10 August 2026 · adversarial search, looking for work that kills the claim
**Tools:** Consensus (Semantic Scholar / PubMed / Scopus / arXiv), web search, direct fetch
**Level:** deep review · 13 sources retrieved and graded · saturation reached on the load-bearing thread

---

## Bottom line

**The claim survives, but only in a narrowed form, and it is now flanked by three papers that
must be cited and one that must be answered.**

The specific element — *different risk statistics on different action branches within a single
agent, assigned by a pre-training sensitivity measurement* — was not found in this search, and a
2024 systematic review of risk-sensitive RL states it does not identify any work applying distinct
risk measures across action dimensions within one agent [4]. That is the strongest positive
evidence available, and it is an absence-of-evidence result, not a proof.

Three things narrow it. CVaR-based risk-sensitive RL for wireless scheduling **already exists**
in IEEE TWC [5], so no wireless-CVaR novelty is available. Distributional RL for RRM with a
factored action space **already exists** [3]. And most sharply: **RiskQ's ablation already tried
heterogeneous risk measures across agents and reports that it performs poorly** [1] — a reviewer
who knows that paper will ask why your setting escapes its conclusion, and the manuscript must
answer before being asked.

One item is a genuine exposure: **Bjorck et al., ICLR 2022 [9] measures exactly the
between-training-run variance your paper is about, and proposes a competing cure. It is not in
your reference list.**

---

## Verdicts

| # | Question | Verdict | What decides it |
|---|---|---|---|
| 1 | Heterogeneous risk measures across action dimensions/branches within one agent | **SURVIVES (narrowed)** | Not found; review [4] reports none. But RiskQ [1] is adjacent and its ablation is hostile |
| 2 | Pre-training sensitivity measurement deciding *where* risk sensitivity is applied | **SURVIVES** | Nothing found; action-space-shaping literature addresses granularity, not which statistic to optimise |
| 3 | Distributional / CVaR RL in wireless RRM | **KILLED as novelty** | [5] uses CVaR for eMBB/URLLC scheduling; [3] uses QR-DQN for RRM. You do not claim this — keep it that way |
| 4 | Risk-sensitive RL *replacing* a shield/safety layer | **SURVIVES (weak)** | No head-to-head found. [2] and [7] use risk *alongside* safety layers, never instead of |
| 5 | Distributional RL reducing between-seed variance | **SURVIVES** | [8] controls estimator variance, not seed variance — do not let a reviewer conflate them |
| 6 | Run-to-run dispersion of an *operational* metric in wireless DRL | **SURVIVES** | [9] does this for benchmark returns in continuous control, not for a deployed wireless KPI |

---

## Q1 — the load-bearing claim

**Nearest prior work: RiskQ [1].** It models the joint return distribution as weighted quantile
mixtures of per-agent return distributions, and introduces the Risk-sensitive Individual-Global-Max
(RIGM) principle. Two properties matter to you.

First, it is **across agents, not across action dimensions within an agent**. Retrieved directly
from the paper: action selection is per-agent, maximising `ψ_α[Z_i(τ_i, u_i)]`; the framework is
"designed for multi-agent coordination, not intra-agent action factorization."

Second — and this is the one that will be used against you — **RiskQ's ablation deliberately gives
different agents different risk measures and reports that it fails.** From the paper: when about
50% of agents follow different risk measures (VaR₁, CVaR₁, CVaR₀.₂) than the rest using Wang₀.₇₅,
RiskQ "performs poorly in all the three cases, highlighting the importance of satisfying the RIGM
principle that agents act according to the same risk measure."

**Inference (mine, not the paper's):** that negative result should not transfer to your setting,
and the reason is structural rather than empirical. RIGM is a *credit-assignment* constraint — it
exists so that decentralised per-agent argmaxes reconstruct the centralised risk-sensitive argmax.
Your controller is a single agent with a factored action space, one shared scalar reward, and no
decentralised execution requirement, so there is no individual-global-max consistency to violate.
**This argument must be in the manuscript before a referee raises it, not after.** It is the single
most likely attack on the contribution.

**Supporting negative evidence:** the 2024 systematic review of risk sensitivity in Markov games
and MARL [4] "does not identify any work applying distinct risk measures to different action
dimensions, factors, or branches within a single agent's decision-making." Treat this as a strong
but not conclusive signal — a review of *multi-agent* risk sensitivity is not guaranteed to cover
single-agent factored-action work.

**Confidence: moderate.** Two independent lines (a targeted search and a systematic review) agree,
but both are absence arguments and neither field indexes this exact combination well.

## Q2 — the assignment rule

Nothing found that uses a pre-training measurement to decide which action dimension receives which
*treatment*. The adjacent literature — action space shaping, adaptive discretisation, decomposed
action spaces — concerns how finely to discretise or how to factor, not which statistic to optimise
per factor. **Confidence: moderate-low**, because this is a hard thing to search for; the concept
has no settled term of art, which is itself weak evidence that it is not established.

## Q3 — killed, and you should say so plainly

**[5] Zhang et al., IEEE Transactions on Wireless Communications, 2024** is the closest wireless
work. It "incorporates the conditional Value-at-Risk (CVaR) as the risk criterion for optimization"
for eMBB/URLLC joint scheduling, with a distributional critic that "predicts the entire distribution
over future returns instead of simply the expectation," and reports that across CVaR levels it
"achieves similar expected returns but reduces long-tail behavior."

That is CVaR-greedy risk-sensitive control in wireless, published, in a strong venue. It does not
do per-branch asymmetry and does not touch seed variance — but any sentence implying you are first
to bring CVaR to radio resource management is dead on arrival.

**[3] arXiv 2409.16764** is the nearest distributional-RRM work: QR-DQN plus conservative Q-learning
for user scheduling, with a factored action space across access points. Retrieved from the paper:
action selection "appears to use expected values from the learned quantile distribution" — the
distribution is used for representation, not for risk-sensitive action selection. It contains **no
discussion of random seeds, multiple independent runs, or run-to-run variance**, which is
corroborating evidence for your reproducibility gap.

## Q4 — the guard-replacement claim

No paper found showing a risk-sensitive learner *replacing* a hand-designed shield. The literature
runs the other way:

- **SafeSlice [2]** (O-RAN slicing, 2025) combines "a sigmoid-based risk-sensitive reward function"
  *with* "a safety layer that projects the slicing agent's RA actions to the nearest safe actions."
  Risk and shield together, not one instead of the other. This is the closest structural analogue
  to your guard and should be cited when you describe it.
- **CPPO [7]** constrains CVaR below a threshold rather than shielding, but does not compare against
  a shield.

**Inference:** the head-to-head — same system, same reward, learned risk rule versus hand-placed
safety layer, with the layer removed — appears genuinely unaddressed. If `qr_asym_u` works, this is
the most defensible part of the contribution, and it is worth more than the asymmetry itself.

## Q5 and the reproducibility angle — read this one carefully

Two traps.

**Trap one: [8] Kuang et al., ICML 2023, "Variance Control for Distributional RL."** The title
looks like it kills Q5. It does not. Retrieved from the paper: it controls "the variance of the
Q-function error terms themselves, not return distribution variance, gradient variance, or seed
variance." Different quantity entirely. But the title is close enough that a referee may cite it at
you, so pre-empt it in one clause.

**Trap two, and the real exposure: [9] Bjorck, Gomes & Weinberger, ICLR 2022, "Is High Variance
Unavoidable in RL? A Case Study in Continuous Control."** This measures between-seed variance of
final performance, identifies "poor outlier runs which completely fail to learn" as a major source,
explicitly rules out weight initialisation and initial exploration as causes, blames "unstable
network parametrization which leads to saturating nonlinearities," and proposes penultimate feature
normalisation, reporting a reduction in average standard deviation by a factor greater than 3 across
21 tasks.

That is **a competing cure for your headline problem**, and it is cheap — a normalisation layer,
not an action-space redesign. It is not in your reference list. Two consequences:

1. Cite it. A paper whose subject is between-training variance that omits the ICLR paper on
   between-training variance looks like an incomplete literature search.
2. Expect "did you try penultimate normalisation?" The honest answer is that you did not, and the
   distinction that protects you is scope: [9] measures **benchmark returns in continuous control**,
   while you measure an **operational wireless KPI whose coupling to one action axis is a measured
   cliff**. Your mechanism is physical and domain-specific; theirs is an optimisation pathology.
   Those can both be true at once. Say so, and list normalisation as untested.

---

## The narrowed wording the claim must be restricted to

> Within a single-agent branched value function for sub-THz vehicular radio resource management, a
> policy-free pre-training sensitivity sweep of each controllable resource against the target KPI is
> used to assign **heterogeneous risk statistics across action branches** — a tail statistic (CVaR)
> to the branch whose measured KPI coupling is cliff-like, the expectation to the branch measured as
> decoupled — and this assignment is shown by ablation against symmetric-distributional and
> symmetric-expectation baselines, and against an inverted assignment, to reduce the between-training
> dispersion of an operational reliability metric **without a non-learned safety layer**.

**Must not be claimed:** novelty of distributional value learning, of CVaR-greedy control, of
CVaR in wireless resource management [5], of distributional RL in RRM [3], of branched action
spaces, or of risk-sensitive RL generally.

**Must be cited and positioned against:** RiskQ [1] (and its heterogeneous-risk ablation, answered
explicitly), Zhang et al. [5], the distributional-RRM work [3], SafeSlice [2] as the
risk-plus-safety-layer arrangement, Lim et al. [10] for the CVaR action-selection subtlety in
distributional RL, and Bjorck et al. [9] as a competing cure.

**Must be disclosed:** α = 0.25 and N = 51 are literature defaults, unswept.

---

## Evidence table

| # | Claim it supports | Source | Tier | What the source actually says | Confidence |
|---|---|---|---|---|---|
| 1 | Nearest risk+factorization work; hostile ablation | Shen et al. 2023, RiskQ | C (arXiv, 33 cit.) | "the collection of risk-sensitive action selections of each agent should be equivalent to the risk-sensitive action selection of the central policy"; heterogeneous-risk ablation "performs poorly in all the three cases" | High — fetched full text |
| 2 | Risk + safety layer used together in O-RAN | Nagib et al. 2025, SafeSlice | B (IEEE ICMLCN) | "sigmoid-based risk-sensitive reward function" plus "a safety layer that projects the slicing agent's RA actions to the nearest safe actions" | High |
| 3 | Distributional RL in RRM exists; no seed-variance analysis | arXiv 2409.16764 | C (preprint) | QR-DQN + conservative Q-learning; action selection uses expected values; "no mention found" of seeds or run-to-run variance | High — fetched |
| 4 | No per-dimension risk heterogeneity in the literature | Systematic review, arXiv 2406.06041 | C (preprint review) | "does not identify any work applying distinct risk measures to different action dimensions, factors, or branches within a single agent" | Moderate — MARL-scoped |
| 5 | CVaR in wireless scheduling is prior art | Zhang et al. 2024, IEEE TWC | A | "incorporates the conditional Value-at-Risk (CVaR) as the risk criterion for optimization"; "reduces long-tail behavior" | High |
| 7 | Risk constraint as an alternative to shielding | Ying et al. 2022, CPPO | C (arXiv, 71 cit.) | "formalizes the risk-sensitive constrained optimization problem by keeping its CVaR under a given threshold" | Moderate |
| 8 | "Variance control" ≠ seed variance | Kuang et al. 2023, ICML | A (PMLR) | controls "the variance of the Q-function error terms themselves, not… seed variance" | High — fetched |
| 9 | Competing cure for between-seed variance | Bjorck, Gomes & Weinberger 2022, ICLR | A | "poor outlier runs which completely fail to learn"; "weight initialization and initial exploration are not at fault"; penultimate normalisation reduces average std by ">3" across 21 tasks | High — fetched |
| 10 | CVaR action selection in distributional RL is subtle | Lim & Malalur 2022, NeurIPS | A | "the standard action-selection strategy when applying the distributional Bellman optimality operator can result in convergence to neither the dynamic, Markovian CVaR nor the static, non-Markovian CVaR" | High |
| 11 | Branched Q-network for joint subcarrier + power, same journal | Xu et al. 2026, Sci. Rep. | B | "action-branching architecture to independently optimize subcarrier allocation, discrete transmit power levels, and computational resource allocation" | Moderate — **URL not resolved, verify by hand** |
| 12 | BDQ in wireless resource management | Wei et al. 2020, IEEE TNSM | A | BDQ for network slice reconfiguration to handle "multi-dimensional discrete action space" | High |

---

## A methodological warning worth more than the citations

**[10] Lim & Malalur, NeurIPS 2022** is a correctness issue for your implementation, not just a
citation. It shows "the standard action-selection strategy when applying the distributional Bellman
optimality operator can result in convergence to neither the dynamic, Markovian CVaR nor the static,
non-Markovian CVaR," and proposes a modified distributional Bellman operator.

Your `q4_qr_branched.py` does exactly the standard thing: it selects the next action by the online
network's CVaR and bootstraps from the target network's quantiles at that action. **That is the
strategy this paper says converges to neither CVaR definition.**

This does not invalidate the experiment — the arm is still a well-defined algorithm and the ablation
against symmetric baselines is still valid — but you cannot write "the branch optimises CVaR"
without qualification. Either implement their corrected operator, or state precisely that the branch
uses CVaR-greedy action selection under the standard distributional Bellman operator, with the
convergence caveat cited. **The second option is cheap and honest; the first costs another build.**

---

## Search strategy

**Threads:** (1) heterogeneous/per-dimension risk in factored action spaces; (2) measurement-driven
per-dimension treatment assignment; (3) distributional/CVaR RL in wireless RRM; (4) risk-sensitive RL
replacing shields; (5) distributional RL and seed variance; (6) run-to-run dispersion of operational
metrics in wireless DRL.

**Queries run (10 Aug 2026):** Consensus — "risk-sensitive reinforcement learning factored action
space per-dimension risk measure"; "branching dueling Q-network action branch architecture
multi-dimensional discrete control"; "risk-sensitive deep reinforcement learning O-RAN xApp network
slicing conditional value at risk". Web — heterogeneous risk per action dimension with CVaR on one
branch and expectation on another; distributional RL CVaR handover mmWave 6G RRM; risk-sensitive RL
replacing safety shield / action masking; distributional RL and seed variance / reproducibility;
action-space design via sensitivity analysis; per-branch risk in single-agent factored spaces;
wireless RL seed variance of operational KPIs. Direct fetch — RiskQ full text, the distributional-RRM
preprint, the MARL risk systematic review, the ICML variance-control paper, the ICLR variance paper.

**Failed or partial retrievals, logged rather than hidden:** one Consensus query returned a rate-limit
error and was not re-run (distributional RL + CVaR + wireless power control) — partially covered by
the later O-RAN query and web searches, but coverage on that exact phrasing is thinner than the rest.
The abstract-page fetch of arXiv 2409.16764 returned almost nothing; the HTML version was fetched
instead and used. **The Scientific Reports 2026 PER-BDQ paper [11] could not be resolved to a DOI or
publisher URL** — the only hits were the Consensus record and a Chinese aggregator page. Its metadata
is therefore unverified and it must be looked up by hand before it is cited.

**Not found, which is itself information:** no paper applying different risk measures to different
action branches within one agent; no paper using a pre-training sensitivity sweep to assign risk
sensitivity; no head-to-head of risk-sensitive learning against a removed safety shield; no paper
measuring between-seed variance of an operational wireless KPI.

**Stopping rule:** saturation — the final two query variants returned only already-seen material.

---

## Residual uncertainty

Every surviving verdict rests on absence of evidence from one afternoon's search across four
databases, and the concept has no settled term of art, which makes recall unreliable. Two specific
blind spots: non-English literature, and patents. Before the claim ships I would also run the exact
phrase families "per-branch risk", "branch-wise risk", "dimension-wise risk sensitivity" and
"asymmetric risk factored MDP" against Google Scholar directly, which this session could not query.

---

## References

All entries were retrieved in this session on 10 August 2026. Nothing here is reconstructed from
memory. Where a record came only from the Consensus index (Semantic Scholar / Scopus / arXiv
aggregation) rather than from the publisher page, that is stated.

1. Shen, S. et al. **RiskQ: Risk-sensitive Multi-Agent Reinforcement Learning Value Factorization.**
   arXiv, 2023. 33 citations. Full text retrieved: https://arxiv.org/html/2311.01753v2 ·
   index record: https://consensus.app/papers/details/d91432ce165451a39a8105f6298933c5/

2. Nagib, A. M. et al. **SafeSlice: Enabling SLA-Compliant O-RAN Slicing via Safe Deep Reinforcement
   Learning.** IEEE ICMLCN, 2025. 9 citations. Index record:
   https://consensus.app/papers/details/d93f696cbf0e5bc69c18f961c52646ac/

3. **Offline and Distributional Reinforcement Learning for Radio Resource Management.** arXiv
   preprint 2409.16764, 2024. Full text retrieved: https://arxiv.org/html/2409.16764v1 ·
   abstract page: https://arxiv.org/abs/2409.16764

4. **Risk Sensitivity in Markov Games and Multi-Agent Reinforcement Learning: A Systematic Review.**
   arXiv preprint 2406.06041, 2024. Full text retrieved: https://arxiv.org/html/2406.06041v1

5. Zhang, W. et al. **Constrained Risk-Sensitive Deep Reinforcement Learning for eMBB-URLLC Joint
   Scheduling.** IEEE Transactions on Wireless Communications, 2024. 16 citations. Index record:
   https://consensus.app/papers/details/7e1134073b3f547d9af1fe64a779d96e/

6. Tavakoli, A. et al. **Action Branching Architectures for Deep Reinforcement Learning.** arXiv,
   2017. 335 citations. Index record:
   https://consensus.app/papers/details/dbaf819b66ae5fbeab6180291b768e7b/

7. Ying, C. et al. **Towards Safe Reinforcement Learning via Constraining Conditional Value-at-Risk.**
   arXiv, 2022. 71 citations. Index record:
   https://consensus.app/papers/details/6689382ee003596f952e737c2ad78fb1/

8. Kuang, Q. et al. **Variance Control for Distributional Reinforcement Learning.** ICML 2023,
   PMLR v202. Retrieved: https://proceedings.mlr.press/v202/kuang23a.html

9. Bjorck, J., Gomes, C. P. & Weinberger, K. Q. **Is High Variance Unavoidable in RL? A Case Study
   in Continuous Control.** ICLR 2022. arXiv 2110.11222. Full text retrieved:
   https://ar5iv.labs.arxiv.org/html/2110.11222 · author copy:
   https://www.cs.cornell.edu/gomes/pdf/2022_bjorck_iclr_variance.pdf · index record:
   https://www.semanticscholar.org/paper/c578a295ba3456bebc651f1f516e34ea6e7a5fe3

10. Lim, S. H. & Malalur, I. **Distributional Reinforcement Learning for Risk-Sensitive Policies.**
    Advances in Neural Information Processing Systems 35, 2022. 52 citations. Index record:
    https://consensus.app/papers/details/fdd4badd5ccc5e9a8c5acbff2a7454fc/

11. Xu, S. et al. **Multi-dimensional joint resource scheduling for 5G eMBB by a PER-branch dual deep
    Q-network.** Scientific Reports, 2026. Index record only:
    https://consensus.app/papers/details/3aeb6d6f020a5c8286d3dfa4b4024af0/ ·
    **DOI and publisher URL could not be resolved in this session — verify by hand before citing.**

12. Wei, F. et al. **Network Slice Reconfiguration by Exploiting Deep Reinforcement Learning With
    Large Action Space.** IEEE Transactions on Network and Service Management, 2020. 123 citations.
    Index record: https://consensus.app/papers/details/e670b6d37b9f5060af6928f75942e359/

13. Moghimi, M. et al. **Risk-sensitive Actor-Critic with Static Spectral Risk Measures for Online
    and Offline Reinforcement Learning.** Expert Systems with Applications, 2025. 3 citations.
    Index record: https://consensus.app/papers/details/5549667f50ad5bc09bc9ad320befc00a/

### Verification status

`verify_citations.py` was run twice.

**Pass 1, before the reference list existed** (inline DOI/arXiv scan only): **NOT FOUND 0,
MISMATCH 0, UNREACHABLE 2, NO IDENTIFIER 0.** The two UNREACHABLE entries were arXiv:2409.16764
and arXiv:2406.06041 — the arXiv API returned HTTP 403 to this container, so the script could not
confirm them.

**Pass 2, with the reference list present: the script crashed** —
`NameError: name 'YEAR_RE' is not defined` at `verify_citations.py:370`. That is a defect in the
checker, not a problem with the citations, but it means **no automated pass was obtained over the
reference list** and I am not going to describe this report as machine-verified.

What was verified by hand instead: both UNREACHABLE arXiv IDs were **fetched and read directly in
this session** — 2409.16764 returned the offline/distributional RRM paper and 2406.06041 returned
the risk-sensitivity systematic review, matching the titles claimed here. Entries 8, 9 and 1 were
likewise fetched from their publisher, author-copy or full-text pages. Entries 2, 5, 6, 7, 10, 12
and 13 rest on Consensus index records only, with authors, year, venue and abstract as returned by
that index; their publisher pages were not opened. **Entry 11 has no resolvable identifier at all.**
No retraction notices were encountered on any page opened.
