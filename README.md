# reliability-seed-lottery-6G

Code and results for the paper **"Measurement-Shaped Action Spaces for Deep Reinforcement Learning in 140 GHz Vehicular O-RAN: Exposing and Curing the Reliability Seed Lottery in Joint Handover, Power, and Spectrum Control."**

A single-agent **Dueling *n*-step Double-DQN (Dn-DQN)** controller, deployed as an O-RAN Near-RT RIC xApp, that jointly manages **handover, transmit power, and spectrum (resource-block) allocation** at 140 GHz on a measurement-calibrated NYU 142 GHz close-in channel, evaluated on real **HighD** vehicle trajectories. The headline result is a *measurement-shaped branched controller* — a binary protect/save power actuator plus a non-learned **measured state guard** — that cures the cross-seed "reliability seed lottery" of conventionally-designed DRL controllers.

> **Status:** private repository accompanying a manuscript under review.

## Repository layout

**Environments & channel**
- `channel_model.py`, `channel_model_nyu.py` — path-loss / SINR models (NYU 142 GHz close-in).
- `env_6g_v5.py`, `env_6g_v6.py`, `env_6g_nyu.py` — the 6G sub-THz Gymnasium-style environment (sustained-RLF model; NYU channel upgrade).
- `env_6g_branched.py` — branched-action environment for the cliff-aware joint controller.

**Agents**
- `dndqn.py` — Dueling *n*-step Double-DQN with prioritized experience replay.
- `dndqn_branched.py` — branching dueling Q-network (BDQ) for the proposed measurement-shaped controller.
- `qnet_plain.py` — non-dueling Q-network (architecture ablation).

**Training**
- `train_phase2_nyu.py`, `train_phase2_nyu_rel.py` — proposed Dn-DQN (standard / reliability-tilted reward).
- `train_plain_nyu.py` — Standard-DQN / DDQN+PER baselines.
- `train_branched_nyu.py` — **the proposed branched guarded controller** (Table VII).
- `train_safe_dqn.py`, `train_lagrangian_dqn.py` — Safe-RL baselines (action masking; Lagrangian constraint).
- `train_phase2.py`, `multi_seed_eval.py` — shared training / multi-seed evaluation utilities.

**Measurement & analysis**
- `cliff_sweep_nyu.py` — policy-free power–reliability cliff and RB-flatness sweeps (Sec. III).
- `oracle_power_budget_nyu.py` — policy-free oracle power-saving budget (Sec. III-C).
- `run_frontier.py` — reliability–efficiency reward-tilt frontier (Table III).
- `run_multiseed_rel.py` — the seed lottery: five identical trainings (Table IV).
- `run_multiseed_matched.py` — matched-objective architecture comparison (Table VIII).
- `guard_noise_exp.py` — guard robustness to RSRP noise / reporting delay (Table XIII).
- `significance_nyu.py` — Welch t / Cohen's d significance tests.

**Blockage stress test (Sec. IX-H)**
- `rederive_blockage.py`, `blockage_prep.py` — re-derive 10 Hz pools with real 2D truck-blockage flags from raw HighD.
- `blockage_eval.py`, `blockage_a3.py` — frozen guarded controller vs the 3GPP A3 rule under blockage.

**Evaluation & figures**
- `eval_nyu.py`, `eval_pgrrm_baseline.py` — comparative evaluation (learned + rule baselines).
- `fix_paper1_evals.py` — held-out generalization + matched comparative (Tables XV, X).
- `fix_nstep_ablation_nyu.py`, `fix_reward_ablation_nyu.py`, `fix_alpha_mu_nyu.py` — n-step / reward-component / alpha-mu ablations (Tables XI, II, XII).
- `policy_fig_data.py`, `fig_alpha_mu_sensitivity.py` — figure data / plots.

**`results/`** — per-experiment output CSVs backing the paper's tables and figures (e.g. `cliff_power_sweep_nyu.csv`, `multiseed_rel_nyu.csv`, `branched_ablation_nyu.csv`, `oracle_power_budget_nyu.csv`, `guard_noise_sensitivity.csv`, `blockage_guard_stress.csv`). Every headline number in the paper traces to one of these files.

## Requirements
Python 3.11; a single-threaded CPU is sufficient.
```bash
pip install -r requirements.txt
```

## Data availability
Training and evaluation use the **HighD** naturalistic vehicle-trajectory dataset (recordings 35, 40, 50). HighD is available from its providers under an academic-use licence at <https://www.highd-dataset.com> and is **not redistributed here**. The preprocessed training pool (`6G_Handover_Training_Data_v3.csv`) is derived from HighD and is likewise not included. Note: the original ego-trajectory preprocessing script was not preserved; `rederive_blockage.py` reconstructs an equivalent 10 Hz pool (with real truck-blockage flags) directly from the raw HighD recordings, and the frozen models reproduce their published operating point on it before any blockage claim.

## Reproduction
The core pipeline (channel self-test -> training -> frontier / seed-lottery / matched -> comparative eval -> fixes) is documented step-by-step in **[`REPRODUCE.md`](REPRODUCE.md)**.

Additional analyses for the measurement-shaped controller and the R1 revision:
```bash
python cliff_sweep_nyu.py              # power-reliability cliff + RB flatness  (Fig. 4a/b)
python oracle_power_budget_nyu.py      # oracle power-saving budget            (Fig. 4c)
python train_branched_nyu.py 5 800000  # proposed branched guarded controller  (Table VII)
python guard_noise_exp.py              # guard robustness to noise/delay        (Table XIII)
python fix_reward_ablation_nyu.py      # reward-component ablation              (Table II)
python fix_alpha_mu_nyu.py             # alpha-mu fading sensitivity            (Table XII)
python significance_nyu.py             # Welch t / Cohen's d

# Blockage stress test (Table XIV) — needs raw HighD under ./highD-dataset-v1.0/
python rederive_blockage.py 35 && python rederive_blockage.py 40 && python rederive_blockage.py 50
python blockage_eval.py && python blockage_a3.py
```
Each script's exact arguments and outputs are documented in its module docstring (top of file). Exact per-number equality across machines is **not** expected (PyTorch CPU nondeterminism); the qualitative pattern and the seed mean +/- std are the reproducibility evidence.

## License
Released under the MIT License — see [`LICENSE`](LICENSE).

## Citation
If you use this code, please cite (full publication details to be added on acceptance):
```
Daya Sagar G and Deepa Nivethika S, "Measurement-Shaped Action Spaces for Deep
Reinforcement Learning in 140 GHz Vehicular O-RAN: Exposing and Curing the Reliability
Seed Lottery in Joint Handover, Power, and Spectrum Control," 2026.
```

## Contact
Daya Sagar G — School of Computer Science and Engineering, VIT Chennai. ORCID: 0009-0006-8912-4959.
