# Paper 1 — Reproduction Commands (Windows PowerShell)

Run these on your machine to reproduce every Paper 1 number. I run the **same** scripts in the
cloud in parallel, so your CSVs and mine cross-confirm (reproducibility check). Compare each
`results\*.csv` you produce against the cloud copy of the same name.

> Order matters: training steps (2–4) must run before the eval/ablation steps (5–10).
> All commands are run from the folder that contains the scripts **and** `6G_Handover_Training_Data_v3.csv`.

## 0. Setup
```powershell
cd "D:\DAYA PHD\PHD WORK\Startfromscratch"
# optional but recommended: isolated env
python -m venv .venv ; .\.venv\Scripts\Activate.ps1
pip install numpy pandas torch
```

## 1. Channel calibration self-test  (expect FSPL(1 m)=75.36 dB, CI(n=2)==free-space)
```powershell
python -u channel_model_nyu.py
```

## 2. Train the proposed Dn-DQN on the NYU channel
```powershell
# standard reward -> dqn_phase2_dndqn_nyu.pt
python -u train_phase2_nyu.py
# reliability-tilted reward (W 0.8/0.1/0.1, RLF -300) -> dqn_phase2_dndqn_nyu_rel.pt
python -u train_phase2_nyu_rel.py
```

## 3. Train the plain baselines on the NYU channel
```powershell
# DDQN+PER (no dueling, no n-step). IN is a dueling ckpt -> auto RANDOM init (intended).
python -u train_plain_nyu.py dqn_phase2_dndqn_nyu.pt dqn_ddqnper_nyu.pt ddqn 800000
# Standard DQN (no dueling, no double, no n-step, no PER)
python -u train_plain_nyu.py dqn_phase2_dndqn_nyu.pt dqn_stddqn_nyu.pt std 800000
```

## 4. Reliability–efficiency FRONTIER sweep  (the asymmetry finding; makes fr_C1..fr_C5)
```powershell
python -u run_frontier.py 500000        # -> results\frontier_nyu.csv
```

## 5. Multi-seed reliability robustness  (5 seeds x 800k; makes ms_rel_s0..4)
```powershell
python -u run_multiseed_rel.py 5 800000  # -> results\multiseed_rel_nyu.csv
```

## 6. Matched-objective architecture ablation  (Proposed vs DDQN+PER vs Standard; makes mm_*)
```powershell
python -u run_multiseed_matched.py 3 400000  # -> results\matched_rel_nyu.csv + matched_rel_summary.csv
```

## 7. Comparative table  (Proposed vs baselines + rule A3/PG-RRM, standard-reward point)
```powershell
python -u eval_nyu.py            # -> results\eval_nyu.csv
```

## ---- Paper 1 FIXES (new scripts) ----

## 8. Held-out generalization + matched-reliability comparative  (EVAL-ONLY; needs steps 2,3,4)
```powershell
python -u fix_paper1_evals.py    # -> results\generalization_fast_nyu.csv, results\comparative_matched_nyu.csv
```

## 9. n-step ablation on the NYU channel  (n=1,5 x seeds 0-2; n=3 reuses step 5; needs step 5 done)
```powershell
python -u fix_nstep_ablation_nyu.py   # -> results\nstep_ablation_nyu.csv
```
> On your machine this starts immediately (it only waits if `run_multiseed_rel` is actively running).

## Cross-confirmation checklist
| Your file | Compare vs cloud | Should match (training-seed variance aside) |
|---|---|---|
| frontier_nyu.csv | frontier_nyu.csv | asymmetry: Spec ~82–90% across tilts, Power swings 2–87% |
| multiseed_rel_nyu.csv | multiseed_rel_nyu.csv | RLF ~10–17% per seed |
| matched_rel_summary.csv | matched_rel_summary.csv | Proposed RLF lowest + Spec highest, most consistent |
| generalization_fast_nyu.csv | generalization_fast_nyu.csv | held-out `fast`: HO ~96%, RLF ~2%, Spec ~87% |
| nstep_ablation_nyu.csv | nstep_ablation_nyu.csv | n=3 best Spec vs n=1 |

Exact per-number equality is not expected across machines (Torch CPU nondeterminism differs);
the **qualitative pattern + seed-mean±std** must match. That agreement is the reproducibility evidence.
