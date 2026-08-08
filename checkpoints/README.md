# Archived model checkpoints

**These are archived because they cannot be regenerated.**

The training driver constructs its environment without a channel seed, so every training episode
draws its fading and its trajectory start from fresh entropy. Two runs at the same declared seed
are therefore *not the same experiment* — that is the central finding of the paper, not an
oversight. It follows that re-running a training script will not reproduce the weights in this
directory, and no seed will make it do so.

Evaluation, by contrast, **is** reproducible: the evaluation channel is seeded per episode
(`seed x 1000 + episode index`, and `BASE + 100000 x repeat + n` for the common-random-number
protocol), so a frozen checkpoint re-evaluated later returns identical numbers. Every table in the
paper can therefore be reproduced exactly from these files plus the evaluation scripts in the
repository root.

`.gitignore` previously excluded `*.pt` on the stated grounds that checkpoints are "regenerable
from the training scripts". That premise is false for this project, and the exclusion has been
removed for this directory.

## Contents

`MANIFEST.csv` lists every file with its original path in the working tree, size, md5 and
modification date. Names are flattened from the working tree: `results/bdq5/bdq5_s0.pt` is archived
as `results__bdq5__bdq5_s0.pt`.

200 checkpoints, ~28 MB total.

## Provenance groups

| prefix | what it is |
|---|---|
| `ms_rel_s*` | flat Dn-DQN, reliability tilt — the published flat arm |
| `gd_s*`, `lot_guard_*` | branched binary + 9 dB state guard — the published proposed arm |
| `br_inc_*`, `results__bdq5__*` | branched incremental (BDQ-style), including the 2026-08-03 retrain that removed the weight-provenance confound |
| `fs_flat_r*`, `vs_flat_r*` | fixed-seed and varying-seed replications behind the "seed is a label" result |
| `results__a1_selection__*` | 2026-08-08 dual-checkpoint campaign: best- and final-step weights from the same runs, used to exclude the checkpoint-selection confound |
| `results__f1_lr__*` | 2026-08-08 learning-rate sweep, flat controller, 1e-4 and 1e-3 |
| `results__f2_guard_lr__*` | 2026-08-08 learning-rate sweep, guarded controller, 1e-4 and 1e-3 |

## Licence

Same as the repository (MIT). The HighD trajectory data these were trained on is **not**
redistributed here and requires its own academic licence from the dataset providers.
