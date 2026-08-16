# Provenance note — the `incremental` units in this directory are SUPERSEDED

**Added 16 August 2026 during a full numerical audit of the manuscript.**

`unit_incremental_s0..s4_laptop.csv` in this directory are **not** the source of the
branched-incremental (BDQ-style) row of manuscript Table 1, and recomputing that row from them will
disagree with the paper.

## Why

These five units mix weight provenance: seeds 0 and 1 are the surviving 2026-07-12 originals
(`br_inc_s0.pt`, `br_inc_s1.pt`), while seeds 2–4 are regenerations (`br_inc_regen_s{2,3,4}.pt`,
2026-08-01) produced after the originals were lost. When the reported quantity is a **variance**,
every source of variance inside the sample must be the one being claimed — a sample built from two
provenances confounds training-seed variance with pipeline variance.

The arm was therefore **retrained in full**, all five checkpoints from one protocol under a single
trainer hash, and the manuscript reports that campaign.

## What the manuscript actually uses

| | source | RLF |
|---|---|---|
| **Table 1, branched-incremental row** | `results/bdq5/bdq5.csv` (`bdq5_s0.pt`…`bdq5_s4.pt`, origin `retrained-20260803`) | 10.05 ± 2.30 |
| these superseded units | `unit_incremental_s*_laptop.csv` (2 original + 3 regenerated) | 12.49 ± 5.88 |

Manuscript paragraph following Table 1 states this explicitly: *"The branched-incremental arm was
retrained in full — all five checkpoints (bdq5_s0.pt to bdq5_s4.pt, origin retrained-20260803) from
one protocol under a single trainer hash — so no arm here mixes weight provenance."*

The other three arms in this directory — `flat`, `binary`, `guarded` — **are** the Table 1 sources
and reproduce the published row exactly.

## Why these files are kept rather than deleted

They are the evidence for the provenance statement above, and `results/tablevii_r8_laptop.csv` and
`results/tablevii_r8_summary_laptop.json` were produced from them. They are retained for audit and
marked here so that no reader mistakes them for the reported arm.
