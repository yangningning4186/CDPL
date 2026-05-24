# Step 1: UBT Baseline vs CDPL Class Calibration

This document is generated from server run artifacts under:

`/data4/ynz/cdpl_runs/step1_baseline_vs_cdpl_calib`

Important scope note: this step evaluates the currently implemented CDPL class-calibration stage. It is not Full CDPL with localization-aware loss routing.

## Execution Notes

- This is a supplemental repeat, not the formal Step 1 comparison. The formal seed0 result is in `docs/superpowers/experiments/step1-baseline-vs-cdpl-calibration.md`.
- `cdpl_calib_1gpu_seed1` resumed after the `2026-05-24 15:01:06 +0800` server reboot from `model_0014999.pth` and completed at iter 35999.
- The repeat was run on the verified single-GPU fallback path on GPU1 after multi-GPU NCCL probes failed.

## Runs

| Run | Commit | Exit | Complete | mAP | AP30 | AP50 | AP75 | AP_small |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| ubt_baseline_1gpu_seed1 | 8dd446c16d05 | 0 | yes | 9.176 | 37.556 | 24.211 | 5.205 | 4.009 |
| cdpl_calib_1gpu_seed1 | 8dd446c16d05 | 0 | yes | 7.524 | 29.292 | 19.596 | 4.141 | 2.446 |

Supplemental deltas: mAP `-1.652`, AP30 `-8.264`, AP50 `-4.615`, AP75 `-1.064`, AP_small `-1.564`.

## Artifact Check

### ubt_baseline_1gpu_seed1

- config: yes
- command: yes
- log: yes
- checkpoint: yes
- predictions: yes
- git_commit: yes
- run directory: `/data4/ynz/cdpl_runs/step1_baseline_vs_cdpl_calib/ubt_baseline_1gpu_seed1`

### cdpl_calib_1gpu_seed1

- config: yes
- command: yes
- log: yes
- checkpoint: yes
- predictions: yes
- git_commit: yes
- run directory: `/data4/ynz/cdpl_runs/step1_baseline_vs_cdpl_calib/cdpl_calib_1gpu_seed1`

## Per-Class AP

| Class | ubt_baseline_1gpu_seed1 | cdpl_calib_1gpu_seed1 |
| --- | ---: | ---: |
| D-PED | 3.252 | 2.177 |
| D-SHM | 5.141 | 5.006 |
| ERM | 11.823 | 12.806 |
| F-PED | 4.916 | 3.240 |
| IRF | 12.262 | 10.959 |
| MH | 9.587 | 9.603 |
| RS | 6.361 | 0.358 |
| S-PED | 13.657 | 10.106 |
| SRF | 15.589 | 13.462 |

## Judgment

CDPL class calibration changed mAP by -1.652 and AP75 by -1.064.
This repeat supports the formal seed0 conclusion: class calibration alone regressed the main detection metrics. It should be used as supporting evidence, not as an independent positive result.
