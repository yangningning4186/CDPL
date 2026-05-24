# Step 1: UBT Baseline vs CDPL Class Calibration

This document is generated from server run artifacts under:

`/data4/ynz/cdpl_runs/step1_baseline_vs_cdpl_calib`

Important scope note: this step evaluates the currently implemented CDPL class-calibration stage. It is not Full CDPL with localization-aware loss routing.

## Execution Notes

- Formal comparison: `ubt_baseline_1gpu_seed0` vs `cdpl_calib_1gpu_seed0` on commit `8dd446c16d05`.
- Training limit: `MAX_ITER=36000`, `BURN_UP_STEP=9000`, `EVAL_PERIOD=3000`, `CHECKPOINT_PERIOD=3000`.
- The requested 4-GPU run was attempted first, but it hung after GPU3 reported `nvidia-smi` `Unknown Error`. Follow-up 3-GPU and 2-GPU NCCL probes failed, so the formal run used the verified single-GPU fallback on GPU0 with batch 8 and `BASE_LR=0.005`.
- The server rebooted at `2026-05-24 15:01:06 +0800`, killing active CDPL jobs. The formal CDPL run resumed from `model_0017999.pth` with the OCT-SS data environment restored and completed normally at iter 35999.
- Supplemental repeat `seed1` completed on GPU1 and is recorded in `docs/superpowers/experiments/step1_seed1_summary.md`.
- Supplemental `seed2` baseline completed, but `cdpl_calib_1gpu_seed2` stopped at iter 259 after GPU2 / PCI `86:00.0` reported `nvidia-smi` `Unknown Error`; it was not restarted to avoid risking the formal outputs after the reboot.

## Runs

| Run | Commit | Exit | Complete | mAP | AP30 | AP50 | AP75 | AP_small |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| ubt_baseline_1gpu_seed0 | 8dd446c16d05 | 0 | yes | 9.290 | 41.471 | 24.595 | 5.246 | 3.540 |
| cdpl_calib_1gpu_seed0 | 8dd446c16d05 | 0 | yes | 7.783 | 31.219 | 19.535 | 4.567 | 3.042 |

Formal deltas: mAP `-1.506`, AP30 `-10.253`, AP50 `-5.060`, AP75 `-0.679`, AP_small `-0.498`.

## Artifact Check

### ubt_baseline_1gpu_seed0

- config: yes
- command: yes
- log: yes
- checkpoint: yes
- predictions: yes
- git_commit: yes
- run directory: `/data4/ynz/cdpl_runs/step1_baseline_vs_cdpl_calib/ubt_baseline_1gpu_seed0`

### cdpl_calib_1gpu_seed0

- config: yes
- command: yes
- log: yes
- checkpoint: yes
- predictions: yes
- git_commit: yes
- run directory: `/data4/ynz/cdpl_runs/step1_baseline_vs_cdpl_calib/cdpl_calib_1gpu_seed0`

## Per-Class AP

| Class | ubt_baseline_1gpu_seed0 | cdpl_calib_1gpu_seed0 |
| --- | ---: | ---: |
| D-PED | 3.863 | 3.055 |
| D-SHM | 5.600 | 5.983 |
| ERM | 12.008 | 12.508 |
| F-PED | 5.097 | 5.037 |
| IRF | 12.623 | 12.654 |
| MH | 10.161 | 9.015 |
| RS | 8.860 | 0.127 |
| S-PED | 10.988 | 10.310 |
| SRF | 14.407 | 11.361 |

## Requirement Audit

- Formal baseline/CDPL configs, launch commands, logs, final checkpoints, predictions, git commits, exit codes, and completion markers are present.
- AP30 was recomputed from `inference/coco_instances_results.json` against `/data4/ynz/OCT_SS-main/datasets/annotations/test_v3_1.json`.
- Formal JSON summary is saved as `docs/superpowers/experiments/step1_seed0_summary.json`; supplemental summaries are saved as `step1_seed1_summary.*` and `step1_seed2_baseline_summary.json`.
- Hardware fallback, server reboot, GPU2/GPU3 failures, and the incomplete seed2 CDPL run are documented in this file.

## Judgment

CDPL class calibration changed mAP by -1.506 and AP75 by -0.679.
This Step 1 result is negative for class calibration alone: mAP, AP30, AP50, AP75, AP_small, and most per-class AP values regress, with the largest visible collapse on `RS`.

The next experiment should not treat class calibration as a standalone improvement. Before investing heavily in Full CDPL, inspect pseudo-label retention by class and confidence threshold, then run a tighter ablation: UBT, weak/late class calibration, and Full CDPL with localization-aware routing. Full CDPL is still worth a controlled follow-up only if the localization-aware component explains or reverses this regression.
