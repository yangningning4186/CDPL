# Step 1: UBT Baseline vs CDPL Class Calibration

This document is generated from server run artifacts under:

`/data4/ynz/cdpl_runs/step1_baseline_vs_cdpl_calib`

Important scope note: this step evaluates the currently implemented CDPL class-calibration stage. It is not Full CDPL with localization-aware loss routing.

## Execution Notes

- This file records the incomplete seed2 repeat.
- `ubt_baseline_1gpu_seed2` completed normally on commit `8dd446c16d05`.
- `cdpl_calib_1gpu_seed2` stopped at iter 259 after GPU2 / PCI `86:00.0` reported `nvidia-smi` `Unknown Error`; no Python traceback or final evaluation was produced.
- The CDPL seed2 run was not restarted after the `2026-05-24 15:01:06 +0800` server reboot because GPU2 may have contributed to the reboot and the priority was preserving formal seed0/seed1 outputs.

## Runs

| Run | Commit | Exit | Complete | mAP | AP30 | AP50 | AP75 | AP_small |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| ubt_baseline_1gpu_seed2 | 8dd446c16d05 | 0 | yes | 9.961 | 38.009 | 26.795 | 6.671 | 4.315 |

## Artifact Check

### ubt_baseline_1gpu_seed2

- config: yes
- command: yes
- log: yes
- checkpoint: yes
- predictions: yes
- git_commit: yes
- run directory: `/data4/ynz/cdpl_runs/step1_baseline_vs_cdpl_calib/ubt_baseline_1gpu_seed2`

## Per-Class AP

| Class | ubt_baseline_1gpu_seed2 |
| --- | ---: |
| D-PED | 4.079 |
| D-SHM | 6.261 |
| ERM | 10.884 |
| F-PED | 7.380 |
| IRF | 14.980 |
| MH | 10.914 |
| RS | 7.996 |
| S-PED | 13.835 |
| SRF | 13.323 |

## Judgment

The seed2 pair is incomplete and must not be used as a baseline-vs-CDPL conclusion. It only confirms that the seed2 baseline artifact is valid and that the CDPL repeat was interrupted by hardware instability.
