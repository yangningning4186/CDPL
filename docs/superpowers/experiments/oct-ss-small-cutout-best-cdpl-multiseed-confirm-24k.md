# OCT-SS Small-Scale Cutout：Best CDPL Multiseed Confirmation

## 目标

上一轮 delayed + weak CDPL sweep 显示，最佳配置
`START=10000/RAMP=4000/MIN=0.45/ALPHA=0.05` 在 seed0 的 `best_test`
口径取得 mAP `10.769`，相对 UBT `+0.276`，并带来 `MH +2.407`、`RS +5.377`、
`AP_small +0.911`。本轮不再扩大扫参，而是验证该收益是否跨 seed 稳定。

## 主结果口径

本轮主结果统一使用 `best_test` checkpoint：先用训练期间 periodic test `bbox/AP`
选择最佳 checkpoint，再单独执行 eval-only，并补算 `mAP`、`AP30`、`AP50`、
`AP75`、`AP_small` 与九类 per-class AP。

`model_final.pth` 不作为方法效果主结果，仅作为训练后段稳定性审计。

## 路径约束

- 无网络服务器：`ynz@10.77.110.185`。
- 代码目录：`/data2/ynz/CDPL-src`。
- 新输出根目录：
  `/data2/ynz/cdpl_runs/step2_cdpl_diagnostics_and_weak_calib/oct_ss_small_cutout_best_cdpl_multiseed_confirm_24k`。
- 不覆盖上一轮 `oct_ss_small_cutout_ubt_vs_cdpl_24k` 或
  `oct_ss_small_cutout_delayed_weak_cdpl_24k`。
- 数据路径：`/data2/ynz/cdpl_inputs/annotations/train_labeled.json`、
  `/data2/ynz/OCT_SS/datasets/annotations/train_unlabeled_v3_1.json`、
  `/data2/ynz/OCT_SS/datasets/annotations/test_v3_1.json`。
- 图像路径：`/data2/ynz/OCT_SS/datasets/images`。
- Conda 环境：`oct_ss`。
- 禁止使用任何 `OCT_SS-CDPL` 目录。

## 固定条件

- Config：`configs/oct_ss_ubt_repro.yaml`。
- `SOLVER.MAX_ITER=24000`。
- 单卡训练。
- labeled/unlabeled batch：`4+4`。
- FocalLoss。
- official-style small-scale Cutout：
  `(0.7,(0.005,0.01),(0.3,3.3))`、`(0.5,(0.002,0.01),(0.1,6))`、
  `(0.3,(0.002,0.01),(0.05,8))`。
- `BBOX_THRESHOLD=0.5`、`BURN_UP_STEP=5000`。
- `CDPL_TAU_BASE=0.5`、`CDPL_MAX_THRESHOLD_OFFSET=-1.0`。

## 实验矩阵

| Run | Seed | Trainer | CDPL_START_ITER | CDPL_RAMP_ITERS | CDPL_MIN_CLS_THRESHOLD | CDPL_ALPHA_TAIL |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| `ubt_ms_confirm_24000iter_1gpu_seed1` | 1 | UBT | NA | NA | NA | NA |
| `ubt_ms_confirm_24000iter_1gpu_seed2` | 2 | UBT | NA | NA | NA | NA |
| `ubt_ms_confirm_24000iter_1gpu_seed3` | 3 | UBT | NA | NA | NA | NA |
| `cdpl_best_dw_s10000_r4000_min045_a005_24000iter_1gpu_seed1` | 1 | CDPL | 10000 | 4000 | 0.45 | 0.05 |
| `cdpl_best_dw_s10000_r4000_min045_a005_24000iter_1gpu_seed2` | 2 | CDPL | 10000 | 4000 | 0.45 | 0.05 |
| `cdpl_best_dw_s10000_r4000_min045_a005_24000iter_1gpu_seed3` | 3 | CDPL | 10000 | 4000 | 0.45 | 0.05 |

seed0 的上一轮结果只作为历史参考。跨 seed 结论主要基于 seed1-3 的同批对照。

## 启动与后处理

- 多 seed 启动脚本：`scripts/experiments/launch_oct_ss_best_cdpl_multiseed_confirm_24k.sh`。
- 单 run 训练入口：`scripts/experiments/run_oct_ss_repro_24k_run.sh`。
- 后处理脚本：`scripts/experiments/postprocess_oct_ss_best_cdpl_multiseed_confirm_24k.sh`。
- 后处理 watcher：`scripts/experiments/launch_oct_ss_best_cdpl_multiseed_confirm_postprocess_watcher.sh`。
- 结果聚合：`scripts/experiments/summarize_oct_ss_best_cdpl_multiseed_confirm.py`。
- 产物审计：`scripts/experiments/audit_oct_ss_best_cdpl_multiseed_confirm_results.py`。

## 状态

等待远端部署与启动。最终结果会在训练、best_test eval-only、CDPL actual-policy
diagnostics、multiseed aggregate 与 artifact audit 完成后填写。

## 判定标准

本轮判断 Best Delayed Weak CDPL 是否跨 seed 稳定优于 UBT，重点关注：

- seed1-3 的 best_test `mAP` 平均 delta 与标准差。
- `AP30`、`AP50`、`AP75`、`AP_small` 的平均 delta。
- `MH`、`RS`、`F-PED`、`S-PED` 的 per-class AP delta。
- CDPL diagnostics 中 actual-policy ramp、`F-PED` mapping、无 numeric key 与
  `cdpl_minus_ubt` 一致性。

