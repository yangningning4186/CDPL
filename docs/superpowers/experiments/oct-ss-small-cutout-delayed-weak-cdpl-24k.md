# OCT-SS Small-Scale Cutout：Delayed + Weak CDPL Sweep

## 目标

上一轮 official-style small-scale Cutout 公平实验显示，当前强 CDPL
class calibration 虽然在 final 多保留 `43100` 个伪标签 boxes、在 best_test
多保留 `16796` 个 boxes，但 final 与 best_test 的 mAP/AP30/AP50/AP75/AP_small
均低于 UBT baseline。本轮实验检验更保守的 delayed + weak class calibration
是否能减少这种回退，并观察 `MH` 的潜在正收益是否可保留。

## 路径约束

- 无网络服务器：`ynz@10.77.110.185`。
- 代码目录：`/data2/ynz/CDPL-src`。
- 新输出根目录：`/data2/ynz/cdpl_runs/step2_cdpl_diagnostics_and_weak_calib/oct_ss_small_cutout_delayed_weak_cdpl_24k`。
- 上一轮只读 UBT baseline：`/data2/ynz/cdpl_runs/step2_cdpl_diagnostics_and_weak_calib/oct_ss_small_cutout_ubt_vs_cdpl_24k/ubt_repro_24000iter_1gpu_seed0`。
- 数据路径：`/data2/ynz/cdpl_inputs/annotations/train_labeled.json`、`/data2/ynz/OCT_SS/datasets/annotations/train_unlabeled_v3_1.json`、`/data2/ynz/OCT_SS/datasets/annotations/test_v3_1.json`。
- 图像路径：`/data2/ynz/OCT_SS/datasets/images`。
- Conda 环境：`oct_ss`。
- 禁止使用任何 `OCT_SS-CDPL` 目录。

本轮不覆盖上一轮输出目录。UBT baseline 只作为审计对照复用，不重写其训练产物。

## 固定条件

- Config：`configs/oct_ss_ubt_repro.yaml`。
- `SOLVER.MAX_ITER=24000`。
- Seed：`0`。
- 单卡训练。
- labeled/unlabeled batch：`4+4`。
- official-style small-scale Cutout：
  `(0.7,(0.005,0.01),(0.3,3.3))`、`(0.5,(0.002,0.01),(0.1,6))`、
  `(0.3,(0.002,0.01),(0.05,8))`。
- `BBOX_THRESHOLD=0.5`、`BURN_UP_STEP=5000`。
- `CDPL_TAU_BASE=0.5`、`CDPL_MAX_THRESHOLD_OFFSET=-1.0`。

## 参数矩阵

| Run | 轴 | `CDPL_START_ITER` | `CDPL_RAMP_ITERS` | `CDPL_MIN_CLS_THRESHOLD` | `CDPL_ALPHA_TAIL` |
| --- | --- | ---: | ---: | ---: | ---: |
| `cdpl_dw_s10000_r4000_min045_a005_24000iter_1gpu_seed0` | 主方案 | 10000 | 4000 | 0.45 | 0.05 |
| `cdpl_dw_s10000_r4000_min045_a010_24000iter_1gpu_seed0` | 更强 alpha | 10000 | 4000 | 0.45 | 0.10 |
| `cdpl_dw_s10000_r4000_min040_a005_24000iter_1gpu_seed0` | 更低 floor | 10000 | 4000 | 0.40 | 0.05 |
| `cdpl_dw_s12000_r4000_min045_a005_24000iter_1gpu_seed0` | 更晚启用 | 12000 | 4000 | 0.45 | 0.05 |
| `cdpl_dw_s8000_r4000_min045_a005_24000iter_1gpu_seed0` | 更早启用 | 8000 | 4000 | 0.45 | 0.05 |
| `cdpl_dw_s10000_r8000_min045_a005_24000iter_1gpu_seed0` | 更慢 ramp | 10000 | 8000 | 0.45 | 0.05 |
| `cdpl_dw_s10000_r4000_min048_a005_24000iter_1gpu_seed0` | 更高 floor | 10000 | 4000 | 0.48 | 0.05 |

默认启动脚本会把上述 7 个变体分别分配到 GPU1-7；若某卡显存不足，
启动脚本会拒绝启动，避免覆盖或重复运行。

## 启动与后处理

- 多参数启动脚本：`scripts/experiments/launch_oct_ss_delayed_weak_sweep_24k.sh`。
- 单 run 训练入口：`scripts/experiments/run_oct_ss_repro_24k_run.sh`。
- 后处理脚本：`scripts/experiments/postprocess_oct_ss_delayed_weak_sweep_24k.sh`。
- 后处理 watcher：`scripts/experiments/launch_oct_ss_delayed_weak_sweep_postprocess_watcher.sh`。
- 审计脚本：`scripts/experiments/audit_oct_ss_delayed_weak_sweep_results.py`。

后处理会对上一轮 UBT baseline 与所有 CDPL 变体执行同口径 final/best eval-only，
并为每个 CDPL 变体生成 actual-policy pseudo-label diagnostics。若 best checkpoint
处于 `CDPL_START_ITER` 之前或 ramp 中，diagnostics 会使用对应的实际 ramp factor，
而不是直接套用 full-strength 阈值。

## 状态

等待远端部署与启动。最终结果、diagnostics 与结论将在训练和审计完成后填写。
