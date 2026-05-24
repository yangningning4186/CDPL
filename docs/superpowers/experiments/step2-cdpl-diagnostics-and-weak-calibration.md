# Step 2：CDPL 诊断与弱化类校准实验

本文档用于记录 CDPL Step 2。目标是解释并缓解 Step 1 中 class calibration alone 相比 UBT baseline 的回退。

## 实验约束

- 服务器代码目录：`/data4/ynz/CDPL-src`
- 输出目录：`/data4/ynz/cdpl_runs/step2_cdpl_diagnostics_and_weak_calib`
- 禁止使用：`/data4/ynz/OCT_SS-CDPL`
- 训练上限：`MAX_ITER=24000`
- 评估口径：同时记录 final checkpoint 和 test periodic eval best checkpoint；best checkpoint 需要单独 eval-only 并补算 AP30。

## 核心问题

Step 1 的正式 seed0 结果显示，class calibration alone 相比 UBT baseline 在 mAP、AP30、AP50、AP75 上整体回退，只有 AP_small 小幅改善；回退较明显的类别包括 `RS`、`F-PED`、`MH`。
Step 2 先不进入 localization-aware Full CDPL，而是先回答两个更低风险的问题：

1. class calibration 是否把太多低质量 pseudo boxes 放进训练，尤其影响 `RS`、`F-PED`、`MH`。
2. 如果只延后、减弱或限制 calibration 的阈值变化，是否能保留 tail/small-object 收益，同时减少主指标回退。

## 诊断输出

使用已有 Step 1 checkpoints 在 unlabeled set 上统计 teacher pseudo labels：

- 每类 teacher boxes 总数。
- UBT threshold kept 数量和比例。
- CDPL threshold kept 数量和比例。
- `cdpl_minus_ubt`、drop/keep ratio。
- 全量、UBT kept、CDPL kept 的 score 分布。
- 全量、UBT kept、CDPL kept 的 box 面积分布。
- 重点查看 `RS`、`F-PED`、`MH`，同时保留全部类别记录。

计划输出位置：

- `/data4/ynz/cdpl_runs/step2_cdpl_diagnostics_and_weak_calib/diagnostics/step1_ubt_baseline_seed0_best_iter17999.json`
- `/data4/ynz/cdpl_runs/step2_cdpl_diagnostics_and_weak_calib/diagnostics/step1_cdpl_calib_seed0_best_iter11999.json`
- 如时间允许，同步生成 seed1 诊断作为稳定性参考。

## 改进实验

| 实验 | 目的 | 关键设置 |
| --- | --- | --- |
| `late_weak` | 延后并减弱 class calibration，避免 burn-in 后立刻注入大量低分伪标签 | `CDPL_ALPHA_TAIL=0.05`，`CDPL_START_ITER=18000`，`CDPL_RAMP_ITERS=6000`，`CDPL_MIN_CLS_THRESHOLD=0.35` |
| `floor_clamp` | 限制阈值下降幅度，避免 tail 类阈值过低 | `CDPL_ALPHA_TAIL=0.15`，`CDPL_MIN_CLS_THRESHOLD=0.45`，`CDPL_MAX_THRESHOLD_OFFSET=0.05` |

训练上限统一为 `MAX_ITER=24000`，每 `3000` iter 做一次 test periodic eval 和 checkpoint。
优先使用空闲 GPU 并行单卡 seeds；如果 NCCL 或 GPU2/GPU3 再次不稳定，正式 seed0 保留在更稳定的 GPU0/GPU1 上，其他 seed 作为补充结果。

## 评估口径

- `final`：对 `model_final.pth` 单独 eval-only，并从预测 JSON 计算 mAP、AP30、AP50、AP75、AP_small、每类 AP。
- `best_test`：从 `metrics.json` 中选择 periodic test `bbox/AP` 最高的 checkpoint；不使用 val；再单独 eval-only，并补算同一组指标。
- 每个 run 需保留 `config_input.yaml`、`config_overrides.json`、`launch_command.sh`、`train.log`、`git_commit.txt`、checkpoint、eval 输出和 pseudo-label 诊断 JSON。

## 当前状态

- 本地已实现 late/weak、threshold floor/offset clamp 两组配置开关和启动脚本。
- 本地已实现 Step 1 pseudo-label 诊断脚本和 Step 2 final/best-test 汇总脚本。
- 等待把最新本地提交同步到服务器后，先重跑诊断 smoke，确认 `F-PED` 等类别阈值映射正确，再启动完整诊断和 24000 iter 实验。
