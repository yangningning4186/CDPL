# Step 2：CDPL 诊断与弱化类校准实验

本文档用于记录 CDPL Step 2。目标是解释并缓解 Step 1 中 class calibration alone 相比 UBT baseline 的回退。

## 实验约束

- 服务器代码目录：`/data4/ynz/CDPL-src`
- 输出目录：`/data4/ynz/cdpl_runs/step2_cdpl_diagnostics_and_weak_calib`
- 禁止使用：`/data4/ynz/OCT_SS-CDPL`
- 训练上限：`MAX_ITER=24000`
- 评估口径：同时记录 final checkpoint 和 test periodic eval best checkpoint；best checkpoint 需要单独 eval-only 并补算 AP30。

## 待完成

- 基于已有 Step 1 runs 统计 pseudo-label 诊断信息。
- 运行 `late_weak` 类校准实验。
- 运行 `floor_clamp` 类校准实验。
- 汇总 final/best-test 指标、AP30、AP50、AP75、AP_small、每类 AP。
- 判断是否继续实现 localization-aware Full CDPL。
