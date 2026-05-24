# Step 1：UBT Baseline vs CDPL 类校准

本文档由服务器实验产物自动汇总并人工补充说明，实验根目录为：

`/data4/ynz/cdpl_runs/step1_baseline_vs_cdpl_calib`

重要范围说明：本步骤只评估当前已经实现的 CDPL 类校准阶段，不是带有定位感知 loss routing 的 Full CDPL。主结果按 test 集 periodic eval 的最高 mAP checkpoint 统计，不使用最终 checkpoint 作为主结果。

## 执行说明

- 这是补充重复实验，不是正式 Step 1 对比。正式 seed0 结果见 `docs/superpowers/experiments/step1-baseline-vs-cdpl-calibration.md`。
- `cdpl_calib_1gpu_seed1` 在 `2026-05-24 15:01:06 +0800` 服务器重启后，从 `model_0014999.pth` 恢复，并训练到 iter 35999。
- 多卡 NCCL 探测失败后，该重复实验走已验证的单卡 fallback，在 GPU1 上完成。
- 每 3000 iter 的 test 结果记录在各 run 的 `metrics.json`。本表先按 test `bbox/AP` 选择各自最优 checkpoint，再对该 checkpoint 单独跑 eval-only 生成预测文件并补算 AP30。

## 实验结果

| 实验 | 最优 iter | checkpoint | 提交 | mAP | AP30 | AP50 | AP75 | AP_small |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| ubt_baseline_1gpu_seed1 | 20999 | `model_0020999.pth` | 8dd446c16d05 | 10.591 | 41.950 | 27.306 | 6.985 | 5.090 |
| cdpl_calib_1gpu_seed1 | 11999 | `model_0011999.pth` | 8dd446c16d05 | 9.510 | 39.290 | 25.150 | 5.723 | 5.235 |

补充实验差值：mAP `-1.081`，AP30 `-2.660`，AP50 `-2.156`，AP75 `-1.262`，AP_small `+0.145`。

## 产物检查

### ubt_baseline_1gpu_seed1

- 配置文件：是
- 启动命令：是
- 训练日志：是
- 最终 checkpoint：是
- 预测结果：是
- git commit：是
- best-test eval 输出：`/data4/ynz/cdpl_runs/step1_baseline_vs_cdpl_calib/best_test_eval/ubt_baseline_1gpu_seed1_iter20999`
- 实验目录：`/data4/ynz/cdpl_runs/step1_baseline_vs_cdpl_calib/ubt_baseline_1gpu_seed1`

### cdpl_calib_1gpu_seed1

- 配置文件：是
- 启动命令：是
- 训练日志：是
- 最终 checkpoint：是
- 预测结果：是
- git commit：是
- best-test eval 输出：`/data4/ynz/cdpl_runs/step1_baseline_vs_cdpl_calib/best_test_eval/cdpl_calib_1gpu_seed1_iter11999`
- 实验目录：`/data4/ynz/cdpl_runs/step1_baseline_vs_cdpl_calib/cdpl_calib_1gpu_seed1`

## 分类别 AP

| 类别 | ubt_baseline_1gpu_seed1 | cdpl_calib_1gpu_seed1 |
| --- | ---: | ---: |
| D-PED | 4.742 | 4.139 |
| D-SHM | 5.896 | 5.948 |
| ERM | 14.088 | 15.949 |
| F-PED | 5.723 | 3.478 |
| IRF | 13.745 | 14.822 |
| MH | 10.086 | 12.457 |
| RS | 10.374 | 3.016 |
| S-PED | 15.181 | 12.015 |
| SRF | 15.481 | 13.762 |

## 结论判断

按 test 最优 checkpoint 选择后，CDPL 类校准使 mAP 下降 `1.081`，AP75 下降 `1.262`，AP30 下降 `2.660`；AP_small 小幅上升 `0.145`。
该重复实验仍支持正式 seed0 的总体结论：只做类校准没有带来稳定主指标收益。它应作为补充证据使用，而不是作为独立正向结果。
