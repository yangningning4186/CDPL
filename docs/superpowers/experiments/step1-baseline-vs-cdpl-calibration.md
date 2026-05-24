# Step 1：UBT Baseline vs CDPL 类校准

本文档由服务器实验产物自动汇总并人工补充说明，实验根目录为：

`/data4/ynz/cdpl_runs/step1_baseline_vs_cdpl_calib`

重要范围说明：本步骤只评估当前已经实现的 CDPL 类校准阶段，不是带有定位感知 loss routing 的 Full CDPL。主结果按 test 集 periodic eval 的最高 mAP checkpoint 统计，不使用最终 checkpoint 作为主结果。

## 执行说明

- 正式对比实验：`ubt_baseline_1gpu_seed0` vs `cdpl_calib_1gpu_seed0`，代码提交为 `8dd446c16d05`。
- 训练上限：`MAX_ITER=36000`，`BURN_UP_STEP=9000`，`EVAL_PERIOD=3000`，`CHECKPOINT_PERIOD=3000`。
- 最初按要求尝试了 4 GPU 训练，但 GPU3 报出 `nvidia-smi` `Unknown Error` 后训练挂起。后续 3 GPU 和 2 GPU 的 NCCL 探测也失败，因此正式实验切换到已验证的单卡 fallback：GPU0、batch 8、`BASE_LR=0.005`。
- 服务器在 `2026-05-24 15:01:06 +0800` 重启，导致当时活跃的 CDPL 任务被杀掉。正式 CDPL run 恢复 OCT-SS 数据环境后，从 `model_0017999.pth` 继续训练，并在 iter 35999 正常完成。
- 每 3000 iter 的 test 结果记录在各 run 的 `metrics.json`。本表先按 test `bbox/AP` 选择各自最优 checkpoint，再对该 checkpoint 单独跑 eval-only 生成预测文件并补算 AP30。
- 补充重复实验 `seed1` 已在 GPU1 完成，记录在 `docs/superpowers/experiments/step1_seed1_summary.md`。
- 补充实验 `seed2` 的 baseline 已完成，但 `cdpl_calib_1gpu_seed2` 在 iter 259 停止；当时 GPU2 / PCI `86:00.0` 报出 `nvidia-smi` `Unknown Error`。为避免重启后再次影响正式结果，seed2 CDPL 没有继续重跑。

## 实验结果

| 实验 | 最优 iter | checkpoint | 提交 | mAP | AP30 | AP50 | AP75 | AP_small |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| ubt_baseline_1gpu_seed0 | 17999 | `model_0017999.pth` | 8dd446c16d05 | 10.254 | 42.300 | 27.513 | 6.521 | 4.451 |
| cdpl_calib_1gpu_seed0 | 11999 | `model_0011999.pth` | 8dd446c16d05 | 9.205 | 39.521 | 24.864 | 5.107 | 4.886 |

正式对比差值：mAP `-1.048`，AP30 `-2.779`，AP50 `-2.649`，AP75 `-1.414`，AP_small `+0.435`。

## 产物检查

### ubt_baseline_1gpu_seed0

- 配置文件：是
- 启动命令：是
- 训练日志：是
- 最终 checkpoint：是
- 预测结果：是
- git commit：是
- best-test eval 输出：`/data4/ynz/cdpl_runs/step1_baseline_vs_cdpl_calib/best_test_eval/ubt_baseline_1gpu_seed0_iter17999`
- 实验目录：`/data4/ynz/cdpl_runs/step1_baseline_vs_cdpl_calib/ubt_baseline_1gpu_seed0`

### cdpl_calib_1gpu_seed0

- 配置文件：是
- 启动命令：是
- 训练日志：是
- 最终 checkpoint：是
- 预测结果：是
- git commit：是
- best-test eval 输出：`/data4/ynz/cdpl_runs/step1_baseline_vs_cdpl_calib/best_test_eval/cdpl_calib_1gpu_seed0_iter11999`
- 实验目录：`/data4/ynz/cdpl_runs/step1_baseline_vs_cdpl_calib/cdpl_calib_1gpu_seed0`

## 分类别 AP

| 类别 | ubt_baseline_1gpu_seed0 | cdpl_calib_1gpu_seed0 |
| --- | ---: | ---: |
| D-PED | 4.554 | 3.861 |
| D-SHM | 6.134 | 4.166 |
| ERM | 14.516 | 15.775 |
| F-PED | 6.532 | 4.528 |
| IRF | 14.330 | 15.860 |
| MH | 12.727 | 10.489 |
| RS | 5.197 | 2.143 |
| S-PED | 12.378 | 12.117 |
| SRF | 15.915 | 13.910 |

## 需求审计

- 正式 baseline/CDPL 的配置文件、启动命令、日志、最终 checkpoint、预测结果、git commit、退出码和完成标记均已存在。
- best-test checkpoint 已由 `metrics.json` 中 periodic test `bbox/AP` 选择，baseline 为 iter 17999，CDPL 为 iter 11999。
- AP30 使用 best-test eval 输出中的 `inference/coco_instances_results.json` 和 `/data4/ynz/OCT_SS-main/datasets/annotations/test_v3_1.json` 重新计算。
- 正式 final-checkpoint JSON 汇总保存为 `docs/superpowers/experiments/step1_seed0_summary.json`；正式 best-test JSON 汇总保存为 `docs/superpowers/experiments/step1_best_test_seed0_summary.json`；补充实验汇总保存为 `step1_seed1_summary.*` 和 `step1_seed2_baseline_summary.json`。
- 本文件已经记录硬件 fallback、服务器重启、GPU2/GPU3 故障，以及未完成的 seed2 CDPL run。

## 结论判断

按 test 最优 checkpoint 选择后，CDPL 类校准使 mAP 下降 `1.048`，AP75 下降 `1.414`，AP30 下降 `2.779`；AP_small 小幅上升 `0.435`。
Step 1 对“只做类校准”的总体结果仍然偏负：主指标 mAP/AP30/AP50/AP75 均低于 baseline，虽然 `IRF`、`ERM` 和 AP_small 有小幅改善，但 `RS`、`MH`、`F-PED` 等类别下降明显。

后续实验不应该把类校准视为一个独立有效的增益点。继续投入 Full CDPL 前，建议先按类别和置信度阈值检查 pseudo-label 保留情况，再做更紧的消融：UBT、弱化/后置的类校准、以及加入定位感知 routing 的 Full CDPL。只有当定位感知部分能够解释或扭转这次回退时，Full CDPL 才值得继续作为主线推进。
