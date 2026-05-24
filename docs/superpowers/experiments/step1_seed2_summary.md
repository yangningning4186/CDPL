# Step 1：UBT Baseline vs CDPL 类校准

本文档由服务器实验产物自动汇总并人工补充说明，实验根目录为：

`/data4/ynz/cdpl_runs/step1_baseline_vs_cdpl_calib`

重要范围说明：本步骤只评估当前已经实现的 CDPL 类校准阶段，不是带有定位感知 loss routing 的 Full CDPL。

## 执行说明

- 本文件记录未完成的 seed2 重复实验。
- `ubt_baseline_1gpu_seed2` 在提交 `8dd446c16d05` 上正常完成。
- `cdpl_calib_1gpu_seed2` 在 iter 259 停止；当时 GPU2 / PCI `86:00.0` 报出 `nvidia-smi` `Unknown Error`，没有 Python traceback，也没有最终评估结果。
- `2026-05-24 15:01:06 +0800` 服务器重启后没有继续重启 seed2 CDPL，因为 GPU2 可能参与触发了重启，当时优先目标是保护正式 seed0/seed1 输出。

## 实验结果

| 实验 | 提交 | 退出码 | 完成 | mAP | AP30 | AP50 | AP75 | AP_small |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| ubt_baseline_1gpu_seed2 | 8dd446c16d05 | 0 | 是 | 9.961 | 38.009 | 26.795 | 6.671 | 4.315 |

## 产物检查

### ubt_baseline_1gpu_seed2

- 配置文件：是
- 启动命令：是
- 训练日志：是
- 最终 checkpoint：是
- 预测结果：是
- git commit：是
- 实验目录：`/data4/ynz/cdpl_runs/step1_baseline_vs_cdpl_calib/ubt_baseline_1gpu_seed2`

## 分类别 AP

| 类别 | ubt_baseline_1gpu_seed2 |
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

## 结论判断

seed2 对照是不完整的，不能用于得出 baseline-vs-CDPL 结论。它只能说明 seed2 baseline 产物有效，以及 CDPL 重复实验被硬件不稳定中断。
