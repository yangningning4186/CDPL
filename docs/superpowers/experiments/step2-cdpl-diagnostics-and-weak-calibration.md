# Step 2：CDPL 诊断与弱化类校准实验

本文档记录 CDPL Step 2 的正式执行结果。实验目标是解释 Step 1 中 class calibration alone 相比 UBT baseline 的回退，并验证延后/减弱或限制阈值下降能否缓解该问题。

## 实验约束与执行环境

- 执行服务器：`ynz@10.77.110.185`（无网络）。
- 代码目录：`/data2/ynz/CDPL-src`，Step 2 运行提交：`07d938028dca28fdeaf471df44ee6a05ede23c88`。
- 输出目录：`/data2/ynz/cdpl_runs/step2_cdpl_diagnostics_and_weak_calib`。
- Step 1 输入目录：`/data2/ynz/cdpl_runs/step1_baseline_vs_cdpl_calib`。
- 数据路径：`/data2/ynz/cdpl_inputs/annotations/train_labeled.json`、`/data2/ynz/OCT_SS/datasets/annotations/train_unlabeled_v3_1.json`、`/data2/ynz/OCT_SS/datasets/annotations/test_v3_1.json`、`/data2/ynz/OCT_SS/datasets/images`。
- Conda 环境：`oct_ss`。`ubtv2_oct` 因 Detectron2 缺少 `FastRCNNOutputs` 不兼容，未用于正式运行。
- 运行未使用任何 `OCT_SS-CDPL` 目录；最终运行元数据和受检日志中没有该路径或旧 `/data4` 路径引用。

## 验证流程

1. 在 GPU7 上用 `oct_ss` 重跑 5-image smoke。输出为 `diagnostics/smoke_mapping_oct_ss_gpu7.json`：`teacher_boxes=487`、`ubt_kept=27`、`cdpl_kept=54`，threshold keys 包含 `F-PED` 且没有 numeric key。
2. 对 Step 1 四个 checkpoint 运行完整 unlabeled set 诊断，每份均覆盖 `13541` 张图，映射检查均通过。`cdpl_minus_ubt` 分别为：CDPL seed0 `+10009`、CDPL seed1 `+11193`、UBT seed0 `+6405`、UBT seed1 `+5806`。
3. 复用 Step 1 baseline seed0 的 24k final 与 periodic-test best 作为 reference，并对 best checkpoint 单独补算 AP30。
4. 完成两组单卡 `MAX_ITER=24000` 训练：`late_weak_1gpu_seed0` 使用 GPU3，`floor_clamp_1gpu_seed0` 使用 GPU4；二者 `exit_code.txt=0`、`complete.ok` 与 `model_final.pth` 均存在。
5. 训练结束后，在确认空闲的 GPU3 上执行 final/best eval-only 与缺失 diagnostics。`step2_eval_summary.json`、`step2_results_generated.md` 和 `postprocess_step2.ok` 于 `2026-05-26 08:18:33 CST` 生成。原计划等待 GPU5 的 watcher 已在发现 GPU5 被外部任务占用后停止，未与 GPU3 汇总重复运行。

## 改进实验设置

| Run | 目的 | 关键设置 |
| --- | --- | --- |
| `late_weak_1gpu_seed0` | 延后并减弱校准注入 | `CDPL_ALPHA_TAIL=0.05`，`CDPL_START_ITER=18000`，`CDPL_RAMP_ITERS=6000`，`CDPL_MIN_CLS_THRESHOLD=0.35` |
| `floor_clamp_1gpu_seed0` | 限制阈值下降幅度 | `CDPL_ALPHA_TAIL=0.15`，`CDPL_MIN_CLS_THRESHOLD=0.45`，`CDPL_MAX_THRESHOLD_OFFSET=0.05` |

## 评估口径

- `final`：对 `model_final.pth` 单独执行 eval-only，并从预测结果计算 mAP、AP30、AP50、AP75、AP_small 和每类 AP。
- `best_test`：选择 periodic test `bbox/AP` 最大的 checkpoint，不使用 val；随后单独 eval-only 并补算同一组指标。
- Baseline reference 的训练提交为 Step 1 提交 `8dd446c16d055c035d405844b77a1d9c78937548`；Step 2 评估代码及两组改进实验均使用提交 `07d938028dca28fdeaf471df44ee6a05ede23c88`。

## 主指标结果

| Run | Checkpoint | Iter | mAP | AP30 | AP50 | AP75 | AP_small |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| UBT baseline reference | final | 23999 | 9.443 | 42.262 | 24.937 | 5.970 | 3.905 |
| UBT baseline reference | best_test | 17999 | 10.254 | 42.300 | 27.513 | 6.521 | 4.451 |
| `late_weak` | final | final | 9.529 | 41.169 | 25.422 | 5.716 | 3.802 |
| `late_weak` | best_test | 11999 | **10.454** | 41.732 | **27.524** | 6.644 | 3.961 |
| `floor_clamp` | final | final | 9.011 | 39.084 | 23.316 | 6.274 | 3.865 |
| `floor_clamp` | best_test | 17999 | 10.283 | 42.058 | 26.028 | **6.712** | **4.652** |

相对 baseline best，`late_weak` best 的 mAP 为 `+0.200`、AP50 为 `+0.010`、AP75 为 `+0.123`，但 AP30 为 `-0.568`、AP_small 为 `-0.490`。`floor_clamp` best 的 mAP 为 `+0.029`、AP75 为 `+0.191`、AP_small 为 `+0.201`，但 AP30 为 `-0.242`、AP50 为 `-1.486`。两组 final 均低于各自较早的 best checkpoint。

## Best-Test 每类 AP

| Class | Baseline best | `late_weak` best | `floor_clamp` best |
| --- | ---: | ---: | ---: |
| D-PED | 4.554 | 4.171 | 5.403 |
| D-SHM | 6.134 | 6.542 | 6.028 |
| ERM | 14.516 | 15.142 | 15.520 |
| F-PED | 6.532 | 5.206 | 4.639 |
| IRF | 14.330 | 14.901 | 13.735 |
| MH | 12.727 | 12.304 | 12.034 |
| RS | 5.197 | 7.436 | 10.082 |
| S-PED | 12.378 | 13.901 | 10.553 |
| SRF | 15.915 | 14.483 | 14.554 |

完整 final 与 best-test 每类 AP 已保存在 `step2_eval_summary.json` 和自动生成的 `step2_results_generated.md`。

## Pseudo-Label 诊断

所有正式诊断覆盖 `13541` 张 unlabeled 图像，threshold mapping 均含 `F-PED` 且不含 numeric key。

| Run / checkpoint policy | Threshold 概况 | `cdpl_minus_ubt` total | F-PED | RS | MH |
| --- | --- | ---: | ---: | ---: | ---: |
| `late_weak` iter11999 actual policy | 全类 `0.5`，校准尚未启用 | 0 | 0 | 0 | 0 |
| `floor_clamp` iter11999 actual policy | 除 ERM=`0.5` 外均 `0.45` | +2113 | +145 | +60 | +37 |
| `late_weak` final policy | tail threshold 约 `0.451` 至 `0.468`，ERM=`0.5` | +2430 | +188 | +229 | +27 |
| `floor_clamp` final policy | 除 ERM=`0.5` 外均 `0.45` | +4529 | +282 | +239 | +39 |

`late_weak` 的 best checkpoint 是 iter11999，早于 `CDPL_START_ITER=18000`，所以机制解释采用 `pseudo_label_diagnostics_iter11999_actual_policy.json`：此时没有额外引入 pseudo boxes。`pseudo_label_diagnostics_best_test.json` 则是用最终阈值对 best 模型进行的补充重算，不代表 iter11999 训练时实际启用的策略。

## 结论

- Step 1 诊断确认 class calibration 会显著放宽伪标签保留量，CDPL 两个 seed 相比 UBT threshold 分别多保留 `10009` 和 `11193` 个 boxes；这为 Step 1 回退提供了直接机制证据。
- `late_weak` 的最佳 mAP 出现在校准尚未激活时，激活后的 final mAP 降至 `9.529`。因此本次实验不支持在训练后段继续注入弱化 class calibration 能带来稳定收益。
- `floor_clamp` 在 iter17999 取得轻微 mAP 提升，并改善 AP75、AP_small 与 `RS`，说明限制阈值下降比原 class calibration 更稳；但 AP50、`F-PED` 与 `MH` 未恢复，final 也回落，收益不足以作为稳定方案。
- 下一阶段若继续推进 Full CDPL，应优先验证 localization-aware 筛选或更严格的持续抑制机制，而不是仅扩大 class-based pseudo-label 接收范围。

## 产物与审计结论

- 汇总产物：`/data2/ynz/cdpl_runs/step2_cdpl_diagnostics_and_weak_calib/step2_eval_summary.json`、`step2_results_generated.md`、`postprocess_step2.ok`。
- 诊断产物：`diagnostics/smoke_mapping_oct_ss_gpu7.json`、四份 Step 1 全量诊断、两组 Step 2 `pseudo_label_diagnostics_final.json`、`pseudo_label_diagnostics_best_test.json` 及 iter11999 actual-policy 诊断。
- 训练产物：两组 Step 2 run 的配置、命令、提交记录、日志、checkpoint、final eval 与 best-test eval 均存在，运行提交和退出码已核验。
- 审计结论：Step 2 请求的 smoke、Step 1 diagnostics、baseline reference、两组 24k 实验、final/best AP30/AP50/AP75/AP_small/per-class AP、中文结论与路径约束核验均已完成。
