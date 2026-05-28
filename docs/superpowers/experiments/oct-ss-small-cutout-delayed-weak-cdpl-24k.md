# OCT-SS Small-Scale Cutout：Delayed + Weak CDPL Sweep

## 状态

本文档记录 OCT-SS official-style small-scale Cutout 下，Delayed + Weak
CDPL 多参数 sweep 对 UBT baseline 与上一轮强 CDPL 的公平对照。7 个 CDPL
变体训练、final/best eval-only、actual-policy pseudo-label diagnostics、
产物审计与 watcher 后处理均已完成。

截至 `2026-05-28 23:55:51 CST` 复核：

- 服务器：`ynz@10.77.110.185`。
- 远端代码目录：`/data2/ynz/CDPL-src`。
- 远端 HEAD：`928e1ad2bab03faf34328c1c96f25edd15c598fe`，工作树 clean。
- UBT baseline 训练 provenance：`3555268ad84e2489eeef9ca6513a99d255d2ef32`。
- 本轮 CDPL sweep 训练 provenance：`928e1ad2bab03faf34328c1c96f25edd15c598fe`。
- 输出根目录文本扫描：`OCT_SS-CDPL` 引用数为 `0`。
- `delayed_weak_sweep_artifact_audit.json`：`passed=true`、`failures=[]`、184
  项检查通过。
- `postprocess.ok` 与 `postprocess_watcher.ok`：`2026-05-28 11:46:00 CST`。

## 路径约束

- 新输出根目录：
  `/data2/ynz/cdpl_runs/step2_cdpl_diagnostics_and_weak_calib/oct_ss_small_cutout_delayed_weak_cdpl_24k`。
- 上一轮只读 UBT baseline：
  `/data2/ynz/cdpl_runs/step2_cdpl_diagnostics_and_weak_calib/oct_ss_small_cutout_ubt_vs_cdpl_24k/ubt_repro_24000iter_1gpu_seed0`。
- 数据路径：`/data2/ynz/cdpl_inputs/annotations/train_labeled.json`、
  `/data2/ynz/OCT_SS/datasets/annotations/train_unlabeled_v3_1.json`、
  `/data2/ynz/OCT_SS/datasets/annotations/test_v3_1.json`。
- 图像路径：`/data2/ynz/OCT_SS/datasets/images`。
- Conda 环境：`oct_ss`。
- 禁止使用任何 `OCT_SS-CDPL` 目录。

本轮没有覆盖上一轮 `oct_ss_small_cutout_ubt_vs_cdpl_24k` 输出目录。后处理仅把
上一轮 UBT baseline 复制/链接到新 sweep 目录作为只读对照。

## 固定条件

- Config：`configs/oct_ss_ubt_repro.yaml`。
- `SOLVER.MAX_ITER=24000`。
- Seed：`0`。
- 单卡训练。
- labeled/unlabeled batch：`4+4`。
- FocalLoss。
- official-style small-scale Cutout：
  `(0.7,(0.005,0.01),(0.3,3.3))`、`(0.5,(0.002,0.01),(0.1,6))`、
  `(0.3,(0.002,0.01),(0.05,8))`。
- `BBOX_THRESHOLD=0.5`、`BURN_UP_STEP=5000`。
- `CDPL_TAU_BASE=0.5`、`CDPL_MAX_THRESHOLD_OFFSET=-1.0`。

## 参数矩阵

| Run | GPU | 完成时间 CST | `CDPL_START_ITER` | `CDPL_RAMP_ITERS` | `CDPL_MIN_CLS_THRESHOLD` | `CDPL_ALPHA_TAIL` |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| `cdpl_dw_s10000_r4000_min045_a005_24000iter_1gpu_seed0` | 1 | `2026-05-28 09:33:36` | 10000 | 4000 | 0.45 | 0.05 |
| `cdpl_dw_s10000_r4000_min045_a010_24000iter_1gpu_seed0` | 2 | `2026-05-28 09:27:05` | 10000 | 4000 | 0.45 | 0.10 |
| `cdpl_dw_s10000_r4000_min040_a005_24000iter_1gpu_seed0` | 3 | `2026-05-28 09:11:00` | 10000 | 4000 | 0.40 | 0.05 |
| `cdpl_dw_s12000_r4000_min045_a005_24000iter_1gpu_seed0` | 4 | `2026-05-28 09:13:10` | 12000 | 4000 | 0.45 | 0.05 |
| `cdpl_dw_s8000_r4000_min045_a005_24000iter_1gpu_seed0` | 5 | `2026-05-28 09:06:05` | 8000 | 4000 | 0.45 | 0.05 |
| `cdpl_dw_s10000_r8000_min045_a005_24000iter_1gpu_seed0` | 6 | `2026-05-28 09:06:38` | 10000 | 8000 | 0.45 | 0.05 |
| `cdpl_dw_s10000_r4000_min048_a005_24000iter_1gpu_seed0` | 7 | `2026-05-28 09:11:39` | 10000 | 4000 | 0.48 | 0.05 |

所有变体均保留 `complete.ok`、`exit_code.txt=0`、`model_final.pth`、周期
checkpoint、`metrics.json`、启动命令、配置和日志。错误扫描包含 `Traceback`、
`CUDA out of memory`、`OutOfMemory`、`Exception during training`、`RuntimeError`
与 `Killed`，复核时未发现异常。

## 启动与后处理

- 多参数启动脚本：`scripts/experiments/launch_oct_ss_delayed_weak_sweep_24k.sh`。
- 单 run 训练入口：`scripts/experiments/run_oct_ss_repro_24k_run.sh`。
- 后处理脚本：`scripts/experiments/postprocess_oct_ss_delayed_weak_sweep_24k.sh`。
- 后处理 watcher：`scripts/experiments/launch_oct_ss_delayed_weak_sweep_postprocess_watcher.sh`。
- 审计脚本：`scripts/experiments/audit_oct_ss_delayed_weak_sweep_results.py`。

后处理对上一轮 UBT baseline 与 7 个 CDPL 变体执行同口径 final/best eval-only。
`best_test` 选择口径为 periodic test `bbox/AP` 最大的 checkpoint，然后单独
eval-only 并从预测 JSON 补算 `mAP`、`AP30`、`AP50`、`AP75`、`AP_small` 与九类
per-class AP。CDPL diagnostics 按对应 checkpoint 的实际策略计算 ramp factor：
处于 `CDPL_START_ITER` 之前或 ramp 中的 checkpoint 不会套用 full-strength 阈值。

## 主结果：Best-Test 口径

本实验以后续可复现的 `best_test` checkpoint 作为结果口径：先用训练期间 periodic
test 的 `bbox/AP` 选择最佳 checkpoint，再单独执行 eval-only 并补算 AP30/AP50/AP75/
AP_small 与九类 AP。`model_final.pth` 不作为本轮方法效果的主结果，只保留为训练
稳定性审计。

| Run | best iter | mAP | AP30 | AP50 | AP75 | APs | mAP vs UBT | AP30 vs UBT | AP50 vs UBT | AP75 vs UBT | APs vs UBT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| UBT | 9999 | 10.493 | 40.823 | 27.846 | 6.224 | 4.259 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| s10000_r4000_min045_a005 | 8499 | 10.769 | 42.184 | 28.107 | 5.870 | 5.170 | +0.276 | +1.361 | +0.262 | -0.354 | +0.911 |
| s10000_r4000_min045_a010 | 10499 | 10.136 | 42.204 | 27.729 | 5.113 | 4.939 | -0.356 | +1.382 | -0.116 | -1.111 | +0.680 |
| s10000_r4000_min040_a005 | 10499 | 10.054 | 42.034 | 26.202 | 5.296 | 4.482 | -0.439 | +1.211 | -1.643 | -0.928 | +0.223 |
| s12000_r4000_min045_a005 | 12999 | 10.525 | 42.828 | 28.831 | 5.459 | 4.393 | +0.032 | +2.005 | +0.985 | -0.765 | +0.134 |
| s8000_r4000_min045_a005 | 11499 | 10.584 | 41.439 | 28.095 | 5.825 | 4.369 | +0.092 | +0.616 | +0.250 | -0.399 | +0.110 |
| s10000_r8000_min045_a005 | 9499 | 10.382 | 41.737 | 27.703 | 5.528 | 4.757 | -0.111 | +0.914 | -0.143 | -0.696 | +0.498 |
| s10000_r4000_min048_a005 | 15499 | 10.685 | 44.012 | 28.544 | 5.983 | 4.429 | +0.193 | +3.189 | +0.698 | -0.241 | +0.170 |

按 best_test 口径，本轮最佳结果是 `s10000_r4000_min045_a005`：

- mAP `10.769`，相对 UBT `+0.276`。
- AP30 `42.184`，相对 UBT `+1.361`。
- AP50 `28.107`，相对 UBT `+0.262`。
- AP_small `5.170`，相对 UBT `+0.911`。

Final checkpoint 仅作为稳定性审计：

| Run | final mAP | final AP30 | final AP50 | final AP75 | final APs | final mAP vs UBT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| UBT | 9.384 | 42.663 | 24.350 | 5.393 | 3.343 | 0.000 |
| s10000_r4000_min045_a005 | 9.141 | 39.759 | 26.034 | 5.051 | 3.826 | -0.243 |
| s10000_r4000_min045_a010 | 8.505 | 39.860 | 23.800 | 4.475 | 4.025 | -0.879 |
| s10000_r4000_min040_a005 | 8.362 | 37.516 | 22.220 | 4.720 | 4.286 | -1.022 |
| s12000_r4000_min045_a005 | 9.168 | 40.644 | 25.073 | 5.584 | 3.789 | -0.216 |
| s8000_r4000_min045_a005 | 9.027 | 39.625 | 23.886 | 5.179 | 3.774 | -0.357 |
| s10000_r8000_min045_a005 | 9.265 | 41.067 | 24.281 | 6.067 | 4.282 | -0.119 |
| s10000_r4000_min048_a005 | 8.810 | 39.853 | 25.216 | 4.441 | 4.233 | -0.574 |

## 九类 AP

Final checkpoint 每类 AP（稳定性审计，不作为主结果）：

| Class | UBT | s10000_r4000_min045_a005 | s10000_r4000_min045_a010 | s10000_r4000_min040_a005 | s12000_r4000_min045_a005 | s8000_r4000_min045_a005 | s10000_r8000_min045_a005 | s10000_r4000_min048_a005 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| F-PED | 3.995 | 5.652 | 4.836 | 4.376 | 4.701 | 4.381 | 5.195 | 4.675 |
| S-PED | 13.859 | 9.953 | 11.103 | 9.794 | 12.656 | 12.034 | 13.766 | 11.609 |
| D-PED | 4.482 | 3.257 | 4.075 | 4.000 | 3.433 | 3.521 | 4.429 | 4.315 |
| SRF | 12.050 | 14.173 | 11.166 | 14.944 | 13.163 | 15.336 | 13.808 | 14.289 |
| IRF | 14.625 | 13.551 | 12.865 | 11.671 | 13.029 | 13.361 | 14.474 | 12.338 |
| RS | 5.065 | 9.267 | 6.137 | 2.069 | 7.814 | 6.008 | 3.052 | 4.398 |
| MH | 10.737 | 9.838 | 9.748 | 8.947 | 9.002 | 9.268 | 11.316 | 9.810 |
| ERM | 13.975 | 13.397 | 13.034 | 13.421 | 13.467 | 13.581 | 13.320 | 13.401 |
| D-SHM | 5.671 | 3.184 | 3.580 | 6.039 | 5.253 | 3.752 | 4.028 | 4.454 |

Best-test checkpoint 每类 AP（主结果口径）：

| Class | UBT | s10000_r4000_min045_a005 | s10000_r4000_min045_a010 | s10000_r4000_min040_a005 | s12000_r4000_min045_a005 | s8000_r4000_min045_a005 | s10000_r8000_min045_a005 | s10000_r4000_min048_a005 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| F-PED | 3.512 | 2.969 | 4.230 | 4.721 | 5.065 | 4.229 | 5.297 | 5.018 |
| S-PED | 14.482 | 10.908 | 12.445 | 12.542 | 10.273 | 12.662 | 13.917 | 13.349 |
| D-PED | 4.523 | 5.667 | 5.507 | 5.301 | 5.093 | 4.459 | 5.031 | 4.335 |
| SRF | 14.574 | 13.376 | 11.642 | 15.109 | 14.661 | 13.022 | 12.055 | 13.401 |
| IRF | 16.122 | 15.972 | 14.175 | 13.951 | 14.343 | 14.799 | 15.896 | 14.744 |
| RS | 7.321 | 12.698 | 10.879 | 8.285 | 11.923 | 10.871 | 8.827 | 10.128 |
| MH | 12.629 | 15.036 | 11.635 | 10.079 | 12.007 | 15.396 | 11.444 | 14.838 |
| ERM | 15.467 | 14.793 | 14.927 | 14.796 | 15.506 | 14.918 | 15.236 | 14.028 |
| D-SHM | 5.805 | 5.498 | 5.786 | 5.702 | 5.855 | 4.904 | 5.731 | 6.328 |

## Pseudo-Label Diagnostics

Diagnostics 只对 CDPL 变体执行。所有 diagnostics 的 `target_cdpl_thresholds`、
`cdpl_thresholds` 与 `per_class` 都包含完整九类
`F-PED`、`S-PED`、`D-PED`、`SRF`、`IRF`、`RS`、`MH`、`ERM`、`D-SHM`，
并已显式核验没有 numeric key。

| Run | diag | actual ramp | teacher boxes | UBT kept | CDPL kept | CDPL - UBT | F-PED + | RS + | MH + |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| s10000_r4000_min045_a005 | final | 1.000 | 321558 | 53936 | 56176 | +2240 | +159 | +99 | +20 |
| s10000_r4000_min045_a005 | best_test | 0.000 | 388848 | 34441 | 34441 | +0 | +0 | +0 | +0 |
| s10000_r4000_min045_a010 | final | 1.000 | 320293 | 54836 | 57719 | +2883 | +194 | +185 | +19 |
| s10000_r4000_min045_a010 | best_test | 0.125 | 340288 | 38258 | 38494 | +236 | +17 | +6 | +2 |
| s10000_r4000_min040_a005 | final | 1.000 | 345429 | 55394 | 58216 | +2822 | +230 | +725 | +16 |
| s10000_r4000_min040_a005 | best_test | 0.125 | 326042 | 36138 | 36327 | +189 | +16 | +13 | +2 |
| s12000_r4000_min045_a005 | final | 1.000 | 275244 | 50070 | 51904 | +1834 | +242 | +127 | +14 |
| s12000_r4000_min045_a005 | best_test | 0.250 | 264328 | 37251 | 37558 | +307 | +31 | +15 | +4 |
| s8000_r4000_min045_a005 | final | 1.000 | 311730 | 54289 | 56498 | +2209 | +148 | +200 | +17 |
| s8000_r4000_min045_a005 | best_test | 0.875 | 297127 | 37706 | 39018 | +1312 | +114 | +58 | +19 |
| s10000_r8000_min045_a005 | final | 1.000 | 313680 | 51422 | 53609 | +2187 | +147 | +440 | +33 |
| s10000_r8000_min045_a005 | best_test | 0.000 | 326282 | 33480 | 33480 | +0 | +0 | +0 | +0 |
| s10000_r4000_min048_a005 | final | 1.000 | 307086 | 55339 | 56209 | +870 | +63 | +73 | +3 |
| s10000_r4000_min048_a005 | best_test | 1.000 | 288890 | 44063 | 44802 | +739 | +45 | +32 | +2 |

相对上一轮强 CDPL，delayed + weak 配置显著降低了伪标签扰动规模：强 CDPL
final 多保留 `43100` 个 boxes、best_test 多保留 `16796` 个 boxes；本轮各变体
final 仅多保留 `870` 到 `2883` 个 boxes，best_test 为 `0` 到 `1312` 个 boxes。

## 与上一轮强 CDPL 对照

上一轮强 CDPL 使用 `CDPL_ALPHA_TAIL=0.15`、`CDPL_MIN_CLS_THRESHOLD=0.35`、
`CDPL_START_ITER=-1`、`CDPL_RAMP_ITERS=0`，训练提交为
`3555268ad84e2489eeef9ca6513a99d255d2ef32`。按 best_test 主结果口径，其结果为：

| Run | best mAP | best AP30 | best AP50 | best AP75 | best APs |
| --- | ---: | ---: | ---: | ---: | ---: |
| UBT | 10.493 | 40.823 | 27.846 | 6.224 | 4.259 |
| Strong CDPL | 9.467 | 36.233 | 25.190 | 5.431 | 3.901 |
| Strong CDPL - UBT | -1.025 | -4.590 | -2.656 | -0.793 | -0.358 |

本轮 delayed + weak 的主结论按 best_test 口径给出：

- best_test 口径有 4 个变体超过 UBT mAP：主方案 `+0.276`、更晚启用 `+0.032`、
  更早启用 `+0.092`、更高 floor `+0.193`。
- best_test 的 AP30 全部高于 UBT，增量 `+0.616` 到 `+3.189`；AP_small 也全部高于
  UBT，增量 `+0.110` 到 `+0.911`。
- best_test 的 AP75 仍全部低于 UBT，说明较宽松/动态阈值更可能改善低 IoU 或小目标
  召回，而不是高质量定位。
- final checkpoint 只作为稳定性审计：所有变体 final mAP 仍低于 UBT final，但这不作为
  本轮方法效果排序依据。

## 结论

本轮核心假设得到支持：按 best_test 结果口径，Delayed + Weak CDPL 相比上一轮强
CDPL 明显更优，并且出现了可审计的正增益。本文档不使用最后的 `model_final.pth`
作为主结果；final 只用于说明训练后段稳定性。

最值得保留的候选是主方案 `s10000_r4000_min045_a005`：

- best_test mAP `10.769`，相对 UBT `+0.276`，为本轮最高。
- best_test AP30 `+1.361`、AP50 `+0.262`、AP_small `+0.911`。
- best_test 的 `MH` `+2.407`、`RS` `+5.377`，保留了上一轮观察到的 `MH` 潜在收益，
  同时显著改善 `RS`。

因此本轮可审计结论是：

1. 不采用上一轮强 CDPL 配置。
2. Delayed + Weak 的最佳配置采用 `START=10000/RAMP=4000/MIN=0.45/ALPHA=0.05`。
3. 后续报告和对外比较以 best_test checkpoint 为准；final checkpoint 只作为稳定性审计。
4. 下一步若继续实验，应围绕 best checkpoint 的收益复现和 AP75/定位质量改进推进，
   例如 localization-aware filter 或对 best_test 优势明显的 `MH`/`RS` 做类别特异质量约束；
   不建议继续单纯降低阈值下限。
