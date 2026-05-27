# OCT-SS Small-Scale Cutout：UBT 与 CDPL 类校准公平对照

## 状态

本文档记录 official-style small-scale Cutout 条件下，UBT baseline 与仅启用
class calibration 的 CDPL 对照实验。训练于 `2026-05-27 00:25 CST` 启动，
两组 `MAX_ITER=24000` 训练、同口径 final/best eval-only、CDPL
pseudo-label diagnostics 与结果产物审计均已完成。

截至 `2026-05-27 22:31:14 CST` 复核：

| Run | GPU | 状态 | 完成时间 | 退出码 | 最终 checkpoint | 错误扫描 |
| --- | ---: | --- | --- | ---: | --- | ---: |
| `ubt_repro_24000iter_1gpu_seed0` | 1 | complete | `2026-05-27 09:20:13` | 0 | `model_final.pth` | 0 |
| `cdpl_calib_repro_24000iter_1gpu_seed0` | 2 | complete | `2026-05-27 09:16:12` | 0 | `model_final.pth` | 0 |

错误扫描包含 `Traceback`、`CUDA out of memory`、`OutOfMemory`、
`Exception during training`、`RuntimeError` 与 `Killed`。两组训练均保留
`complete.ok`、`exit_code.txt=0`、`model_final.pth`、周期 checkpoint、
`metrics.json`、启动命令、配置和日志。后处理 watcher 于
`2026-05-27 09:46:23 CST` 写入 `postprocess_watcher.ok` 与 `postprocess.ok`。

## 路径约束

- 无网络服务器：`ynz@10.77.110.185`。
- 代码目录：`/data2/ynz/CDPL-src`。
- 实验输出根目录：`/data2/ynz/cdpl_runs/step2_cdpl_diagnostics_and_weak_calib/oct_ss_small_cutout_ubt_vs_cdpl_24k`。
- 标注数据：`/data2/ynz/cdpl_inputs/annotations/train_labeled.json`、`/data2/ynz/OCT_SS/datasets/annotations/train_unlabeled_v3_1.json`、`/data2/ynz/OCT_SS/datasets/annotations/test_v3_1.json`。
- 图像数据：`/data2/ynz/OCT_SS/datasets/images`。
- Conda 环境：`oct_ss`。
- 本实验不得读取或写入任何 `OCT_SS-CDPL` 目录。当前输出产物路径扫描无该目录引用。

## 对照协议

两组实验使用相同的 `configs/oct_ss_ubt_repro.yaml`、seed `0`、单卡训练、
labeled/unlabeled batch `4+4`、`SOLVER.MAX_ITER=24000`、FocalLoss 能力和
数据划分。实验变量仅为是否启用 CDPL class calibration：

| Run | Trainer | `CDPL_ENABLED` | 训练提交 |
| --- | --- | --- | --- |
| `ubt_repro_24000iter_1gpu_seed0` | `ubteacher` | `False` | `3555268ad84e2489eeef9ca6513a99d255d2ef32` |
| `cdpl_calib_repro_24000iter_1gpu_seed0` | `cdpl` | `True` | `3555268ad84e2489eeef9ca6513a99d255d2ef32` |

共同半监督参数：

| 参数 | 值 |
| --- | ---: |
| `BBOX_THRESHOLD` | 0.5 |
| `BURN_UP_STEP` | 5000 |
| `TEACHER_UPDATE_ITER` | 1 |
| `EMA_KEEP_RATE` | 0.9996 |
| `UNSUP_LOSS_WEIGHT` | 2.0 |

CDPL 组参数：

| 参数 | 值 |
| --- | ---: |
| `CDPL_TAU_BASE` | 0.5 |
| `CDPL_ALPHA_TAIL` | 0.15 |
| `CDPL_MIN_CLS_THRESHOLD` | 0.35 |
| `CDPL_MAX_THRESHOLD_OFFSET` | -1.0 |
| `CDPL_START_ITER` | -1 |
| `CDPL_RAMP_ITERS` | 0 |

## 图像增强

复刻的 official-style small-scale Cutout 仅作用于强增强视图，三次
`RandomErasing` 参数如下：

| 次序 | `p` | `scale` | `ratio` | `value` |
| ---: | ---: | --- | --- | --- |
| 1 | 0.7 | `(0.005, 0.01)` | `(0.3, 3.3)` | `random` |
| 2 | 0.5 | `(0.002, 0.01)` | `(0.1, 6)` | `random` |
| 3 | 0.3 | `(0.002, 0.01)` | `(0.05, 8)` | `random` |

本次不实现 Double filter；该因素不进入本对照的变量范围。

## 中途观测

在 `BURN_UP_STEP=5000` 前，teacher `bbox/AP=0` 属预期，不能用于判断
class calibration 的收益。已观察到的 student periodic `bbox/AP` 为：

| Iter | UBT | CDPL |
| ---: | ---: | ---: |
| 499 | 0.438 | 0.428 |
| 999 | 1.643 | 2.129 |
| 1499 | 4.151 | 4.992 |
| 1999 | 6.010 | 5.762 |

以上仅用于确认两条训练链路能够推进，不作为最终对比结论。最终结论以下方
final/best eval-only 与 diagnostics 审计为准。

## 收尾与审计口径

训练完成后已按下列口径形成最终证据：

1. 核验两组 `model_final.pth`、周期 checkpoint、`metrics.json`、配置、命令、
   日志、训练提交、`exit_code.txt=0` 与 `complete.ok`。
2. 对两组 `model_final.pth` 和 periodic test `bbox/AP` 最优 checkpoint
   分别执行 eval-only，从预测结果补算 `mAP`、`AP30`、`AP50`、`AP75`、
   `AP_small` 与九类 per-class AP。
3. 对 CDPL final 与实际策略下的 best checkpoint 执行 pseudo-label
   diagnostics，核验类别映射包含 `F-PED`、不存在 numeric key，并汇总
   `cdpl_minus_ubt` 及重点类别额外保留 boxes。
4. 通过专用审计脚本核验结果产物后，补写结果与结论，并完成路径、进程及
   文档证据的最终审计。

专用收尾和结果产物审计代码已推送至提交
`b4c8cc2d103992576838ce0d3d6f21b6ebb330e2`。为保持运行中训练的代码
证据不变，新增文件的部署不得修改训练配置或训练实现。后处理 watcher
仅在两组均存在 `complete.ok` 且有显存足够的空闲 GPU 时启动收尾脚本；
若训练非零退出或 launcher 在完成标记前消失，watcher 将退出并留下日志，
由监控流程先诊断和恢复训练，不会自动重复启动失败运行。

本次后处理由 watcher 自动触发并通过审计：

- 服务器工作树：`/data2/ynz/CDPL-src`，后处理执行时 HEAD
  `90589c9e001d7811ff9349c65fcd63890a1cf411`，工作树 clean。该提交相对训练
  提交只新增 docs、postprocess/audit/watcher 脚本与测试，不修改训练实现。
- 两组训练产物中的 `git_commit.txt` 均为
  `3555268ad84e2489eeef9ca6513a99d255d2ef32`。
- `small_cutout_result_artifact_audit.json`：`passed=true`、`failures=[]`、
  34 项检查通过；两组 final/best 预测文件、标准指标与九类 AP mapping 均通过。
- 输出根目录文本扫描未发现 `OCT_SS-CDPL` 引用。

## 最终结果

主指标如下。`best_test` 选择口径为 periodic test `bbox/AP` 最大的 checkpoint，
然后单独执行 eval-only 并从预测 JSON 补算 `AP30`、`AP50`、`AP75`、
`AP_small` 与每类 AP。

| Run | Checkpoint | Iter | mAP | AP30 | AP50 | AP75 | AP_small |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| UBT | final | final | 9.384 | 42.663 | 24.350 | 5.393 | 3.343 |
| CDPL | final | final | 6.956 | 30.917 | 19.331 | 3.643 | 2.801 |
| CDPL - UBT | final | final | -2.428 | -11.746 | -5.019 | -1.750 | -0.543 |
| UBT | best_test | 9999 | 10.493 | 40.823 | 27.846 | 6.224 | 4.259 |
| CDPL | best_test | 12999 | 9.467 | 36.233 | 25.190 | 5.431 | 3.901 |
| CDPL - UBT | best_test | - | -1.025 | -4.590 | -2.656 | -0.793 | -0.358 |

每类 AP：

| Class | UBT final | CDPL final | Delta | UBT best | CDPL best | Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| F-PED | 3.995 | 1.458 | -2.537 | 3.512 | 2.609 | -0.903 |
| S-PED | 13.859 | 8.585 | -5.274 | 14.482 | 11.977 | -2.506 |
| D-PED | 4.482 | 1.994 | -2.488 | 4.523 | 4.071 | -0.451 |
| SRF | 12.050 | 8.652 | -3.398 | 14.574 | 13.013 | -1.561 |
| IRF | 14.625 | 12.746 | -1.879 | 16.122 | 15.750 | -0.372 |
| RS | 5.065 | 3.468 | -1.597 | 7.321 | 4.292 | -3.029 |
| MH | 10.737 | 10.453 | -0.284 | 12.629 | 15.042 | +2.413 |
| ERM | 13.975 | 12.533 | -1.442 | 15.467 | 13.789 | -1.678 |
| D-SHM | 5.671 | 2.718 | -2.952 | 5.805 | 4.664 | -1.141 |

CDPL diagnostics 只对 CDPL 组产物执行，类别映射为
`F-PED`、`S-PED`、`D-PED`、`SRF`、`IRF`、`RS`、`MH`、`ERM`、`D-SHM`，
不存在 numeric key。实际 class-calibration 阈值如下：

| Class | Threshold |
| --- | ---: |
| F-PED | 0.364 |
| S-PED | 0.355 |
| D-PED | 0.375 |
| SRF | 0.370 |
| IRF | 0.405 |
| RS | 0.356 |
| MH | 0.353 |
| ERM | 0.500 |
| D-SHM | 0.360 |

伪标签筛选汇总：

| Diagnostic | Checkpoint | Teacher boxes | UBT kept | CDPL kept | CDPL - UBT |
| --- | --- | ---: | ---: | ---: | ---: |
| final | `model_final.pth` | 773459 | 114899 | 157999 | +43100 |
| best_test | `model_0012999.pth` | 486099 | 56695 | 73491 | +16796 |

重点类别额外保留 boxes：

| Diagnostic | Class | UBT kept | CDPL kept | CDPL - UBT |
| --- | --- | ---: | ---: | ---: |
| final | F-PED | 4243 | 7438 | +3195 |
| final | RS | 1686 | 2631 | +945 |
| final | MH | 302 | 521 | +219 |
| best_test | F-PED | 1423 | 2670 | +1247 |
| best_test | RS | 906 | 1399 | +493 |
| best_test | MH | 164 | 265 | +101 |

## 结论

在 OCT-SS official-style small-scale Cutout、seed0、24k、公平 batch 4+4
条件下，本仓库当前 class-calibration 版本的 CDPL 不优于 UBT baseline。

证据上，CDPL 在 final 与 best_test 两个口径均低于 UBT：final mAP 低
`2.428`，best_test mAP 低 `1.025`；AP30、AP50、AP75 与 AP_small 也均为负增益。
每类 AP 中，仅 best_test 的 `MH` 为正增益（`+2.413`），其余类别均不占优，
`RS`、`S-PED`、`SRF`、`ERM` 等类别拉低整体结果。

diagnostics 显示 class calibration 确实按类别降低阈值并额外保留伪标签：
final 多保留 `43100` 个 boxes，best_test 多保留 `16796` 个 boxes；其中
`F-PED`、`RS`、`MH` 均有额外 boxes。但这些额外伪标签没有转化为整体检测收益，
反而伴随 final 和 best_test 指标回退。因此，本次小尺度 Cutout 公平实验的可审计
结论是：不要采用当前 `CDPL_ENABLED=True` / class-calibration 配置作为 OCT-SS
small-scale Cutout 的改进方案；后续若继续探索，应优先收紧弱校准策略或加入
localization-aware / 质量约束，而不是直接放大低阈值伪标签规模。
