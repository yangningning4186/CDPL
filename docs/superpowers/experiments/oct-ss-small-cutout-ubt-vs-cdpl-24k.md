# OCT-SS Small-Scale Cutout：UBT 与 CDPL 类校准公平对照

## 状态

本文档记录 official-style small-scale Cutout 条件下，UBT baseline 与仅启用
class calibration 的 CDPL 对照实验。训练于 `2026-05-27 00:25 CST` 启动，
目前仍在运行中；在两组 `MAX_ITER=24000` 训练、同口径 final/best eval-only、
pseudo-label diagnostics 与证据审计完成前，不形成效果结论。

截至 `2026-05-27 00:55:24 CST`：

| Run | GPU | 状态 | 最新 iter | 最新 checkpoint | 错误扫描 |
| --- | ---: | --- | ---: | --- | ---: |
| `ubt_repro_24000iter_1gpu_seed0` | 1 | running | 2059 | `model_0001999.pth` | 0 |
| `cdpl_calib_repro_24000iter_1gpu_seed0` | 2 | running | 2099 | `model_0001999.pth` | 0 |

错误扫描包含 `Traceback`、`CUDA out of memory`、`OutOfMemory`、
`Exception during training`、`RuntimeError` 与 `Killed`。GPU3 在本次检查时
空闲约 `24243 MiB`，若运行因临时显存冲突失败，可作为从最近 checkpoint
恢复失败组的候选卡；正常训练不会主动迁移。

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

以上仅用于确认两条训练链路能够推进，不作为最终对比结论。

## 收尾与审计口径

训练完成后按下列口径形成最终证据：

1. 核验两组 `model_final.pth`、周期 checkpoint、`metrics.json`、配置、命令、
   日志、训练提交、`exit_code.txt=0` 与 `complete.ok`。
2. 对两组 `model_final.pth` 和 periodic test `bbox/AP` 最优 checkpoint
   分别执行 eval-only，从预测结果补算 `mAP`、`AP30`、`AP50`、`AP75`、
   `AP_small` 与九类 per-class AP。
3. 对 CDPL final 与实际策略下的 best checkpoint 执行 pseudo-label
   diagnostics，核验类别映射包含 `F-PED`、不存在 numeric key，并汇总
   `cdpl_minus_ubt` 及重点类别额外保留 boxes。
4. 通过专用审计脚本核验结果产物后，再补写下方结果与结论，并完成路径、
   进程及文档证据的最终审计。

专用收尾和结果产物审计代码已推送至提交
`b4c8cc2d103992576838ce0d3d6f21b6ebb330e2`。为保持运行中训练的代码
证据不变，新增文件的部署不得修改训练配置或训练实现。后处理 watcher
仅在两组均存在 `complete.ok` 且有显存足够的空闲 GPU 时启动收尾脚本；
若训练非零退出或 launcher 在完成标记前消失，watcher 将退出并留下日志，
由监控流程先诊断和恢复训练，不会自动重复启动失败运行。

## 最终结果

等待两组训练与同口径后处理完成后填写。

## 结论

等待最终指标、每类 AP 与 pseudo-label diagnostics 全部通过审计后填写。
