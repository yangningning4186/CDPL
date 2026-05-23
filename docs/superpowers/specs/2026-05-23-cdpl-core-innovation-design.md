# CDPL 核心创新设计文档

## 1. 论文定位

拟定方法名称：

**CDPL: Calibrated Decoupled Pseudo-Labeling for Semi-Supervised OCT Lesion Detection**

中文名称：

**校准式分类-定位解耦伪标签学习方法**

本文应定位为一篇偏计算机视觉方法的半监督目标检测论文，而不是单纯的 OCT 应用论文。OCT B-scan 病灶检测提供了一个高挑战场景：病灶目标小、边界模糊、类别长尾明显、人工框标注稀缺。论文核心应围绕半监督检测中的伪标签质量建模展开。

## 2. 核心问题

现有 UBT、Soft Teacher 等 teacher-student 半监督检测方法通常根据 teacher 的分类置信度决定伪标签是否参与训练。但在 OCT 小病灶检测中，伪标签存在三个关键问题：

- 分类置信度高并不代表定位准确，错误框若参与回归训练会污染 student。
- 尾部类别预测分数通常偏低，容易被统一高阈值错误过滤。
- UBT 已是较强 baseline，简单调阈值或叠加小模块难以带来稳定提升。

在当前 `ubteacher` 代码中，未标注伪标签的 box regression 损失默认被整体置零。这说明第一版 CDPL 不应简单表述为“关闭低质量伪框回归”，而应更准确地表述为：

**在 UBT 的 classification-only 伪标签监督基础上，只让定位可靠的伪标签重新进入回归监督。**

因此本文的核心问题定义为：

**如何在强半监督检测 baseline 上，更细粒度地判断每个伪标签应该参与分类监督、回归监督，还是被丢弃？**

## 3. 核心思想

CDPL 不再将伪标签视为一个整体进行 accept/reject，而是将伪标签可靠性拆分为三个判断：

- 分类是否可靠
- 定位是否可靠
- 该类别是否需要校准保护

根据这些判断，每个伪标签被划分到不同监督路径：

- `P_full`：分类可靠且定位可靠，同时参与分类损失和框回归损失
- `P_cls`：分类可靠但定位不可靠，仅参与分类损失
- `P_drop`：分类不可靠，丢弃

这使得 CDPL 能够保留有分类价值但定位不稳定的伪标签，同时避免低质量框参与回归训练。落到当前 UBT 实现时，`P_cls` 对应原始 UBT 的伪标签分类监督路径，`P_full` 对应 CDPL 新增的可靠伪框回归路径。

## 4. 模块一：分类-定位解耦伪标签监督

对 teacher 在未标注图像上生成的每个 pseudo box，分别计算：

- `s_cls`：分类可靠性
- `s_loc`：定位可靠性

给定类别相关分类阈值 `tau_cls(c)` 和定位阈值 `tau_loc`，伪标签分配规则为：

```text
if s_cls >= tau_cls(c) and s_loc >= tau_loc:
    pseudo label -> P_full
elif s_cls >= tau_cls(c) and s_loc < tau_loc:
    pseudo label -> P_cls
else:
    pseudo label -> P_drop
```

该模块的论文价值在于：现有方法通常用分类置信度同时决定分类监督和回归监督，而 CDPL 将二者解耦。对于会使用伪框回归的 baseline，CDPL 显式避免高分类分数、低定位质量的伪框参与回归训练；对于当前 UBT 这种默认不使用伪框回归的保守实现，CDPL 则提供一种有条件恢复回归监督的机制。

## 5. 模块二：定位可靠性估计

`s_loc` 用于衡量一个 pseudo box 是否适合作为 box regression target。

第一版推荐使用 **box jitter consistency**：

1. 对 teacher 预测框进行 `K` 次轻微扰动。
2. 将扰动框送入 box head 或 ROI head。
3. 得到 `K` 个 refined boxes。
4. 计算这些 refined boxes 之间的一致性。
5. 一致性越高，说明定位越稳定，`s_loc` 越高。

主公式建议使用：

```text
s_loc = mean_pairwise_iou(refined_boxes)
```

备选公式为：

```text
s_loc = exp(- normalized_box_variance)
```

主论文优先采用 `mean_pairwise_iou`，因为它更直观，也更容易解释为“扰动后框回归结果是否稳定”。

## 6. 模块三：类别校准的伪标签筛选

为避免尾部类别被统一阈值过滤，CDPL 为每个类别维护动态分类阈值 `tau_cls(c)`。

一个简洁可执行的设计为：

```text
tau_cls(c) = tau_base - alpha * tail_score(c)
```

其中 `tail_score(c)` 可由类别在标注集中的频次或当前伪标签保留率计算。类别越稀有，`tail_score(c)` 越大，分类阈值适当降低。

关键约束：

**尾部类别只放宽分类进入门槛，不放宽定位回归门槛。**

这意味着尾部类别更容易进入 `P_cls` 获得分类监督，但只有在 `s_loc >= tau_loc` 时才进入 `P_full` 参与回归监督。该设计同时服务于尾部类别保留和定位噪声控制。

## 7. 训练流程

CDPL 第一阶段嵌入当前 `ubteacher` 的 teacher-student 框架：

1. 使用有标注 OCT B-scan 图像训练 supervised warmup detector。
2. 初始化 student，并通过 EMA 从 student 更新 teacher。
3. teacher 在未标注图像的 weak view 上生成 pseudo boxes。
4. 对每个 pseudo box 计算 `s_cls` 和 `s_loc`。
5. 根据 `tau_cls(c)` 和 `tau_loc` 将 pseudo boxes 划分为 `P_full`、`P_cls`、`P_drop`。
6. student 在 strong view 未标注图像上训练。
7. `P_full` 用于分类和回归，`P_cls` 仅用于分类，`P_drop` 不参与训练。
8. 重复 EMA 更新和伪标签划分。

在当前仓库中，主要接入点是 `ubteacher/engine/trainer.py` 的 `run_step_full_semisup`：teacher 产出 ROI pseudo boxes 后，替换原有统一阈值过滤逻辑；student 计算未标注损失时，将分类监督集合和回归监督集合分开处理。

当前工程落地先采用“直接 UBT 仓库运行”方式：`train_net.py` 根据 `SEMISUPNET.Trainer: "cdpl"` 选择 `ubteacher/engine/cdpl_trainer.py`，`configs/oct_ss_cdpl.yaml` 绑定 OCT 数据集名，`ubteacher/data/datasets/oct_coco.py` 通过环境变量注册服务器上的 OCT COCO 标注和图像目录。服务器实验目录应为 `/data4/ynz/CDPL-src`，而不是额外复制出的框架目录。

## 8. 损失函数

整体损失：

```text
L = L_sup + lambda_u * L_unsup
```

未标注损失：

```text
L_unsup = L_cls(P_full union P_cls) + beta * L_reg(P_full)
```

核心差异：

```text
P_cls 不参与 L_reg
```

当前 UBT 代码会将 `loss_rpn_loc_pseudo` 和 `loss_box_reg_pseudo` 乘以 0。CDPL-UBT 第一版应改成只对 `P_full` 打开伪标签回归损失，而不是对所有 pseudo boxes 统一打开回归。这样可以保留 UBT 的稳健分类路径，同时检验可靠定位伪框是否能提升 AP75 和 pseudo-label mean IoU。

第一版推荐使用 hard partition，因为它机制清楚、实现直接、消融实验容易解释。后续可扩展为 reliability weighting：

```text
L_unsup = sum s_cls * L_cls + beta * sum s_loc * L_reg
```

但 reliability weighting 应作为增强实验，而不是第一版主方法依赖。

## 9. 实验设计

### 9.1 主结果对比

建议比较：

- Supervised Only
- Soft Teacher
- UBT
- UBTv2
- LabelMatch
- CDPL

主指标：

- mAP@[.5:.95]
- AP50
- AP75
- AP_small
- AP_tail
- 每类 AP

其中 AP75 和 pseudo-label mean IoU 更能体现定位质量改进。

### 9.2 消融实验

建议消融顺序：

```text
UBT baseline
+ cls/reg decoupling
+ loc reliability
+ class calibration
Full CDPL
```

在当前 `ubteacher` 实现中，建议把消融写得更具体：

```text
UBT baseline: P_cls-only pseudo labels, pseudo box regression disabled
UBT + naive pseudo regression: all accepted pseudo boxes enable regression
UBT + reliable regression: only P_full enables regression
UBT + reliable regression + class calibration
Full CDPL
```

该表用于证明每个模块的独立贡献，并回应“UBT 已经强，是否仍有提升”的问题。

### 9.3 定位噪声分析

重点展示：

- pseudo-label mean IoU
- AP75
- box localization error
- 分类高但定位差的伪框案例
- CDPL 将错误框从 `P_full` 降级到 `P_cls` 的可视化

### 9.4 尾部类别分析

重点展示：

- tail-class AP
- 每类别 AP
- 各类别 pseudo-label retention rate
- class calibration 前后的尾部类别伪标签数量变化

### 9.5 标注比例实验

在固定未标注集的情况下，改变标注数据量：

- 10%
- 20%
- 50%
- 100%

或使用绝对数量：

- 50 张标注
- 100 张标注
- 300 张标注
- 600 张标注

该实验用于证明 CDPL 在低标注条件下的优势。

## 10. 与相关方法的关系

UBT 和 Soft Teacher 是 teacher-student 半监督检测框架，CDPL 不应被表述为对某个旧方法的小改动，而应被表述为一种新的伪标签使用策略。

与 UBT 的差异：

- UBT 更关注 teacher-student 训练与类别偏置问题。
- CDPL 关注每个伪标签的分类监督和定位监督是否应当被解耦使用。
- 在当前 UBT 代码实现中，伪标签回归损失默认关闭；CDPL 的第一版实验重点是只为定位可靠的伪标签恢复回归监督。

与 UBTv2 的差异：

- UBTv2 强调回归不确定性和半监督检测中的定位质量。
- CDPL 将定位可靠性进一步转化为伪标签监督路径分配，即 `P_full` 与 `P_cls` 的差异化训练。

与 LabelMatch 的差异：

- LabelMatch 关注类别分布匹配和类别阈值。
- CDPL 保留类别校准思想，但加入定位可靠性约束，使尾部类别只放宽分类监督，不放宽回归监督。

## 11. 论文贡献表述

建议贡献点写成三条：

1. 提出一种分类-定位解耦的伪标签监督策略，使高分类置信但低定位质量的伪框仅提供分类监督，同时允许定位可靠伪框提供回归监督。
2. 设计一种基于 box jitter consistency 的定位可靠性估计方法，用于判断未标注伪框是否适合作为回归监督。
3. 提出类别校准的伪标签筛选机制，在保护尾部病灶类别的同时保持严格的定位质量约束。

## 12. 范围边界

第一版 CDPL 不引入 agent 或大模型作为核心模块。

第一版 CDPL 不依赖 DETR 架构迁移。

RETFound、MAE、DINO 等领域预训练可作为增强实验或附加消融，不作为 CDPL 的核心创新点。

最终推理阶段应保持为普通 detector 推理，不依赖伪标签筛选模块，也不引入额外人工交互。

工程边界上，第一阶段先服务当前 `ubteacher` 训练代码。核心算法保持框架无关，训练接入通过 adapter 完成；后续迁移到 MMDetection、Soft Teacher 或其他 Detectron2 风格框架时，只重写 pseudo label extraction 和 loss routing adapter，不改 CDPL 的划分规则。

## 13. 成功标准

CDPL 应在以下方面体现优势：

- 相比 UBT baseline，总体 mAP 有稳定提升。
- AP75 或 pseudo-label mean IoU 有明显提升，证明定位质量改进。
- tail-class AP 和尾部伪标签保留率提升，证明类别校准有效。
- 消融实验显示解耦监督、定位可靠性、类别校准分别贡献正向增益。

如果总 mAP 提升有限，但 AP75、AP_tail、伪标签定位质量显著提升，论文仍可围绕“定位噪声与尾部类别鲁棒性”组织贡献。
