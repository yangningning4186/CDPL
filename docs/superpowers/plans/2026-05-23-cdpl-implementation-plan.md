# CDPL 实施计划

> **给 agentic workers 的说明：** 必须使用子技能 `superpowers:subagent-driven-development`（推荐）或 `superpowers:executing-plans`，按任务逐项执行本计划。每一步使用 checkbox（`- [ ]`）语法，方便跟踪进度。

**目标：** 构建一个可复现的 CDPL 研究原型，包括数据集审计、伪标签划分、定位可靠性估计、类别校准、baseline 对比，以及可直接用于论文分析的实验产物。

**架构：** 将 CDPL 拆成“框架无关核心”和“UBT 训练接入”两层。`research/cdpl/` 保存可单测的核心规则、离线分析和实验报告工具；`ubteacher/cdpl/` 保存当前仓库专用的 Detectron2 `Instances` adapter、teacher pseudo box 提取和未标注损失路由。第一阶段先在当前 `ubteacher` 代码上验证；后续迁移到 MMDetection、Soft Teacher 或其他 Detectron2 风格框架时，只重写 adapter，不改 CDPL 核心划分逻辑。

**技术栈：** Python、PyTorch、Detectron2 风格 COCO 标注、NumPy、pandas、matplotlib、pytest、当前 `ubteacher` 配置和模型代码。

**当前 UBT 约束：** `ubteacher/engine/trainer.py` 中未标注伪标签的 `loss_rpn_loc_pseudo` 和 `loss_box_reg_pseudo` 默认被乘以 0。CDPL-UBT 第一版不是简单打开所有 pseudo box regression，而是只对 `P_full` 的定位可靠伪标签恢复回归监督，`P_cls` 继续走 classification-only 路径。

---

## 文件结构

- 创建 `research/cdpl/README.md`：说明 CDPL 原型流程和输入要求。
- 创建 `research/cdpl/config.yaml`：集中管理实验路径和 CDPL 阈值。
- 创建 `research/cdpl/dataset_audit.py`：统计类别频次、目标框面积分布、小病灶阈值，并检查患者级划分。
- 创建 `research/cdpl/pseudo_label_partition.py`：将伪标签划分到 `P_full`、`P_cls`、`P_drop`。
- 创建 `research/cdpl/localization_reliability.py`：计算框一致性指标，例如 mean pairwise IoU 和 normalized variance。
- 创建 `research/cdpl/class_calibration.py`：根据类别频次和伪标签保留统计计算类别自适应阈值。
- 创建 `research/cdpl/evaluate_pseudo_labels.py`：将伪标签与有标注验证集进行比较，计算 precision、recall 和 mean IoU。
- 创建 `research/cdpl/visualize_cdpl_cases.py`：保存高分类置信但低定位可靠性案例，以及尾部类别保留案例。
- 创建 `research/cdpl/training_adapter.py`：将划分后的伪标签转换为分类监督集合和回归监督集合。
- 创建 `ubteacher/cdpl/instances.py`：在 Detectron2 `Instances` 与 CDPL 框架无关 pseudo-label dict 之间转换。
- 创建 `ubteacher/cdpl/loss_routing.py`：为 UBT 训练器提供 `P_full` / `P_cls` 的未标注损失路由辅助函数。
- 创建 `tests/research/test_pseudo_label_partition.py`：测试 CDPL 伪标签划分逻辑。
- 创建 `tests/research/test_localization_reliability.py`：测试定位可靠性指标。
- 创建 `tests/research/test_class_calibration.py`：测试类别校准阈值。
- 创建 `tests/research/test_training_adapter.py`：测试分类-only 与回归-enabled 的伪标签路由。
- 创建 `tests/ubteacher/test_cdpl_instances.py`：测试 UBT `Instances` 转换不丢失 bbox、class、score、loc_score。
- 接入完整训练循环时，参考 `ubteacher/engine/trainer.py`、`ubteacher/modeling/meta_arch/rcnn.py`、`ubteacher/modeling/roi_heads/roi_heads.py` 和 `ubteacher/modeling/roi_heads/fast_rcnn.py`。

## 任务 1：研究原型目录

**文件：**
- 创建：`research/cdpl/README.md`
- 创建：`research/cdpl/config.yaml`

- [ ] **步骤 1：创建原型说明 README**

将以下内容写入 `research/cdpl/README.md`：

```markdown
# CDPL Research Prototype

This folder contains isolated utilities for Calibrated Decoupled Pseudo-Labeling.

## Workflow

1. Audit labeled OCT annotations with `dataset_audit.py`.
2. Generate pseudo labels with a UBT-style teacher model.
3. Compute localization reliability with `localization_reliability.py`.
4. Compute class-calibrated thresholds with `class_calibration.py`.
5. Partition pseudo labels with `pseudo_label_partition.py`.
6. Evaluate pseudo-label quality with `evaluate_pseudo_labels.py`.
7. Produce qualitative figures with `visualize_cdpl_cases.py`.
8. Connect the validated rules to `ubteacher/engine/trainer.py` through UBT-specific adapters.

## Core Sets

- `P_full`: pseudo labels used for classification and regression.
- `P_cls`: pseudo labels used only for classification.
- `P_drop`: pseudo labels discarded from unlabeled training.
```

- [ ] **步骤 2：创建基础配置**

将以下内容写入 `research/cdpl/config.yaml`：

```yaml
paths:
  labeled_annotations: datasets/annotations/test_v3_1.json
  unlabeled_images: datasets/images_unlabeled
  pseudo_labels: outputs/cdpl/pseudo_labels.json
  output_dir: outputs/cdpl

cdpl:
  tau_base: 0.7
  tau_loc: 0.6
  alpha_tail: 0.15
  small_area_percentile: 33
  jitter_times: 8

classes:
  names:
    - F-PED
    - S-PED
    - D-PED
    - SRF
    - IRF
    - RS
    - MH
    - ERM
    - D-SHM
```

- [ ] **步骤 3：验证原型文件存在**

运行：

```bash
test -f research/cdpl/README.md
test -f research/cdpl/config.yaml
```

预期：两个命令都以状态码 `0` 退出。

- [ ] **步骤 4：产物检查点**

运行：

```bash
ls research/cdpl
```

预期输出包含：

```text
README.md
config.yaml
```

当前 workspace 是 git 仓库。完成本任务后可使用以下命令提交本任务：

```bash
git add research/cdpl/README.md research/cdpl/config.yaml
git commit -m "docs: add cdpl research scaffold"
```

## 任务 2：数据集审计脚本

**文件：**
- 创建：`research/cdpl/dataset_audit.py`

- [ ] **步骤 1：添加数据集审计实现**

将以下内容写入 `research/cdpl/dataset_audit.py`：

```python
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def load_coco(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def box_area_xywh(bbox):
    _, _, width, height = bbox
    return float(width) * float(height)


def audit_annotations(coco, small_area_percentile=33):
    categories = {cat["id"]: cat["name"] for cat in coco.get("categories", [])}
    class_counts = Counter()
    areas_by_class = defaultdict(list)
    all_areas = []

    for ann in coco.get("annotations", []):
        category_id = ann["category_id"]
        area = box_area_xywh(ann["bbox"])
        class_counts[category_id] += 1
        areas_by_class[category_id].append(area)
        all_areas.append(area)

    if all_areas:
        small_threshold = float(np.percentile(all_areas, small_area_percentile))
    else:
        small_threshold = 0.0

    per_class = []
    for category_id, name in sorted(categories.items()):
        areas = areas_by_class.get(category_id, [])
        per_class.append(
            {
                "category_id": category_id,
                "name": name,
                "count": int(class_counts.get(category_id, 0)),
                "mean_area": float(np.mean(areas)) if areas else 0.0,
                "median_area": float(np.median(areas)) if areas else 0.0,
            }
        )

    return {
        "num_images": len(coco.get("images", [])),
        "num_annotations": len(coco.get("annotations", [])),
        "small_area_threshold": small_threshold,
        "per_class": per_class,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--small-area-percentile", type=float, default=33)
    args = parser.parse_args()

    coco = load_coco(args.annotations)
    report = audit_annotations(coco, args.small_area_percentile)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
```

- [ ] **步骤 2：运行审计**

运行：

```bash
python research/cdpl/dataset_audit.py \
  --annotations datasets/annotations/test_v3_1.json \
  --output outputs/cdpl/dataset_audit.json
```

预期：生成 `outputs/cdpl/dataset_audit.json`。

- [ ] **步骤 3：检查审计输出**

运行：

```bash
python -m json.tool outputs/cdpl/dataset_audit.json
```

预期：输出包含 `num_images`、`num_annotations`、`small_area_threshold` 和 `per_class`。

## 任务 3：伪标签划分工具

**文件：**
- 创建：`research/cdpl/pseudo_label_partition.py`
- 创建：`tests/research/test_pseudo_label_partition.py`

- [ ] **步骤 1：编写伪标签划分测试**

将以下内容写入 `tests/research/test_pseudo_label_partition.py`：

```python
from research.cdpl.pseudo_label_partition import partition_pseudo_labels


def test_partition_full_cls_and_drop():
    pseudo_labels = [
        {"category_id": 1, "score": 0.9, "loc_score": 0.8},
        {"category_id": 1, "score": 0.9, "loc_score": 0.2},
        {"category_id": 1, "score": 0.3, "loc_score": 0.9},
    ]
    thresholds = {1: 0.7}

    result = partition_pseudo_labels(pseudo_labels, thresholds, tau_loc=0.6)

    assert len(result["P_full"]) == 1
    assert len(result["P_cls"]) == 1
    assert len(result["P_drop"]) == 1
    assert result["P_full"][0]["partition"] == "P_full"
    assert result["P_cls"][0]["partition"] == "P_cls"
    assert result["P_drop"][0]["partition"] == "P_drop"


def test_unknown_class_uses_base_threshold():
    pseudo_labels = [{"category_id": 9, "score": 0.65, "loc_score": 0.8}]

    result = partition_pseudo_labels(
        pseudo_labels,
        class_thresholds={},
        tau_loc=0.6,
        tau_base=0.7,
    )

    assert len(result["P_drop"]) == 1
```

- [ ] **步骤 2：运行测试，确认实现前失败**

运行：

```bash
pytest tests/research/test_pseudo_label_partition.py -q
```

预期：失败，因为 `research.cdpl.pseudo_label_partition` 尚不存在。

- [ ] **步骤 3：实现伪标签划分工具**

将以下内容写入 `research/cdpl/pseudo_label_partition.py`：

```python
def partition_pseudo_labels(pseudo_labels, class_thresholds, tau_loc, tau_base=0.7):
    partitions = {"P_full": [], "P_cls": [], "P_drop": []}

    for label in pseudo_labels:
        category_id = label["category_id"]
        cls_score = float(label["score"])
        loc_score = float(label.get("loc_score", 0.0))
        tau_cls = float(class_thresholds.get(category_id, tau_base))
        enriched = dict(label)

        if cls_score >= tau_cls and loc_score >= tau_loc:
            enriched["partition"] = "P_full"
            partitions["P_full"].append(enriched)
        elif cls_score >= tau_cls and loc_score < tau_loc:
            enriched["partition"] = "P_cls"
            partitions["P_cls"].append(enriched)
        else:
            enriched["partition"] = "P_drop"
            partitions["P_drop"].append(enriched)

    return partitions
```

- [ ] **步骤 4：运行测试，确认通过**

运行：

```bash
pytest tests/research/test_pseudo_label_partition.py -q
```

预期：通过，显示 `2 passed`。

## 任务 4：定位可靠性工具

**文件：**
- 创建：`research/cdpl/localization_reliability.py`
- 创建：`tests/research/test_localization_reliability.py`

- [ ] **步骤 1：编写框 IoU 一致性测试**

将以下内容写入 `tests/research/test_localization_reliability.py`：

```python
import numpy as np

from research.cdpl.localization_reliability import box_iou_xyxy, mean_pairwise_iou


def test_box_iou_xyxy_identical_boxes():
    box = np.array([0.0, 0.0, 10.0, 10.0])
    assert box_iou_xyxy(box, box) == 1.0


def test_mean_pairwise_iou_is_high_for_consistent_boxes():
    boxes = np.array(
        [
            [0.0, 0.0, 10.0, 10.0],
            [1.0, 1.0, 11.0, 11.0],
            [0.0, 0.0, 9.0, 9.0],
        ]
    )
    assert mean_pairwise_iou(boxes) > 0.65


def test_mean_pairwise_iou_is_zero_for_single_box():
    boxes = np.array([[0.0, 0.0, 10.0, 10.0]])
    assert mean_pairwise_iou(boxes) == 0.0
```

- [ ] **步骤 2：运行测试，确认实现前失败**

运行：

```bash
pytest tests/research/test_localization_reliability.py -q
```

预期：失败，因为 `research.cdpl.localization_reliability` 尚不存在。

- [ ] **步骤 3：实现定位可靠性指标**

将以下内容写入 `research/cdpl/localization_reliability.py`：

```python
import numpy as np


def box_iou_xyxy(box_a, box_b):
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    intersection = inter_w * inter_h

    area_a = max(0.0, float(box_a[2]) - float(box_a[0])) * max(
        0.0, float(box_a[3]) - float(box_a[1])
    )
    area_b = max(0.0, float(box_b[2]) - float(box_b[0])) * max(
        0.0, float(box_b[3]) - float(box_b[1])
    )
    union = area_a + area_b - intersection

    if union <= 0.0:
        return 0.0
    return intersection / union


def mean_pairwise_iou(boxes):
    boxes = np.asarray(boxes, dtype=float)
    if len(boxes) < 2:
        return 0.0

    scores = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            scores.append(box_iou_xyxy(boxes[i], boxes[j]))
    return float(np.mean(scores)) if scores else 0.0
```

- [ ] **步骤 4：运行测试，确认通过**

运行：

```bash
pytest tests/research/test_localization_reliability.py -q
```

预期：通过，显示 `3 passed`。

## 任务 5：类别校准工具

**文件：**
- 创建：`research/cdpl/class_calibration.py`
- 创建：`tests/research/test_class_calibration.py`

- [ ] **步骤 1：编写类别阈值测试**

将以下内容写入 `tests/research/test_class_calibration.py`：

```python
from research.cdpl.class_calibration import compute_tail_scores, compute_class_thresholds


def test_tail_scores_are_higher_for_rare_classes():
    counts = {1: 100, 2: 10}
    scores = compute_tail_scores(counts)

    assert scores[2] > scores[1]
    assert 0.0 <= scores[1] <= 1.0
    assert 0.0 <= scores[2] <= 1.0


def test_thresholds_are_lower_for_tail_classes():
    counts = {1: 100, 2: 10}
    thresholds = compute_class_thresholds(counts, tau_base=0.7, alpha_tail=0.15)

    assert thresholds[2] < thresholds[1]
    assert thresholds[2] >= 0.5
```

- [ ] **步骤 2：运行测试，确认实现前失败**

运行：

```bash
pytest tests/research/test_class_calibration.py -q
```

预期：失败，因为 `research.cdpl.class_calibration` 尚不存在。

- [ ] **步骤 3：实现类别校准**

将以下内容写入 `research/cdpl/class_calibration.py`：

```python
def compute_tail_scores(class_counts):
    if not class_counts:
        return {}

    max_count = max(float(count) for count in class_counts.values())
    if max_count <= 0:
        return {category_id: 1.0 for category_id in class_counts}

    scores = {}
    for category_id, count in class_counts.items():
        frequency = float(count) / max_count
        scores[category_id] = max(0.0, min(1.0, 1.0 - frequency))
    return scores


def compute_class_thresholds(class_counts, tau_base=0.7, alpha_tail=0.15, min_threshold=0.5):
    tail_scores = compute_tail_scores(class_counts)
    thresholds = {}

    for category_id, tail_score in tail_scores.items():
        threshold = float(tau_base) - float(alpha_tail) * float(tail_score)
        thresholds[category_id] = max(float(min_threshold), threshold)

    return thresholds
```

- [ ] **步骤 4：运行测试，确认通过**

运行：

```bash
pytest tests/research/test_class_calibration.py -q
```

预期：通过，显示 `2 passed`。

## 任务 6：伪标签质量评估

**文件：**
- 创建：`research/cdpl/evaluate_pseudo_labels.py`

- [ ] **步骤 1：实现验证集评估器**

将以下内容写入 `research/cdpl/evaluate_pseudo_labels.py`：

```python
import argparse
import json
from collections import defaultdict

from research.cdpl.localization_reliability import box_iou_xyxy


def xywh_to_xyxy(box):
    x, y, width, height = box
    return [x, y, x + width, y + height]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_pseudo_labels(gt_coco, pseudo_labels, iou_threshold=0.5):
    gt_by_image_class = defaultdict(list)
    for ann in gt_coco.get("annotations", []):
        key = (ann["image_id"], ann["category_id"])
        gt_by_image_class[key].append(xywh_to_xyxy(ann["bbox"]))

    true_positive = 0
    false_positive = 0
    matched_gt = set()
    ious = []

    for pred_index, pred in enumerate(pseudo_labels):
        key = (pred["image_id"], pred["category_id"])
        pred_box = pred["bbox"]
        gt_boxes = gt_by_image_class.get(key, [])

        best_iou = 0.0
        best_gt_index = None
        for gt_index, gt_box in enumerate(gt_boxes):
            iou = box_iou_xyxy(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_index = gt_index

        ious.append(best_iou)
        match_key = (key, best_gt_index)
        if best_iou >= iou_threshold and best_gt_index is not None and match_key not in matched_gt:
            true_positive += 1
            matched_gt.add(match_key)
        else:
            false_positive += 1

    total_gt = sum(len(items) for items in gt_by_image_class.values())
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, total_gt)
    mean_iou = sum(ious) / max(1, len(ious))

    return {
        "precision": precision,
        "recall": recall,
        "mean_iou": mean_iou,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "total_gt": total_gt,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", required=True)
    parser.add_argument("--pseudo", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    args = parser.parse_args()

    gt_coco = load_json(args.gt)
    pseudo = load_json(args.pseudo)
    if isinstance(pseudo, dict) and "annotations" in pseudo:
        pseudo_labels = pseudo["annotations"]
    else:
        pseudo_labels = pseudo

    report = evaluate_pseudo_labels(gt_coco, pseudo_labels, args.iou_threshold)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
```

- [ ] **步骤 2：运行导入检查**

运行：

```bash
python -m py_compile research/cdpl/evaluate_pseudo_labels.py
```

预期：命令以状态码 `0` 退出。

## 任务 7：训练适配器

**文件：**
- 创建：`research/cdpl/training_adapter.py`
- 创建：`tests/research/test_training_adapter.py`
- 创建：`ubteacher/cdpl/instances.py`
- 创建：`ubteacher/cdpl/loss_routing.py`
- 创建：`tests/ubteacher/test_cdpl_instances.py`
- 参考：`ubteacher/engine/trainer.py`
- 参考：`ubteacher/modeling/meta_arch/rcnn.py`
- 参考：`ubteacher/modeling/roi_heads/roi_heads.py`
- 参考：`ubteacher/modeling/roi_heads/fast_rcnn.py`

- [ ] **步骤 1：编写分类和回归路由测试**

将以下内容写入 `tests/research/test_training_adapter.py`：

```python
from research.cdpl.training_adapter import build_unlabeled_supervision


def test_build_unlabeled_supervision_routes_cls_and_reg_sets():
    partitions = {
        "P_full": [{"id": "full-1"}, {"id": "full-2"}],
        "P_cls": [{"id": "cls-1"}],
        "P_drop": [{"id": "drop-1"}],
    }

    result = build_unlabeled_supervision(partitions)

    assert [item["id"] for item in result["classification"]] == [
        "full-1",
        "full-2",
        "cls-1",
    ]
    assert [item["id"] for item in result["regression"]] == ["full-1", "full-2"]
    assert result["dropped"][0]["id"] == "drop-1"


def test_build_unlabeled_supervision_reports_counts():
    partitions = {
        "P_full": [{"id": "full-1"}],
        "P_cls": [{"id": "cls-1"}, {"id": "cls-2"}],
        "P_drop": [],
    }

    result = build_unlabeled_supervision(partitions)

    assert result["stats"] == {
        "num_p_full": 1,
        "num_p_cls": 2,
        "num_p_drop": 0,
    }
```

- [ ] **步骤 2：运行测试，确认实现前失败**

运行：

```bash
pytest tests/research/test_training_adapter.py -q
```

预期：失败，因为 `research.cdpl.training_adapter` 尚不存在。

- [ ] **步骤 3：实现训练适配器**

将以下内容写入 `research/cdpl/training_adapter.py`：

```python
def build_unlabeled_supervision(partitions):
    p_full = list(partitions.get("P_full", []))
    p_cls = list(partitions.get("P_cls", []))
    p_drop = list(partitions.get("P_drop", []))

    classification = p_full + p_cls
    regression = p_full

    return {
        "classification": classification,
        "regression": regression,
        "dropped": p_drop,
        "stats": {
            "num_p_full": len(p_full),
            "num_p_cls": len(p_cls),
            "num_p_drop": len(p_drop),
        },
    }
```

- [ ] **步骤 4：运行测试，确认通过**

运行：

```bash
pytest tests/research/test_training_adapter.py -q
```

预期：通过，显示 `2 passed`。

- [ ] **步骤 5：记录训练接入参考位置**

运行：

```bash
rg -n "branch ==|pseudo|threshold|scores|loss_box_reg|loss_cls" \
  ubteacher/engine/trainer.py \
  ubteacher/modeling/meta_arch/rcnn.py \
  ubteacher/modeling/roi_heads/roi_heads.py \
  ubteacher/modeling/roi_heads/fast_rcnn.py
```

预期：输出能定位当前 Detectron2 风格代码中伪标签过滤、分类损失和回归损失产生的位置。

- [ ] **步骤 6：训练接入约定**

将 CDPL 接入完整 trainer 时，使用以下接口约定：

```python
from research.cdpl.pseudo_label_partition import partition_pseudo_labels
from research.cdpl.training_adapter import build_unlabeled_supervision


partitions = partition_pseudo_labels(
    pseudo_labels=teacher_pseudo_labels,
    class_thresholds=class_thresholds,
    tau_loc=tau_loc,
    tau_base=tau_base,
)
supervision = build_unlabeled_supervision(partitions)
classification_pseudo_labels = supervision["classification"]
regression_pseudo_labels = supervision["regression"]
partition_stats = supervision["stats"]
```

预期行为：`classification_pseudo_labels` 输入未标注分类损失，`regression_pseudo_labels` 输入未标注框回归损失，`partition_stats` 在训练过程中记录日志。

- [ ] **步骤 7：定义 UBT `Instances` 转换边界**

在 `ubteacher/cdpl/instances.py` 中实现以下边界：

```python
def instances_to_pseudo_labels(instances_per_image):
    """Convert Detectron2 Instances to CDPL pseudo-label dictionaries."""


def pseudo_labels_to_instances(pseudo_labels, image_size, device):
    """Convert CDPL pseudo-label dictionaries back to Detectron2 Instances."""
```

转换必须保留：

```text
bbox / gt_boxes
category_id / gt_classes
score / scores
loc_score
partition
```

预期行为：CDPL 核心代码不直接依赖 Detectron2，只有 `ubteacher/cdpl/` 处理 `Instances`。

- [ ] **步骤 8：定义 UBT 第一版低风险训练接入**

第一版训练接入采用 two-pass loss routing，优先验证算法有效性：

```text
pass 1: P_full union P_cls -> classification pseudo loss, pseudo box regression stays disabled
pass 2: P_full only -> reliable pseudo box regression loss
```

在 `ubteacher/engine/trainer.py` 的 `run_step_full_semisup` 中，替换原有 `process_pseudo_label(..., "roih", "thresholding")` 后的单集合伪标签逻辑：

```text
teacher ROI pseudo boxes
-> compute loc_score
-> partition into P_full / P_cls / P_drop
-> create classification Instances from P_full + P_cls
-> create regression Instances from P_full
-> compute unlabeled cls/objectness losses and reliable regression losses separately
```

保留原始 UBT 行为作为配置开关：

```yaml
SEMISUPNET:
  Trainer: "ubteacher"
  CDPL:
    ENABLED: true
    TAU_LOC: 0.6
    JITTER_TIMES: 8
    REG_LOSS_WEIGHT: 1.0
```

预期行为：`CDPL.ENABLED: false` 时完全回到 UBT baseline；`CDPL.ENABLED: true` 时只有 `P_full` 影响 pseudo box regression。

## 任务 8：实验矩阵

**文件：**
- 创建：`research/cdpl/experiment_matrix.md`

- [ ] **步骤 1：创建实验矩阵**

将以下内容写入 `research/cdpl/experiment_matrix.md`：

```markdown
# CDPL Experiment Matrix

## Main Baselines

| Method | Backbone | Labeled Ratio | Unlabeled Data | Notes |
| --- | --- | --- | --- | --- |
| Supervised Only | R50-FPN | 100% | No | Fully supervised lower bound |
| Soft Teacher | R50-FPN | 100% | Yes | SSOD baseline |
| UBT | R50-FPN | 100% | Yes | Current code path; pseudo box regression disabled |
| UBT + naive pseudo regression | R50-FPN | 100% | Yes | Sanity check; all accepted pseudo boxes enable regression |
| UBT + reliable regression | R50-FPN | 100% | Yes | Only `P_full` enables pseudo box regression |
| UBTv2 | R50-FPN | 100% | Yes | Regression-aware baseline |
| LabelMatch | R50-FPN | 100% | Yes | Class-aware threshold baseline |
| CDPL | R50-FPN | 100% | Yes | Proposed method |

## Label Ratios

| Labeled Samples | Ratio Name |
| --- | --- |
| 50 | low-50 |
| 100 | low-100 |
| 300 | mid-300 |
| 600 | full-600 |

## Ablations

| Method | Pseudo Box Regression | Loc Reliability | Class Calibration |
| --- | --- | --- | --- |
| UBT | Disabled | No | No |
| UBT + naive pseudo regression | All accepted pseudo boxes | No | No |
| UBT + reliable regression | `P_full` only | Yes | No |
| CDPL | `P_full` only | Yes | Yes |

## Required Metrics

- mAP@[.5:.95]
- AP50
- AP75
- AP_small
- AP_tail
- per-class AP
- pseudo-label precision
- pseudo-label recall
- pseudo-label mean IoU
- pseudo-label retention rate per class
```

- [ ] **步骤 2：检查实验矩阵覆盖情况**

运行：

```bash
rg -n "CDPL|UBT|AP75|AP_tail|pseudo-label" research/cdpl/experiment_matrix.md
```

预期：输出包含所有必要实验组和评价指标。

## 任务 9：计划级验证

**文件：**
- 读取：`docs/superpowers/specs/2026-05-23-cdpl-core-innovation-design.md`
- 读取：`docs/superpowers/plans/2026-05-23-cdpl-implementation-plan.md`

- [ ] **步骤 1：运行单元测试**

运行：

```bash
pytest tests/research -q
pytest tests/ubteacher/test_cdpl_instances.py -q
```

预期：所有 CDPL 核心工具测试和 UBT adapter 测试通过。

- [ ] **步骤 2：运行静态导入检查**

运行：

```bash
python -m py_compile \
  research/cdpl/dataset_audit.py \
  research/cdpl/pseudo_label_partition.py \
  research/cdpl/localization_reliability.py \
  research/cdpl/class_calibration.py \
  research/cdpl/evaluate_pseudo_labels.py \
  ubteacher/cdpl/instances.py \
  ubteacher/cdpl/loss_routing.py
```

预期：命令以状态码 `0` 退出。

- [ ] **步骤 3：验证 spec 到 plan 的覆盖关系**

运行：

```bash
rg -n "分类-定位|定位可靠性|类别校准|AP75|tail|P_full|P_cls" \
  docs/superpowers/specs/2026-05-23-cdpl-core-innovation-design.md \
  docs/superpowers/plans/2026-05-23-cdpl-implementation-plan.md
```

预期：spec 和 plan 中都包含 CDPL 核心概念、实验指标和伪标签划分。
