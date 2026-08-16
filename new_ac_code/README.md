# ThyroidXL 任务 A + C 代码工程（级联 cascade，三分类）

**级联设计**：任务 A 的分割结果作为任务 C 的输入 ——
**U-Net 分割结节 → 用预测 mask 外接框裁剪结节 → CNN 三分类器判定 良性 / 恶性乳头状 / 恶性非乳头状**。

- **任务 A 结节分割**：输出二值 mask，指标 Dice / IoU / HD95
- **任务 C 三分类（良恶性 + 乳头状亚型）**：输入裁剪出的结节 patch，输出
  `0=良性 / 1=恶性乳头状 / 2=恶性非乳头状`，指标 Macro-F1 / Balanced Accuracy / Accuracy + 混淆矩阵


## 目录结构（本文件夹 `ac_code/`）

```
ac_code/
├─ config.py                 # 全局配置（数据根目录、超参、归一化、裁剪/分类参数）
├─ requirements.txt          # 依赖
├─ README.md
├─ train.py                  # 两阶段训练：先 seg 再 cls（--stage seg/cls/both）
├─ evaluate.py               # 评估：seg 指标 + 完整级联三分类指标（患者级）
├─ infer.py                  # 推理：分割 -> 裁剪 -> 三分类，输出 mask 与 cls csv
├─ data/
│  ├─ thyroid_dataset.py     # 数据集（Seg / Patch）、letterbox、患者级划分
│  ├─ crop_utils.py          # mask -> 外接框 -> 裁剪结节 patch（train/eval/infer 共用）
│  └─ build_cls_labels.py    # 由 annotations 生成默认 cls_labels.csv（三分类）
└─ models/
   ├─ unet.py                # 任务 A 分割 SegUNet
   ├─ classifier.py          # 任务 C 三分类器（timm EfficientNet 或内置 CNN，num_classes=3）
   ├─ losses.py              # Dice+BCE / 三分类 CrossEntropy（带 class_weight）
   └─ metrics.py             # Dice/IoU/HD95；三分类 Macro-F1/Bal-Acc/Acc + 混淆矩阵
```

数据根目录默认取本工程上级目录（即 `D:\study\ge\train`，里面含 `images/ masks/ train_annotations.json`）。
可用环境变量 `THYROID_DATA_ROOT` 或命令行 `--data_root` 覆盖。

## 三分类标签（任务 C）

代码读取 `数据根目录/cls_labels.csv`，格式：

```
patient_id,cls
00000058,1
00000370,0
...
```

- `cls=0`：良性（benign）
- `cls=1`：恶性-乳头状（malignant papillary，组织学确诊乳头状）
- `cls=2`：恶性-非乳头状（malignant non-papillary）
- `cls=-1`：不确定，训练/评估时排除

该文件**缺失时会自动**由 `train_annotations.json` + `labels_orig` 生成默认标签（**口径 B**）：
- 良性 = `labels_orig` 标为 `lanhtinh`；
- 恶性 = `labels_orig` 标为 `actinh`，其中组织学含 "papillary thyroid" → 类 1，其余（含无组织学记录）→ 类 2；
- 标良性但组织学为乳头状者 → 归为类 1（乳头状即恶性，优先）；
- 既非良性也非恶性的患者 → 排除(-1)。

**后续别人区分好的标签直接覆盖 `cls_labels.csv` 即可，代码自动识别，无需改代码。**

> ⚠️ 标签口径说明：数据集中“恶性-非乳头状”的可靠组织学金标准极少（仅约 10 例有非乳头状恶性记录），
> 其余类 2 来自“恶性(actinh)但未确认乳头状”，可能混入少量真实乳头状但无活检者。这是对现有标注最合理的默认。
> 历史二分类（PTC / 非PTC）脚本 `data/build_ptc_labels.py`、`data/build_ptc_labels_fnac.py` 已不再被工程调用，仅作归档参考。

## 安装依赖

```bash
pip install -r requirements.txt
```

## 训练（两阶段）

```bash
python train.py                 # 先训分割(seg)，再训分类(cls)
python train.py --stage seg     # 只训分割
python train.py --stage cls     # 只训分类（不需要 seg 模型；用 GT mask 裁剪训练）
python train.py --epochs 80 --cls_epochs 40 --patch_size 256 --batch_size 8
```

- 自动按患者划分 train/val（`splits.json`），同一患者不跨集（防泄漏）。
- **阶段1(seg)**：分割 U-Net，保存 `runs/seg_best.pth` / `seg_last.pth`。
- **阶段2(cls)**：用 GT mask 外接框（扩展 `margin`）裁剪结节 patch，训练三分类器（softmax + 带 class_weight 的 CrossEntropy）；
  保存 `runs/cls_best.pth` / `cls_last.pth`。训练期分类用 GT crop（最干净），推理/评估用预测 mask crop。

## 评估

```bash
python evaluate.py              # 加载 best 模型，输出 任务A 与 任务C 报告
```

任务 C 会同时报告两种裁剪来源：
- `crop_from=pred`：用**预测** mask 裁剪（与推理一致，反映真实级联性能）
- `crop_from=gt`  ：用 **GT** mask 裁剪（性能上限参考）

并输出每类 Precision/Recall/F1 与混淆矩阵（行=真值，列=预测）。

## 推理

```bash
python infer.py                       # 用 runs/seg_best.pth + cls_best.pth
python infer.py --ckpt_seg runs/seg_best.pth --ckpt_cls runs/cls_best.pth --out predictions
```

输出：
- `predictions/masks/<name>.png`：预测 mask（原始尺寸）
- `predictions/cls_preds.csv`：`patient_id, image, p_benign, p_mal_pap, p_mal_nonpap, cls_pred`


## 关键约定（来自数据探查）

- 图像真 RGB、尺寸不一 → letterbox 保持比例后 resize 到 512×512。
- mask 灰度 {0,255}，前景=255；结节面积占比均值 ~7.8%（小目标，强不平衡）→ 分割用 Dice+BCE。
- 文件名 `{患者ID}_{检查ID}_{图序}.png`，前缀=患者ID，用于患者级划分。
- 任务 C 标签是**患者级**：同一患者多张图共享标签，评估在患者级聚合（图像级概率平均后取 argmax）。
- 默认标签中有 3 例 labels_orig 标良性但病理为乳头状癌的矛盾样本，归为类 1（乳头状恶性）；用自有标签覆盖后可消除。
- 分类器输入是**结节 patch 而非整图**，直接喂整图容易学背景；这是级联的核心价值。
