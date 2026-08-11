# ThyroidXL 任务 A + C 代码工程（级联 cascade）

**级联设计**：任务 A 的分割结果作为任务 C 的输入 ——
**U-Net 分割结节 → 用预测 mask 外接框裁剪结节 → CNN 分类器判定 PTC（乳头状）**。

- **任务 A 结节分割**：输出二值 mask，指标 Dice / IoU / HD95
- **任务 C PTC 预测**：输入裁剪出的结节 patch，输出“是否为乳头状癌”，指标 Macro-F1 / Balanced Accuracy / Accuracy


## 目录结构（本文件夹 `ac_code/`）

```
ac_code/
├─ config.py                 # 全局配置（数据根目录、超参、归一化、裁剪/分类参数）
├─ requirements.txt          # 依赖
├─ README.md
├─ train.py                  # 两阶段训练：先 seg 再 cls（--stage seg/cls/both）
├─ evaluate.py               # 评估：seg 指标 + 完整级联 PTC 指标（患者级）
├─ infer.py                  # 推理：分割 -> 裁剪 -> 分类，输出 mask 与 PTC csv
├─ data/
│  ├─ thyroid_dataset.py     # 数据集（Seg / Patch）、letterbox、患者级划分
│  ├─ crop_utils.py          # mask -> 外接框 -> 裁剪结节 patch（train/eval/infer 共用）
│  └─ build_ptc_labels.py    # 由 annotations 生成默认 ptc_labels.csv
└─ models/
   ├─ unet.py                # 任务 A 分割 SegUNet
   ├─ classifier.py          # 任务 C PTC 分类器（timm EfficientNet 或内置 CNN）
   ├─ losses.py              # Dice+BCE / PTC(BCE+pos_weight)
   └─ metrics.py             # Dice/IoU/HD95；Macro-F1/Bal-Acc/Acc
```

数据根目录默认取本工程上级目录（即 `D:\study\ge\train`，里面含 `images/ masks/ train_annotations.json`）。
可用环境变量 `THYROID_DATA_ROOT` 或命令行 `--data_root` 覆盖。

## PTC 标签（任务 C）

代码读取 `数据根目录/ptc_labels.csv`，格式：

```
patient_id,ptc
00000058,1
00000370,0
...
```

- `ptc=1`：乳头状（正类）
- `ptc=0`：非乳头状（可靠负类，如良性）
- `ptc=-1`：不确定，训练/评估时排除

该文件**缺失时会自动**由 `train_annotations.json` 生成默认标签（乳头状 = Histopathology 含 papillary thyroid 或 FNAC 为乳头状；良性 = 可靠负类）。
**后续别人区分好的标签直接覆盖 `ptc_labels.csv` 即可，代码自动识别，无需改代码。**

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
- **阶段2(cls)**：用 GT mask 外接框（扩展 `margin`）裁剪结节 patch，训练 PTC 分类器；
  保存 `runs/cls_best.pth` / `cls_last.pth`。训练期分类用 GT crop（最干净），推理/评估用预测 mask crop。

## 评估

```bash
python evaluate.py              # 加载 best 模型，输出 任务A 与 任务C 报告
```

任务 C 会同时报告两种裁剪来源：
- `crop_from=pred`：用**预测** mask 裁剪
- `crop_from=gt`  ：用 **GT** mask 裁剪

## 推理

```bash
python infer.py                       # 用 runs/seg_best.pth + cls_best.pth
python infer.py --ckpt_seg runs/seg_best.pth --ckpt_cls runs/cls_best.pth --out predictions
```

输出：
- `predictions/masks/<name>.png`：预测 mask（原始尺寸）
- `predictions/ptc_preds.csv`：`patient_id, image, ptc_prob, ptc_pred`



