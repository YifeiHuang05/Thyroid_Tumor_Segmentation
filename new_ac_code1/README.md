# ThyroidXL 任务 A + C 代码工程（级联 cascade，多任务分类）

**级联设计**：任务 A 的分割结果作为任务 C 的输入 ——
**U-Net 分割结节 → 用预测 mask 外接框裁剪结节 → CNN 多任务分类器判定**。

任务 C 是一个**共享 backbone + 多个并行分类头**的多任务模型，全部基于 A 裁剪出的结节 patch：

1. **bm 头（良恶性）**：良性 / 恶性（所有结节都参与）
2. **ptc 头（PTC 亚型）**：仅在「恶性」内部区分 **PTC（乳头状）/ 非 PTC（非乳头状恶性）**
3. **fnac 头（Bethesda 三分类）**：II / III-IV / V-VI
4. **tirands 头（TI-RADS 五分类）**：原始 1–5 级

- **任务 A 结节分割**：输出二值 mask，指标 Dice / IoU / HD95
- **任务 C 多任务分类**（每个头均患者级聚合）：指标 Macro-F1 / Micro-F1 / Balanced Accuracy / Accuracy + 混淆矩阵

## 目录结构（本文件夹 `new_ac_code/`）

```
new_ac_code/
├─ config.py                 # 全局配置（数据根目录、超参、归一化、裁剪/分类参数、cls_heads()）
├─ requirements.txt          # 依赖
├─ README.md
├─ train.py                  # 两阶段训练：先 seg 再 cls（--stage seg/cls/both）
├─ evaluate.py               # 评估：seg 指标 + 完整级联多任务分类指标（患者级）
├─ infer.py                  # 推理：分割 -> 裁剪 -> 多任务分类，输出 mask 与 cls csv
├─ data/
│  ├─ thyroid_dataset.py     # 数据集（Seg / Patch）、letterbox、患者级划分
│  ├─ crop_utils.py          # mask -> 外接框 -> 裁剪结节 patch（train/eval/infer 共用）
│  └─ build_cls_labels.py    # 由 annotations 生成默认 cls_labels.csv（patient_id,bm,ptc,fnac,tirands）
└─ models/
   ├─ unet.py                # 任务 A 分割 SegUNet
   ├─ classifier.py          # 任务 C 多任务头分类器（timm EfficientNet 或内置 CNN）
   ├─ losses.py              # Dice+BCE / 多任务 CrossEntropy（带 class_weight）
   └─ metrics.py             # Dice/IoU/HD95；多分类 Macro-F1/Micro-F1/Bal-Acc/Acc + 混淆矩阵
```

数据根目录默认取本工程上级目录（即 `D:\study\ge\train`，里面含 `images/ masks/ train_annotations.json`）。
可用环境变量 `THYROID_DATA_ROOT` 或命令行 `--data_root` 覆盖。

## 多任务标签（任务 C）

代码读取 `数据根目录/cls_labels.csv`，格式（**五列**）：

```
patient_id,bm,ptc,fnac,tirands
00000058,0,-1,0,3
00000370,1,1,2,5
...
```

- `bm` （良恶性）：`0`=良性，`1`=恶性，`-1`=排除
- `ptc`（PTC/非PTC，仅恶性内部）：`1`=PTC(乳头状)，`0`=非PTC(非乳头状恶性)，`-1`=不适用/无病理（不参与 PTC 训练）
- `fnac`（Bethesda 三分类）：`0`=Bethesda II，`1`=Bethesda III-IV，`2`=Bethesda V-VI，`-1`=Bethesda I 或缺失（不参与 FNAC 训练）
- `tirands`（TI-RADS 五分类）：`0`..`4`=TR1..TR5，`-1`=缺失/非法（不参与训练）

### 默认标签生成规则（`build_cls_labels.py`，缺失时自动生成）

- **`bm`**：`labels_orig` 标 `lanhtinh` → 0（良性）；标 `actinh` → 1（恶性）；否则 -1。
- **`ptc`**（仅恶性，且 **严格只用明确病理金标准，不使用 FNAC**）：
  - 恶性 且 `Histopathology` 含 `"papillary thyroid"` → `1`（组织学确诊乳头状）
  - 恶性 且 有明确非乳头状病理（如 medullary / follicular / hurthle）→ `0`
  - 恶性 但无明确病理记录 → `-1`（不参与 PTC 训练/评估）
  - 良性 → `-1`（PTC 不适用）
- **`fnac`**（FNAC 原始字段为数字串，按参赛须知合并三分类）：
  - 原始值 `== 2` → `0`（Bethesda II）
  - 原始值 `∈ {3, 4}` → `1`（Bethesda III-IV）
  - 原始值 `∈ {5, 6}`（含 `"6 (papillary thyroid carcinoma)"`）→ `2`（Bethesda V-VI）
  - `== 1` 或缺失/空 → `-1`（排除）
- **`tirands`**：原始 `1..5` → `0..4`；缺失/非法 → `-1`。

**后续别人给出更精确的病理/亚型区分标签，直接覆盖 `cls_labels.csv`（保持 `patient_id,bm,ptc,fnac,tirands` 五列）即可被自动读取，无需改代码。**

> ⚠️ **PTC 负类稀缺警告**：训练集中「恶性 + 有明确非乳头状病理」的病例极少（仅个位数患者），
> 而官方测试集该子集有 578 张图（537 PTC + 41 非PTC）。这是数据固有瓶颈：PTC 头负类样本极少，
> 训练时默认开启 `cls_class_weights` 做类别加权，但模型仍容易偏向预测阳性。
> 若同事能补充更精确的非乳头状恶性标签，PTC 头质量会明显提升。
>
> ⚠️ **FNAC 三分类极不平衡**：Bethesda II 占绝大多数（2463），III-IV 仅 38 例；V-VI 741 例。
> 同样依赖 `cls_class_weights`，并建议留意 III-IV 头的召回。

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
- **阶段2(cls)**：用 GT mask 外接框（扩展 `margin`）裁剪结节 patch，训练多任务头分类器
  （bm / ptc / fnac / tirands 四个头都用 softmax + 带 class_weight 的 CrossEntropy，按各自类别频率加权）；
  保存 `runs/cls_best.pth` / `cls_last.pth`。训练期分类用 GT crop（最干净），推理/评估用预测 mask crop。

## 评估

```bash
python evaluate.py              # 加载 best 模型，输出 任务A 与 任务C 四头报告
```

任务 C 会分别给出四个头的报告：
- **任务 C-bm 良性/恶性**：全部有 bm 标签的患者
- **任务 C-ptc PTC/非PTC**：仅恶性 + 有病理的患者（`ptc>=0`）
- **任务 C-fnac Bethesda 三分类**：有 fnac 标签的患者（`fnac>=0`）
- **任务 C-tirands TI-RADS 五分类**：有 tirands 标签的患者（`tirands>=0`）

每个头均报告两种裁剪来源（`crop_from=pred` 用预测 mask，与推理一致；`crop_from=gt` 用 GT mask 做上限参考），
以及 Accuracy / Balanced-Acc / Macro-F1 / **Micro-F1** + 每类 Precision/Recall/F1 + 混淆矩阵。

## 推理

```bash
python infer.py                       # 用 runs/seg_best.pth + cls_best.pth
python infer.py --ckpt_seg runs/seg_best.pth --ckpt_cls runs/cls_best.pth --out predictions
```

输出：
- `predictions/masks/<name>.png`：预测 mask（原始尺寸）
- `predictions/cls_preds.csv`：表头为
  `patient_id, image, p_benign, p_malignant, bm_pred, p_ptc, ptc_pred,
  p_fnac_Bethesda_II, p_fnac_Bethesda_III-IV, p_fnac_Bethesda_V-VI, fnac_pred,
  p_TR1, p_TR2, p_TR3, p_TR4, p_TR5, tirands_pred`
  （`bm_pred=1` 恶性时才有 `p_ptc`/`ptc_pred`；良性时 PTC 列为空，`ptc_pred=-1` 表示不适用）

## 关键约定（来自数据探查）

- 图像真 RGB、尺寸不一 → letterbox 保持比例后 resize 到 512×512。
- mask 灰度 {0,255}，前景=255；结节面积占比均值 ~7.8%（小目标，强不平衡）→ 分割用 Dice+BCE。
- 文件名 `{患者ID}_{检查ID}_{图序}.png`，前缀=患者ID，用于患者级划分。
- 任务 C 标签是**患者级**：同一患者多张图共享标签，评估在患者级聚合（图像级概率平均后取 argmax）。
- 分类器输入是**结节 patch 而非整图**，直接喂整图容易学背景；这是级联的核心价值。
- 历史三分类脚本已弃用；旧二分类（PTC/非PTC）脚本 `data/build_ptc_labels.py`、`data/build_ptc_labels_fnac.py` 已不再被工程调用，仅作归档参考。
