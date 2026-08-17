"""
全局配置：自动定位数据根目录，集中管理超参数。
数据根目录默认取本工程上级目录（即 D:/study/ge/train），
可用环境变量 THYROID_DATA_ROOT 或命令行 --data_root 覆盖。

架构：级联（cascade）
    任务 A 分割 U-Net  ->  预测 mask  ->  按 mask 外接框裁剪结节
    ->  任务 C 级联两步分类（输入结节 patch）：
          第 1 步：良性 / 恶性（2 分类，bm 头）
          第 2 步：在“恶性”内部区分 PTC / 非 PTC（2 分类，ptc 头）
    即“A 的输出（mask）送入 C”的流水线；分类本身再分两级。

【标签口径（硬性要求）】
    PTC 仅在“具有明确病理金标准 且 真实为恶性”的结节中定义；
    不使用 FNAC 结果作为 PTC 真值。
    - ptc = 1 ：恶性 且 Histopathology 含 "papillary thyroid"（组织学确诊乳头状）
    - ptc = 0 ：恶性 且 有明确非乳头状病理（如 medullary / follicular / hurthle）
    - ptc = -1：良性，或 恶性但无明确病理记录（不参与 PTC 训练/评估）

【FNAC 三分类（Bethesda）】
    FNAC 原始字段为数字串（'2'..'6'，'6' 含 "papillary thyroid" 等附加说明）；'1'=Bethesda I、空/缺失为无效。
    按参赛须知合并为三分类：
    - fnac = 0 ：Bethesda II                  （原始值 == 2）
    - fnac = 1 ：Bethesda III-IV              （原始值 ∈ {3, 4}）
    - fnac = 2 ：Bethesda V-VI                （原始值 ∈ {5, 6}）
    - fnac = -1：Bethesda I 或 缺失（不参与 FNAC 训练/评估）

【TI-RADS 五分类】
    TIRADS 原始字段为 '1'..'5'；映射为 0..4，缺失/非法为 -1（不参与训练/评估）。
"""
import os
from dataclasses import dataclass

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
# 数据根目录：默认是工程目录的上级（即 train 目录，里面含 images/ masks/ 等）
DEFAULT_DATA_ROOT = os.environ.get("THYROID_DATA_ROOT", os.path.dirname(PROJECT_DIR))


@dataclass
class Config:
    # ---- 路径 ----
    data_root: str = DEFAULT_DATA_ROOT
    images_dir: str = "images"
    masks_dir: str = "masks"
    annotations: str = "train_annotations.json"
    cls_labels: str = "cls_labels.csv"        # 层级标签；缺失时由 build_cls_labels.py 生成默认
    splits: str = "splits.json"               # 患者级 train/val 划分
    out_dir: str = "runs"                     # 训练产物（checkpoint/日志）

    # ---- 分割（任务 A）预处理 ----
    img_size: int = 512                        # 统一 resize 到正方形（letterbox 保持比例）
    in_channels: int = 3
    num_classes: int = 1                       # 分割类别数（单类结节）
    mean: tuple = (0.485, 0.456, 0.406)        # ImageNet 归一化（RGB）
    std: tuple = (0.229, 0.224, 0.225)

    # ---- 分割模型 ----
    base_channels: int = 64

    # ---- 裁剪（用 A 的 mask 框出结节，送进 C） ----
    margin: float = 0.1                        # 外接框向外扩展比例（避免裁掉边缘）
    patch_size: int = 224                      # 裁剪后送入分类器的尺寸

    # ---- 分类（任务 C，多任务头，全部基于 A 裁剪的结节 patch） ----
    # 头1：良性 / 恶性（bm）
    bm_num_classes: int = 2
    bm_names: tuple = ("benign", "malignant")  # 索引 1 = 恶性
    # 头2：PTC / 非 PTC（仅在恶性内部，仅用明确病理，不用 FNAC）
    ptc_num_classes: int = 2
    ptc_names: tuple = ("non_ptc", "ptc")      # 索引 1 = PTC（正类）
    # 头3：FNAC 三分类（Bethesda II / III-IV / V-VI）
    fnac_num_classes: int = 3
    fnac_names: tuple = ("Bethesda_II", "Bethesda_III-IV", "Bethesda_V-VI")
    # 头4：TI-RADS 五分类（原始 1-5 -> 0-4）
    tirands_num_classes: int = 5
    tirands_names: tuple = ("TR1", "TR2", "TR3", "TR4", "TR5")
    cls_backbone: str = "efficientnet_b3"      # timm 可用时的 backbone
    cls_use_timm: bool = True                  # 关闭则强制使用内置 CNN
    cls_pretrained: bool = True                # 是否加载 ImageNet 预训练权重
    cls_class_weights: bool = True             # 是否按训练集类别频率计算 class_weight（应对不平衡）

    # ---- 训练（分割 A） ----
    batch_size: int = 8
    epochs: int = 50
    lr: float = 1e-4
    weight_decay: float = 1e-4
    val_ratio: float = 0.15                    # 患者级验证集比例
    bce_weight: float = 0.5                    # 分割损失中 BCE 占比（其余为 Dice）
    num_workers: int = 4
    seed: int = 42
    seg_aug: bool = True                       # 分割训练数据增强（同步翻转图像+mask）

    # ---- 训练（分类 C） ----
    cls_batch_size: int = 32
    cls_epochs: int = 30
    cls_lr: float = 3e-4
    cls_weight_decay: float = 1e-4
    cls_aug: bool = True                       # 分类训练对 patch 做水平翻转

    def images_path(self):
        return os.path.join(self.data_root, self.images_dir)

    def masks_path(self):
        return os.path.join(self.data_root, self.masks_dir)

    def annotations_path(self):
        return os.path.join(self.data_root, self.annotations)

    def cls_labels_path(self):
        return os.path.join(self.data_root, self.cls_labels)

    def splits_path(self):
        return os.path.join(self.data_root, self.splits)

    def out_path(self):
        return os.path.join(self.data_root, self.out_dir)

    def cls_heads(self):
        """返回分类多头列表 [(name, num_classes, names), ...]。
        顺序即模型输出 dict 的 key 顺序，供 classifier/loss/metrics 统一遍历。"""
        return [
            ("bm", self.bm_num_classes, list(self.bm_names)),
            ("ptc", self.ptc_num_classes, list(self.ptc_names)),
            ("fnac", self.fnac_num_classes, list(self.fnac_names)),
            ("tirands", self.tirands_num_classes, list(self.tirands_names)),
        ]


def parse_overrides(argv_list, cfg: Config) -> Config:
    """轻量命令行覆盖，例如 --data_root xxx --lr 2e-4 --epochs 80 --patch_size 256"""
    import argparse
    p = argparse.ArgumentParser(add_help=False)
    for f in ["data_root", "images_dir", "masks_dir", "annotations", "cls_labels",
              "splits", "out_dir", "img_size", "in_channels", "num_classes",
              "base_channels", "margin", "patch_size",
              "bm_num_classes", "ptc_num_classes", "fnac_num_classes", "tirands_num_classes",
              "cls_backbone", "cls_use_timm", "cls_pretrained", "cls_class_weights",
              "batch_size", "epochs", "lr", "weight_decay",
              "val_ratio", "bce_weight", "num_workers", "seed", "seg_aug",
              "cls_batch_size", "cls_epochs", "cls_lr", "cls_weight_decay", "cls_aug"]:
        p.add_argument(f"--{f}", type=type(getattr(cfg, f)))
    known, _ = p.parse_known_args(argv_list)
    for k, v in vars(known).items():
        if v is not None:
            setattr(cfg, k, v)
    return cfg
