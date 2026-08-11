"""
全局配置：自动定位数据根目录，集中管理超参数。
数据根目录默认取本工程上级目录（即 D:/study/ge/train），
可用环境变量 THYROID_DATA_ROOT 或命令行 --data_root 覆盖。

架构：级联（cascade）
    任务 A 分割 U-Net  ->  预测 mask  ->  按 mask 外接框裁剪结节
    ->  任务 C 分类器（输入结节 patch，输出 PTC 概率）
即“A 的输出（mask）送入 C”的流水线。
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
    ptc_labels: str = "ptc_labels.csv"        # 用户提供；缺失时由 build_ptc_labels.py 生成默认
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

    # ---- 分类（任务 C） ----
    cls_backbone: str = "efficientnet_b3"      # timm 可用时的 backbone
    cls_use_timm: bool = True                  # 关闭则强制使用内置 CNN
    cls_pretrained: bool = True                # 是否加载 ImageNet 预训练权重

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

    def ptc_labels_path(self):
        return os.path.join(self.data_root, self.ptc_labels)

    def splits_path(self):
        return os.path.join(self.data_root, self.splits)

    def out_path(self):
        return os.path.join(self.data_root, self.out_dir)


def parse_overrides(argv_list, cfg: Config) -> Config:
    """轻量命令行覆盖，例如 --data_root xxx --lr 2e-4 --epochs 80 --patch_size 256"""
    import argparse
    p = argparse.ArgumentParser(add_help=False)
    for f in ["data_root", "images_dir", "masks_dir", "annotations", "ptc_labels",
              "splits", "out_dir", "img_size", "in_channels", "num_classes",
              "base_channels", "margin", "patch_size", "cls_backbone", "cls_use_timm",
              "cls_pretrained", "batch_size", "epochs", "lr", "weight_decay",
              "val_ratio", "bce_weight", "num_workers", "seed", "seg_aug",
              "cls_batch_size", "cls_epochs", "cls_lr", "cls_weight_decay", "cls_aug"]:
        p.add_argument(f"--{f}", type=type(getattr(cfg, f)))
    known, _ = p.parse_known_args(argv_list)
    for k, v in vars(known).items():
        if v is not None:
            setattr(cfg, k, v)
    return cfg
