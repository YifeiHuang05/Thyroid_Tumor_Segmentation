"""
ThyroidXL 数据集（级联架构）：
  - ThyroidSegDataset  : 任务 A 分割，图像 + mask 配对（含 letterbox 预处理、可选增强）。
  - ThyroidPatchDataset: 任务 C 分类，用 GT mask 外接框从原图裁出结节 patch（含多任务标签 bm/ptc/fnac/tirands）。

共用：患者级划分（防泄漏）、层级标签读取（缺失自动生成默认）。

标签含义（见 build_cls_labels.py）：
  bm      ∈ {-1, 0, 1}   0=良性, 1=恶性, -1=排除
  ptc     ∈ {-1, 0, 1}   1=PTC(乳头状), 0=非PTC(非乳头状恶性), -1=不适用/无病理（不参与 PTC 训练）
  fnac    ∈ {-1, 0, 1, 2} 0=Bethesda II, 1=Bethesda III-IV, 2=Bethesda V-VI, -1=排除
  tirands ∈ {-1, 0..4}   0..4=TR1..TR5, -1=排除
"""
import os
import csv
import json
import glob
import random

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from data.crop_utils import mask_to_bbox, crop_patch


# ---------------- 预处理 ----------------
def letterbox(arr, size, fill=0):
    """arr: HWC 或 HW numpy；保持比例 resize 后 pad 到 size×size。返回 (padded, params)"""
    h, w = arr.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    nh, nw = max(nh, 1), max(nw, 1)
    if arr.ndim == 3:
        resized = np.array(Image.fromarray(arr).resize((nw, nh), Image.BILINEAR))
    else:
        resized = np.array(Image.fromarray(arr).resize((nw, nh), Image.NEAREST))
    pad_top = (size - nh) // 2
    pad_left = (size - nw) // 2
    if arr.ndim == 3:
        padded = np.full((size, size, arr.shape[2]), fill, dtype=arr.dtype)
        padded[pad_top:pad_top + nh, pad_left:pad_left + nw] = resized
    else:
        padded = np.full((size, size), fill, dtype=arr.dtype)
        padded[pad_top:pad_top + nh, pad_left:pad_left + nw] = resized
    params = dict(orig_h=h, orig_w=w, nh=nh, nw=nw, pad_top=pad_top, pad_left=pad_left)
    return padded, params


def inverse_letterbox(pred, params, orig_h, orig_w):
    """pred: size×size numpy(0/1)；裁回 letterbox 区域并 resize 回原尺寸。"""
    pad_top, pad_left, nh, nw = params["pad_top"], params["pad_left"], params["nh"], params["nw"]
    crop = pred[pad_top:pad_top + nh, pad_left:pad_left + nw]
    return np.array(Image.fromarray(crop).resize((orig_w, orig_h), Image.NEAREST))


# ---------------- 划分 ----------------
def make_or_load_splits(cfg, rebuild=False):
    sp = cfg.splits_path()
    if os.path.exists(sp) and not rebuild:
        with open(sp, "r", encoding="utf-8") as f:
            return json.load(f)
    ann_path = cfg.annotations_path()
    with open(ann_path, "r", encoding="utf-8") as f:
        ann = json.load(f)
    pats = ann["info"] if isinstance(ann, dict) and "info" in ann else ann
    pids = list(pats.keys())
    random.seed(cfg.seed)
    random.shuffle(pids)
    n_val = max(1, int(round(len(pids) * cfg.val_ratio)))
    val = pids[:n_val]
    train = pids[n_val:]
    splits = dict(train=train, val=val)
    with open(sp, "w", encoding="utf-8") as f:
        json.dump(splits, f, ensure_ascii=False, indent=2)
    print(f"[splits] train={len(train)} val={len(val)} -> {sp}")
    return splits


# ---------------- 层级标签 (bm, ptc, fnac, tirands) ----------------
def load_hier_labels(cfg):
    p = cfg.cls_labels_path()
    if not os.path.exists(p):
        from data.build_cls_labels import build_cls_labels
        build_cls_labels(cfg.data_root, p)
    mapping = {}
    with open(p, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            mapping[row["patient_id"]] = dict(
                bm=int(row["bm"]),
                ptc=int(row["ptc"]),
                fnac=int(row["fnac"]),
                tirands=int(row["tirands"]),
            )
    return mapping


# ---------------- 样本清单（seg / patch 共用） ----------------
def _build_samples(cfg, split, require_bm=False):
    """require_bm=True 仅保留有 bm 标签的样本（良性/恶性，用于分类训练）。"""
    splits = make_or_load_splits(cfg)
    pids = splits.get(split, [])
    mapping = load_hier_labels(cfg)
    img_dir = cfg.images_path()
    mask_dir = cfg.masks_path()
    out = []
    for pid in pids:
        for fp in sorted(glob.glob(os.path.join(img_dir, f"{pid}_*.png"))):
            name = os.path.basename(fp).replace(".png", "")
            mpath = os.path.join(mask_dir, name + ".png")
            if not os.path.exists(mpath):
                continue
            lab = mapping.get(pid, dict(bm=-1, ptc=-1, fnac=-1, tirands=-1))
            bm, ptc, fnac, tirands = lab["bm"], lab["ptc"], lab["fnac"], lab["tirands"]
            if require_bm and bm < 0:
                continue
            out.append(dict(
                img=os.path.join(img_dir, name + ".png"),
                mask=mpath,
                patient_id=pid,
                bm=int(bm),
                ptc=int(ptc),
                fnac=int(fnac),
                tirands=int(tirands),
                has_ptc=(ptc >= 0),
                has_bm=(bm >= 0),
                has_fnac=(fnac >= 0),
                has_tirands=(tirands >= 0),
            ))
    return out


# ---------------- 数据集 A：分割 ----------------
class ThyroidSegDataset(Dataset):
    def __init__(self, cfg, split="train", augment=None, transforms=None):
        self.cfg = cfg
        self.size = cfg.img_size
        self.augment = cfg.seg_aug if augment is None else augment
        self.transforms = transforms
        self.mean = np.array(cfg.mean, dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array(cfg.std, dtype=np.float32).reshape(3, 1, 1)
        self.samples = _build_samples(cfg, split, require_bm=False)
        print(f"[SegDataset:{split}] {len(self.samples)} 样本")

    def __len__(self):
        return len(self.samples)

    def _augment(self, img_p, mask_p):
        if not self.augment:
            return img_p, mask_p
        if random.random() < 0.5:                      # 水平翻转（图像+mask 同步）
            img_p = img_p[:, ::-1, :].copy()
            mask_p = mask_p[:, ::-1].copy()
        if random.random() < 0.5:                      # 垂直翻转
            img_p = img_p[::-1, :, :].copy()
            mask_p = mask_p[::-1, :].copy()
        return img_p, mask_p

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = np.array(Image.open(s["img"]).convert("RGB"))
        mask = (np.array(Image.open(s["mask"]).convert("L")) > 127).astype(np.float32)
        img_p, params = letterbox(img, self.size)
        mask_p, _ = letterbox(mask, self.size, fill=0)
        img_p, mask_p = self._augment(img_p, mask_p)
        img_t = (img_p.astype(np.float32).transpose(2, 0, 1) / 255.0 - self.mean) / self.std
        mask_t = mask_p[None, ...].astype(np.float32)
        item = dict(
            image=np.ascontiguousarray(img_t),
            mask=np.ascontiguousarray(mask_t),
            img_raw=img_p.astype(np.uint8),        # letterbox 后 uint8，供 C 裁剪用
            mask_path=s["mask"],
            patient_id=s["patient_id"],
            bm=int(s["bm"]),
            ptc=int(s["ptc"]),
            fnac=int(s["fnac"]),
            tirands=int(s["tirands"]),
            has_ptc=bool(s["has_ptc"]),
            has_bm=bool(s["has_bm"]),
            has_fnac=bool(s["has_fnac"]),
            has_tirands=bool(s["has_tirands"]),
            params=params,
        )
        if self.transforms is not None:
            item = self.transforms(item)
        return item


# ---------------- 数据集 C：结节 patch 分类（层级两步） ----------------
class ThyroidPatchDataset(Dataset):
    """用 GT mask 外接框裁出结节 patch（训练时 crop 来自 GT 最干净；
    推理/评估时 crop 来自预测 mask，由 pipeline 负责）。
    标签：bm 为良性/恶性（全部有 bm 标签的都用），ptc 仅在恶性且有病理时为 0/1；
    fnac / tirands 为患者级标签（每患者单结节），对所有样本均透传。"""

    def __init__(self, cfg, split="train", augment=None, transforms=None):
        self.cfg = cfg
        self.size = cfg.img_size
        self.patch_size = cfg.patch_size
        self.margin = cfg.margin
        self.augment = cfg.cls_aug if augment is None else augment
        self.transforms = transforms
        self.mean = np.array(cfg.mean, dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array(cfg.std, dtype=np.float32).reshape(3, 1, 1)
        self.samples = _build_samples(cfg, split, require_bm=True)
        print(f"[PatchDataset:{split}] {len(self.samples)} 样本 "
              f"(其中 PTC 有标签={sum(1 for x in self.samples if x['has_ptc'])})")

    def __len__(self):
        return len(self.samples)

    def _augment(self, patch):
        if self.augment and random.random() < 0.5:     # 仅水平翻转（已无 mask）
            patch = patch[:, ::-1, :].copy()
        return patch

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = np.array(Image.open(s["img"]).convert("RGB"))
        mask = (np.array(Image.open(s["mask"]).convert("L")) > 127).astype(np.float32)
        img_p, _ = letterbox(img, self.size)
        mask_p, _ = letterbox(mask, self.size, fill=0)
        bbox = mask_to_bbox(mask_p, self.margin)
        if bbox is None:                                # 理论上不会发生（GT 必有前景）
            bbox = (0, 0, self.size, self.size)
        patch = crop_patch(img_p, bbox, self.patch_size)
        patch = self._augment(patch)
        patch_t = (patch.astype(np.float32).transpose(2, 0, 1) / 255.0 - self.mean) / self.std
        return dict(
            patch=np.ascontiguousarray(patch_t),
            bm=np.int64(s["bm"]),
            ptc=np.int64(s["ptc"]),
            fnac=np.int64(s["fnac"]),
            tirands=np.int64(s["tirands"]),
            has_ptc=bool(s["has_ptc"]),
            has_bm=bool(s["has_bm"]),
            has_fnac=bool(s["has_fnac"]),
            has_tirands=bool(s["has_tirands"]),
            patient_id=s["patient_id"],
        )
