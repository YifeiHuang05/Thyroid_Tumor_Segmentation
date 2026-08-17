"""
推理（级联）：对每张图
  1) 分割 U-Net -> 预测 mask（原始尺寸 PNG）
  2) 用预测 mask 外接框裁剪结节 patch
  3) 多任务分类器（共享 patch）：
       头 bm      ：良性/恶性 -> bm_pred
       头 ptc     ：若预测为恶性，再判 PTC/非PTC -> ptc_pred（良性时不适用）
       头 fnac    ：Bethesda 三分类 -> fnac_pred
       头 tirands ：TI-RADS 五分类 -> tirands_pred
输出：
  - out/masks/<name>.png                      预测 mask（原始尺寸）
  - out/cls_preds.csv                        各头概率与预测（见表头）
用法：python infer.py  （可选 --ckpt_seg runs/seg_best.pth --ckpt_cls runs/cls_best.pth --out predictions）
"""
import os
import sys
import csv
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from config import Config, parse_overrides
from data.thyroid_dataset import letterbox, inverse_letterbox
from data.crop_utils import mask_to_bbox, crop_patch
from models.unet import SegUNet
from models.classifier import build_classifier, get_heads


class InferDataset(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.size = cfg.img_size
        self.mean = np.array(cfg.mean, np.float32).reshape(3, 1, 1)
        self.std = np.array(cfg.std, np.float32).reshape(3, 1, 1)
        self.files = sorted(glob.glob(os.path.join(cfg.images_path(), "*.png")))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        fp = self.files[i]
        name = os.path.basename(fp).replace(".png", "")
        pid = name.split("_")[0]
        img = np.array(Image.open(fp).convert("RGB"))
        img_p, params = letterbox(img, self.size)
        img_t = (img_p.astype(np.float32).transpose(2, 0, 1) / 255.0 - self.mean) / self.std
        return dict(image=np.ascontiguousarray(img_t),
                    img_raw=img_p.astype(np.uint8),
                    name=name, patient_id=pid, params=params)


def _collate(batch):
    return dict(
        image=torch.stack([torch.from_numpy(b["image"]) for b in batch]),
        img_raw=[b["img_raw"] for b in batch],
        name=[b["name"] for b in batch],
        patient_id=[b["patient_id"] for b in batch],
        params=[b["params"] for b in batch],
    )


def _load_seg(cfg, ckpt):
    sd = torch.load(ckpt, map_location="cpu")
    mcfg = sd.get("config", vars(cfg))
    model = SegUNet(in_channels=mcfg.get("in_channels", 3),
                    num_classes=mcfg.get("num_classes", 1),
                    base_channels=mcfg.get("base_channels", 64))
    model.load_state_dict(sd["model_state"])
    return model


def _load_cls(cfg, ckpt):
    sd = torch.load(ckpt, map_location="cpu")
    mcfg = sd.get("config", vars(cfg))
    heads = get_heads(mcfg)
    model = build_classifier(patch_size=mcfg.get("patch_size", 224),
                             backbone=mcfg.get("cls_backbone", "efficientnet_b3"),
                             pretrained=False, use_timm=mcfg.get("cls_use_timm", True),
                             heads=heads)
    model.load_state_dict(sd["model_state"])
    return model


def main():
    cfg = parse_overrides(sys.argv[1:], Config())
    import argparse
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--ckpt_seg", type=str, default=None)
    ap.add_argument("--ckpt_cls", type=str, default=None)
    ap.add_argument("--out", type=str, default="predictions")
    a, _ = ap.parse_known_args(sys.argv[1:])

    out_dir = cfg.out_path()
    ckpt_seg = a.ckpt_seg or os.path.join(out_dir, "seg_best.pth")
    ckpt_cls = a.ckpt_cls or os.path.join(out_dir, "cls_best.pth")
    if not os.path.exists(ckpt_seg):
        raise FileNotFoundError(f"找不到分割模型 {ckpt_seg}，请先运行 train.py --stage seg")
    if not os.path.exists(ckpt_cls):
        raise FileNotFoundError(f"找不到分类模型 {ckpt_cls}，请先运行 train.py --stage cls")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seg_model = _load_seg(cfg, ckpt_seg).to(device).eval()
    cls_model = _load_cls(cfg, ckpt_cls).to(device).eval()

    ds = InferDataset(cfg)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=cfg.num_workers, collate_fn=_collate, pin_memory=True)

    mean = np.array(cfg.mean, np.float32).reshape(3, 1, 1)
    std = np.array(cfg.std, np.float32).reshape(3, 1, 1)
    softmax = torch.nn.functional.softmax
    mask_dir = os.path.join(cfg.data_root, a.out, "masks")
    os.makedirs(mask_dir, exist_ok=True)
    csv_path = os.path.join(cfg.data_root, a.out, "cls_preds.csv")

    # 头：bm(2) ptc(2) fnac(3) tirands(5)
    fnac_names = list(cfg.fnac_names)
    tir_names = list(cfg.tirands_names)

    rows = []
    with torch.no_grad():
        for b in loader:
            seg = seg_model(b["image"].to(device))
            seg_prob = torch.sigmoid(seg).cpu().numpy()
            for i in range(seg_prob.shape[0]):
                pred = (seg_prob[i, 0] > 0.5).astype(np.uint8)
                pred_o = inverse_letterbox(pred, b["params"][i],
                                           b["params"][i]["orig_h"], b["params"][i]["orig_w"])
                Image.fromarray((pred_o * 255).astype(np.uint8), "L").save(
                    os.path.join(mask_dir, b["name"][i] + ".png"))

                bbox = mask_to_bbox(pred.astype(np.float32), cfg.margin)
                if bbox is None:
                    p_bm = np.array([1.0, 0.0])          # 无结节 -> 默认良性
                    p_ptc = np.array([1.0, 0.0])
                    p_fnac = np.zeros(len(fnac_names), dtype=np.float32)
                    p_tir = np.zeros(len(tir_names), dtype=np.float32)
                else:
                    patch = crop_patch(b["img_raw"][i], bbox, cfg.patch_size)
                    patch_t = (patch.astype(np.float32).transpose(2, 0, 1) / 255.0 - mean) / std
                    t = torch.from_numpy(patch_t).unsqueeze(0).to(device)
                    out = cls_model(t)
                    p_bm = softmax(out["bm"], dim=1).cpu().numpy()[0]
                    p_ptc = softmax(out["ptc"], dim=1).cpu().numpy()[0]
                    p_fnac = softmax(out["fnac"], dim=1).cpu().numpy()[0]
                    p_tir = softmax(out["tirands"], dim=1).cpu().numpy()[0]

                bm_pred = int(np.argmax(p_bm))
                if bm_pred == 1:                         # 恶性 -> 预测 PTC
                    ptc_pred = int(np.argmax(p_ptc))
                    p_ptc_pos = float(p_ptc[1])
                else:                                    # 良性 -> PTC 不适用
                    ptc_pred = -1
                    p_ptc_pos = float("nan")
                fnac_pred = int(np.argmax(p_fnac))
                tir_pred = int(np.argmax(p_tir))

                row = [b["patient_id"][i], b["name"][i],
                       f"{p_bm[0]:.4f}", f"{p_bm[1]:.4f}", bm_pred]
                # ptc
                row += [f"{p_ptc_pos:.4f}" if bm_pred == 1 else "", ptc_pred]
                # fnac 概率 + 预测
                row += [f"{p_fnac[j]:.4f}" for j in range(len(p_fnac))] + [fnac_pred]
                # tirands 概率 + 预测
                row += [f"{p_tir[j]:.4f}" for j in range(len(p_tir))] + [tir_pred]
                rows.append(tuple(row))

    header = ["patient_id", "image", "p_benign", "p_malignant", "bm_pred",
              "p_ptc", "ptc_pred"]
    header += [f"p_fnac_{fnac_names[j]}" for j in range(len(fnac_names))] + ["fnac_pred"]
    header += [f"p_{tir_names[j]}" for j in range(len(tir_names))] + ["tirands_pred"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"[infer] 完成：mask -> {mask_dir}  | 多任务分类 -> {csv_path}  (共 {len(rows)} 行)")


if __name__ == "__main__":
    main()
