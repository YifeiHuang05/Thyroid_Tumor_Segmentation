"""
推理（级联）：对每张图
  1) 分割 U-Net -> 预测 mask（原始尺寸 PNG）
  2) 用预测 mask 外接框裁剪结节 patch
  3) 分类器 -> PTC 概率
输出：
  - out/masks/<name>.png                预测 mask（原始尺寸）
  - out/ptc_preds.csv                   patient_id, image, ptc_prob, ptc_pred
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
from models.classifier import build_classifier


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
    model = build_classifier(patch_size=mcfg.get("patch_size", 224),
                             backbone=mcfg.get("cls_backbone", "efficientnet_b3"),
                             pretrained=False,
                             use_timm=mcfg.get("cls_use_timm", True))
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
    mask_dir = os.path.join(cfg.data_root, a.out, "masks")
    os.makedirs(mask_dir, exist_ok=True)
    csv_path = os.path.join(cfg.data_root, a.out, "ptc_preds.csv")
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
                    p = 0.0
                else:
                    patch = crop_patch(b["img_raw"][i], bbox, cfg.patch_size)
                    patch_t = (patch.astype(np.float32).transpose(2, 0, 1) / 255.0 - mean) / std
                    t = torch.from_numpy(patch_t).unsqueeze(0).to(device)
                    p = torch.sigmoid(cls_model(t)).item()
                rows.append((b["patient_id"][i], b["name"][i], f"{p:.4f}", int(p >= 0.5)))

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["patient_id", "image", "ptc_prob", "ptc_pred"])
        w.writerows(rows)
    print(f"[infer] 完成：mask -> {mask_dir}  | PTC -> {csv_path}  (共 {len(rows)} 行)")


if __name__ == "__main__":
    main()
