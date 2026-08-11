"""
训练入口：级联两阶段（默认都跑）。
  阶段1 (seg)  : 训练任务 A 分割 U-Net，保存 runs/seg_best.pth / seg_last.pth
  阶段2 (cls)  : 用 GT mask 裁出结节 patch，训练任务 C PTC 分类器，保存 runs/cls_best.pth / cls_last.pth
用法：
    python train.py                 # 先 seg 后 cls
    python train.py --stage seg     # 只训练分割
    python train.py --stage cls     # 只训练分类（不需要 seg 模型）
    python train.py --epochs 80 --cls_epochs 40 --patch_size 256
PTC 标签默认由 annotations 生成 ptc_labels.csv；覆盖该文件即可用自有标签。
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config, parse_overrides
from data.thyroid_dataset import ThyroidSegDataset, ThyroidPatchDataset
from models.unet import SegUNet
from models.classifier import build_classifier
from models.losses import seg_loss, PTCBCE
from evaluate import evaluate_seg, _collate_seg, _collate_cls, _cls_quick_val


def compute_pos_weight_patch(ds):
    pos = sum(1 for s in ds.samples if s["ptc"] == 1)
    neg = sum(1 for s in ds.samples if s["ptc"] == 0)
    if pos == 0 or neg == 0:
        return None
    return torch.tensor([neg / pos], dtype=torch.float32)


# ---------------- 阶段1：分割 ----------------
def train_seg(cfg, device):
    print("\n===== 阶段1：任务 A 分割 =====")
    train_ds = ThyroidSegDataset(cfg, split="train")
    val_ds = ThyroidSegDataset(cfg, split="val")
    train_ld = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                          num_workers=cfg.num_workers, collate_fn=_collate_seg, pin_memory=True)
    val_ld = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=cfg.num_workers, collate_fn=_collate_seg, pin_memory=True)

    model = SegUNet(in_channels=cfg.in_channels, num_classes=cfg.num_classes,
                    base_channels=cfg.base_channels).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[seg] 参数 {n_params:.1f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    out_dir = cfg.out_path()
    os.makedirs(out_dir, exist_ok=True)
    best = dict(dice=-1.0, epoch=0)

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        model.train()
        run = 0.0
        for b in tqdm(train_ld, desc=f"seg epoch {epoch}/{cfg.epochs}", leave=False):
            img = b["image"].to(device)
            mask = b["mask"].to(device)
            logits = model(img)
            loss = seg_loss(logits, mask, cfg.bce_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            run += loss.item()

        seg_m = evaluate_seg(model, val_ld, device)
        print(f"[seg epoch {epoch}] Dice={seg_m['dice']:.4f} IoU={seg_m['iou']:.4f} "
              f"HD95={seg_m['hd95']:.2f} | train_loss={run/len(train_ld):.4f} | {time.time()-t0:.0f}s")

        ckpt = dict(epoch=epoch, model_state=model.state_dict(),
                    config=vars(cfg), seg=seg_m)
        torch.save(ckpt, os.path.join(out_dir, "seg_last.pth"))
        if seg_m["dice"] > best["dice"]:
            best.update(dice=seg_m["dice"], epoch=epoch)
            torch.save(ckpt, os.path.join(out_dir, "seg_best.pth"))
        print(f"  best Dice={best['dice']:.4f} @epoch {best['epoch']}")
    return os.path.join(out_dir, "seg_best.pth")


# ---------------- 阶段2：分类 ----------------
def train_cls(cfg, device):
    print("\n===== 阶段2：任务 C PTC 分类（结节 patch） =====")
    train_ds = ThyroidPatchDataset(cfg, split="train")
    val_ds = ThyroidPatchDataset(cfg, split="val")
    train_ld = DataLoader(train_ds, batch_size=cfg.cls_batch_size, shuffle=True,
                          num_workers=cfg.num_workers, collate_fn=_collate_cls, pin_memory=True)
    val_ld = DataLoader(val_ds, batch_size=cfg.cls_batch_size, shuffle=False,
                        num_workers=cfg.num_workers, collate_fn=_collate_cls, pin_memory=True)

    model = build_classifier(patch_size=cfg.patch_size, backbone=cfg.cls_backbone,
                             pretrained=cfg.cls_pretrained, use_timm=cfg.cls_use_timm).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[cls] 参数 {n_params:.1f}M  patch_size={cfg.patch_size}")

    pos_weight = compute_pos_weight_patch(train_ds)
    crit = PTCBCE(pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.cls_lr, weight_decay=cfg.cls_weight_decay)
    out_dir = cfg.out_path()
    os.makedirs(out_dir, exist_ok=True)
    best = dict(acc=-1.0, epoch=0)

    for epoch in range(1, cfg.cls_epochs + 1):
        t0 = time.time()
        model.train()
        run = 0.0
        for b in tqdm(train_ld, desc=f"cls epoch {epoch}/{cfg.cls_epochs}", leave=False):
            patch = b["patch"].to(device)
            ptc = b["ptc"].to(device).unsqueeze(1)
            logits = model(patch)
            loss = crit(logits, ptc)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            run += loss.item()

        loss_v, acc_v = _cls_quick_val(model, val_ld, device, crit)
        print(f"[cls epoch {epoch}] val_loss={loss_v:.4f} val_acc={acc_v:.4f} "
              f"| train_loss={run/len(train_ld):.4f} | {time.time()-t0:.0f}s")

        ckpt = dict(epoch=epoch, model_state=model.state_dict(),
                    config=vars(cfg), val_acc=acc_v)
        torch.save(ckpt, os.path.join(out_dir, "cls_last.pth"))
        if acc_v > best["acc"]:
            best.update(acc=acc_v, epoch=epoch)
            torch.save(ckpt, os.path.join(out_dir, "cls_best.pth"))
        print(f"  best val_acc={best['acc']:.4f} @epoch {best['epoch']}")
    return os.path.join(out_dir, "cls_best.pth")


def main():
    cfg = parse_overrides(sys.argv[1:], Config())
    import argparse
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--stage", choices=["seg", "cls", "both"], default="both")
    a, _ = ap.parse_known_args(sys.argv[1:])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}  | data_root={cfg.data_root}")

    if a.stage in ("seg", "both"):
        train_seg(cfg, device)
    if a.stage in ("cls", "both"):
        train_cls(cfg, device)
    print("[done] 训练完成。")


if __name__ == "__main__":
    main()
