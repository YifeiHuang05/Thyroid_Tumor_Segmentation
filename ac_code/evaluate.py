"""
评估：
  evaluate_seg   -> 任务 A：Dice / IoU / HD95（原始分辨率）
  evaluate_ptc   -> 任务 C：完整级联（seg 预测 mask -> 裁剪结节 -> 分类），患者级
                    Macro-F1 / Balanced Accuracy / Accuracy（crop_from="pred" 用预测 mask，
                    与推理一致；也可 crop_from="gt" 用 GT mask 做上限参考）
_collate_seg / _collate_cls / _cls_quick_val：供 train.py 复用。

作为脚本运行：python evaluate.py  -> 加载 runs/seg_best.pth 与 cls_best.pth，输出两份报告。
"""
import os
import sys
import csv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from config import Config, parse_overrides
from data.thyroid_dataset import ThyroidSegDataset, inverse_letterbox
from data.crop_utils import mask_to_bbox, crop_patch
from models.unet import SegUNet
from models.classifier import build_classifier
from models.metrics import dice_score, iou_score, hd95, classification_report_patient


# ---------------- collate ----------------
def _collate_seg(batch):
    return dict(
        image=torch.stack([torch.from_numpy(b["image"]) for b in batch]),
        mask=torch.stack([torch.from_numpy(b["mask"]) for b in batch]),
        img_raw=[b["img_raw"] for b in batch],
        ptc=torch.tensor([b["ptc"] for b in batch], dtype=torch.float32),
        has_ptc=torch.tensor([b["has_ptc"] for b in batch], dtype=torch.bool),
        patient_id=[b["patient_id"] for b in batch],
        mask_path=[b["mask_path"] for b in batch],
        params=[b["params"] for b in batch],
    )


def _collate_cls(batch):
    return dict(
        patch=torch.stack([torch.from_numpy(b["patch"]) for b in batch]),
        ptc=torch.tensor([b["ptc"] for b in batch], dtype=torch.float32),
        has_ptc=torch.tensor([b["has_ptc"] for b in batch], dtype=torch.bool),
        patient_id=[b["patient_id"] for b in batch],
    )


# ---------------- 分割评估 ----------------
def evaluate_seg(model, loader, device):
    model.eval()
    dices, ious, hd95s = [], [], []
    with torch.no_grad():
        for b in loader:
            seg = model(b["image"].to(device))
            prob = torch.sigmoid(seg).cpu().numpy()
            for i in range(prob.shape[0]):
                pred = (prob[i, 0] > 0.5).astype(np.uint8)
                pred_o = inverse_letterbox(pred, b["params"][i],
                                           b["params"][i]["orig_h"], b["params"][i]["orig_w"])
                gt = (np.array(Image.open(b["mask_path"][i]).convert("L")) > 127).astype(np.uint8)
                dices.append(dice_score(gt, pred_o))
                ious.append(iou_score(gt, pred_o))
                hd95s.append(hd95(gt, pred_o))
    return dict(
        dice=float(np.mean(dices)),
        iou=float(np.mean(ious)),
        hd95=float(np.mean(hd95s)),
        n=len(dices),
    )


# ---------------- 分类快速校验（GT patch，训练期用） ----------------
def _cls_quick_val(model, loader, device, crit):
    model.eval()
    correct = n = 0
    loss_sum = 0.0
    with torch.no_grad():
        for b in loader:
            patch = b["patch"].to(device)
            ptc = b["ptc"].to(device).unsqueeze(1)
            logits = model(patch)
            loss = crit(logits, ptc)
            loss_sum += loss.item() * patch.size(0)
            pred = (torch.sigmoid(logits) >= 0.5).float()
            correct += (pred == ptc).sum().item()
            n += patch.size(0)
    return loss_sum / max(1, n), correct / max(1, n)


# ---------------- 完整级联 PTC 评估 ----------------
def evaluate_ptc(seg_model, cls_model, loader, device, cfg, crop_from="pred"):
    seg_model.eval()
    cls_model.eval()
    mean = np.array(cfg.mean, np.float32).reshape(3, 1, 1)
    std = np.array(cfg.std, np.float32).reshape(3, 1, 1)
    preds, labels, pids = [], [], []
    with torch.no_grad():
        for b in loader:
            img = b["image"].to(device)
            seg_logits = seg_model(img)
            seg_prob = torch.sigmoid(seg_logits).cpu().numpy()      # B,1,H,W
            gt_mask = b["mask"].cpu().numpy()                        # B,1,H,W 0/1
            img_raw = b["img_raw"]                                   # list of uint8 HWC
            ptc = b["ptc"].numpy()
            has = b["has_ptc"].numpy()
            for i in range(seg_prob.shape[0]):
                if not has[i]:
                    continue
                if crop_from == "pred":
                    m = (seg_prob[i, 0] > 0.5).astype(np.float32)
                else:
                    m = gt_mask[i, 0].astype(np.float32)
                bbox = mask_to_bbox(m, cfg.margin)
                if bbox is None:                                     # 预测为空：当作低概率负类
                    preds.append(0.0)
                else:
                    patch = crop_patch(img_raw[i], bbox, cfg.patch_size)
                    patch_t = (patch.astype(np.float32).transpose(2, 0, 1) / 255.0 - mean) / std
                    t = torch.from_numpy(patch_t).unsqueeze(0).to(device)
                    preds.append(torch.sigmoid(cls_model(t)).item())
                labels.append(float(ptc[i]))
                pids.append(b["patient_id"][i])
    if not preds:
        return None
    rep = classification_report_patient(preds, labels, pids)
    rep["crop_from"] = crop_from
    return rep


# ---------------- 脚本入口 ----------------
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = cfg.out_path()
    seg_ckpt = os.path.join(out_dir, "seg_best.pth")
    cls_ckpt = os.path.join(out_dir, "cls_best.pth")
    if not os.path.exists(seg_ckpt):
        raise FileNotFoundError(f"找不到分割模型 {seg_ckpt}，请先运行 train.py")
    if not os.path.exists(cls_ckpt):
        raise FileNotFoundError(f"找不到分类模型 {cls_ckpt}，请先运行 train.py")

    val_ds = ThyroidSegDataset(cfg, split="val")
    val_ld = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=cfg.num_workers, collate_fn=_collate_seg, pin_memory=True)

    seg_model = _load_seg(cfg, seg_ckpt).to(device)
    cls_model = _load_cls(cfg, cls_ckpt).to(device)

    seg_m = evaluate_seg(seg_model, val_ld, device)
    print("\n===== 任务 A（分割）验证集 =====")
    print(f"  Dice={seg_m['dice']:.4f}  IoU={seg_m['iou']:.4f}  HD95={seg_m['hd95']:.2f}  (n={seg_m['n']})")

    for cf in ("pred", "gt"):
        rep = evaluate_ptc(seg_model, cls_model, val_ld, device, cfg, crop_from=cf)
        if rep is None:
            print(f"\n===== 任务 C（PTC，crop_from={cf}）=====  验证集无标签样本")
            continue
        print(f"\n===== 任务 C（PTC，crop_from={cf}）验证集 =====")
        print(f"  患者数={rep['n_patients']}  Accuracy={rep['accuracy']:.4f}  "
              f"BalancedAcc={rep['balanced_accuracy']:.4f}  MacroF1={rep['macro_f1']:.4f}")
        p0, p1 = rep["per_class"]["p0"], rep["per_class"]["p1"]
        print(f"  非PTC: P={p0['precision']:.3f} R={p0['recall']:.3f} F1={p0['f1']:.3f}")
        print(f"  PTC  : P={p1['precision']:.3f} R={p1['recall']:.3f} F1={p1['f1']:.3f}")


if __name__ == "__main__":
    main()
