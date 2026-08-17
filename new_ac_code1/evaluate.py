"""
评估：
  evaluate_seg   -> 任务 A：Dice / IoU / HD95（原始分辨率）
  evaluate_hier  -> 任务 C：完整级联（seg 预测 mask -> 裁剪结节 -> 多任务分类），患者级
                    头：bm(良性/恶性) / ptc(恶性内部 PTC/非PTC) / fnac(Bethesda 三分类) / tirands(TI-RADS 五分类)
                    每个头输出：Accuracy / Balanced Acc / Macro-F1 / Micro-F1 / 混淆矩阵
                    （crop_from="pred" 用预测 mask，与推理一致；crop_from="gt" 用 GT mask 做上限参考）
  _collate_seg / _collate_cls / _cls_quick_val：供 train.py 复用。

作为脚本运行：python evaluate.py  -> 加载 runs/seg_best.pth 与 cls_best.pth，输出各头报告。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from config import Config, parse_overrides
from data.thyroid_dataset import ThyroidSegDataset, inverse_letterbox
from data.crop_utils import mask_to_bbox, crop_patch
from models.unet import SegUNet
from models.classifier import build_classifier, get_heads
from models.metrics import dice_score, iou_score, hd95, classification_report_patient


# ---------------- collate ----------------
def _collate_seg(batch):
    return dict(
        image=torch.stack([torch.from_numpy(b["image"]) for b in batch]),
        mask=torch.stack([torch.from_numpy(b["mask"]) for b in batch]),
        img_raw=[b["img_raw"] for b in batch],
        bm=torch.tensor([b["bm"] for b in batch], dtype=torch.long),
        ptc=torch.tensor([b["ptc"] for b in batch], dtype=torch.long),
        fnac=torch.tensor([b["fnac"] for b in batch], dtype=torch.long),
        tirands=torch.tensor([b["tirands"] for b in batch], dtype=torch.long),
        has_ptc=torch.tensor([b["has_ptc"] for b in batch], dtype=torch.bool),
        has_bm=torch.tensor([b["has_bm"] for b in batch], dtype=torch.bool),
        has_fnac=torch.tensor([b["has_fnac"] for b in batch], dtype=torch.bool),
        has_tirands=torch.tensor([b["has_tirands"] for b in batch], dtype=torch.bool),
        patient_id=[b["patient_id"] for b in batch],
        mask_path=[b["mask_path"] for b in batch],
        params=[b["params"] for b in batch],
    )


def _collate_cls(batch):
    return dict(
        patch=torch.stack([torch.from_numpy(b["patch"]) for b in batch]),
        bm=torch.tensor([b["bm"] for b in batch], dtype=torch.long),
        ptc=torch.tensor([b["ptc"] for b in batch], dtype=torch.long),
        fnac=torch.tensor([b["fnac"] for b in batch], dtype=torch.long),
        tirands=torch.tensor([b["tirands"] for b in batch], dtype=torch.long),
        has_ptc=torch.tensor([b["has_ptc"] for b in batch], dtype=torch.bool),
        has_bm=torch.tensor([b["has_bm"] for b in batch], dtype=torch.bool),
        has_fnac=torch.tensor([b["has_fnac"] for b in batch], dtype=torch.bool),
        has_tirands=torch.tensor([b["has_tirands"] for b in batch], dtype=torch.bool),
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
def _cls_quick_val(model, loader, device):
    model.eval()
    heads = list(model.heads.keys())
    correct = {n: 0 for n in heads}
    total = {n: 0 for n in heads}
    softmax = torch.nn.functional.softmax
    with torch.no_grad():
        for b in loader:
            patch = b["patch"].to(device)
            logits = model(patch)
            for name in heads:
                t = b[name].to(device).long()
                v = b[f"has_{name}"].to(device)
                if v.any():
                    p = softmax(logits[name][v], dim=1).argmax(1)
                    correct[name] += (p == t[v]).sum().item()
                    total[name] += int(v.sum().item())
    return {n: (correct[n] / total[n] if total[n] else float("nan")) for n in heads}


# ---------------- 完整级联多任务分类评估 ----------------
def evaluate_hier(seg_model, cls_model, loader, device, cfg, crop_from="pred"):
    """返回 dict: {name: report} 对每个头（bm/ptc/fnac/tirands）。"""
    seg_model.eval()
    cls_model.eval()
    heads = get_heads(cfg)
    mean = np.array(cfg.mean, np.float32).reshape(3, 1, 1)
    std = np.array(cfg.std, np.float32).reshape(3, 1, 1)
    softmax = torch.nn.functional.softmax

    # {name: (probs, labels, pids)}
    buf = {name: ([], [], []) for name, _, _ in heads}
    with torch.no_grad():
        for b in loader:
            img = b["image"].to(device)
            seg_logits = seg_model(img)
            seg_prob = torch.sigmoid(seg_logits).cpu().numpy()      # B,1,H,W
            gt_mask = b["mask"].cpu().numpy()                        # B,1,H,W 0/1
            img_raw = b["img_raw"]                                   # list of uint8 HWC
            bm = b["bm"].numpy()
            ptc = b["ptc"].numpy()
            fnac = b["fnac"].numpy()
            tir = b["tirands"].numpy()
            has = {
                "bm": b["has_bm"].numpy(),
                "ptc": b["has_ptc"].numpy(),
                "fnac": b["has_fnac"].numpy(),
                "tirands": b["has_tirands"].numpy(),
            }
            labels = {"bm": bm, "ptc": ptc, "fnac": fnac, "tirands": tir}
            for i in range(seg_prob.shape[0]):
                if crop_from == "pred":
                    m = (seg_prob[i, 0] > 0.5).astype(np.float32)
                else:
                    m = gt_mask[i, 0].astype(np.float32)
                bbox = mask_to_bbox(m, cfg.margin)
                if bbox is None:                                     # 预测为空：各头概率置 0（argmax->类0）
                    plogits = {name: np.zeros(n, dtype=np.float32) for name, n, _ in heads}
                else:
                    patch = crop_patch(img_raw[i], bbox, cfg.patch_size)
                    patch_t = (patch.astype(np.float32).transpose(2, 0, 1) / 255.0 - mean) / std
                    t = torch.from_numpy(patch_t).unsqueeze(0).to(device)
                    out = cls_model(t)
                    plogits = {name: softmax(out[name], dim=1).cpu().numpy()[0].astype(np.float32)
                               for name in out}

                pid = b["patient_id"][i]
                for name, n, _ in heads:
                    if has[name][i]:
                        buf[name][0].append(plogits[name])
                        buf[name][1].append(int(labels[name][i]))
                        buf[name][2].append(pid)

    reports = {}
    for name, n, names in heads:
        probs, labels_, pids = buf[name]
        if probs:
            rep = classification_report_patient(probs, labels_, pids, num_classes=n, class_names=names)
            rep["crop_from"] = crop_from
            reports[name] = rep
        else:
            reports[name] = None
    return reports


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
    heads = get_heads(mcfg)
    model = build_classifier(patch_size=mcfg.get("patch_size", 224),
                             backbone=mcfg.get("cls_backbone", "efficientnet_b3"),
                             pretrained=False, use_timm=mcfg.get("cls_use_timm", True),
                             heads=heads)
    model.load_state_dict(sd["model_state"])
    return model


def _print_report(title, rep):
    if rep is None:
        print(f"\n===== {title} =====  无可用样本（验证集不含该子集）")
        return
    print(f"\n===== {title} =====  (crop_from={rep.get('crop_from')})")
    print(f"  患者数={rep['n_patients']}  Accuracy={rep['accuracy']:.4f}  "
          f"BalancedAcc={rep['balanced_accuracy']:.4f}  "
          f"MacroF1={rep['macro_f1']:.4f}  MicroF1={rep['micro_f1']:.4f}")
    for name, m in rep["per_class"].items():
        print(f"  {name:14s} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")
    if rep.get("confusion_matrix") is not None:
        names = rep["class_names"]
        print("  混淆矩阵（行=真值, 列=预测）:")
        print("        " + "  ".join(f"{n[:8]:>8s}" for n in names))
        for i, row in enumerate(rep["confusion_matrix"]):
            print(f"  {names[i][:8]:>8s} " + "  ".join(f"{v:>8d}" for v in row))


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
        reports = evaluate_hier(seg_model, cls_model, val_ld, device, cfg, crop_from=cf)
        _print_report(f"任务 C-bm 良性/恶性（crop_from={cf}）", reports.get("bm"))
        _print_report(f"任务 C-ptc PTC/非PTC（crop_from={cf}）", reports.get("ptc"))
        _print_report(f"任务 C-fnac Bethesda 三分类（crop_from={cf}）", reports.get("fnac"))
        _print_report(f"任务 C-tirands TI-RADS 五分类（crop_from={cf}）", reports.get("tirands"))


if __name__ == "__main__":
    main()
