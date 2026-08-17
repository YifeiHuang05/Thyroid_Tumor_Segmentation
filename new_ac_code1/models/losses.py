"""损失函数：
分割用 Dice+BCE（应对小目标/强不平衡）；
分类（任务 C，多任务头）：
  - multi_head_loss：对 bm / ptc / fnac / tirands 各头分别计算交叉熵；
    每个头只在其“有效样本”（标签 >= 0）上计算，可各自按类别频率加权。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_coeff(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    inter = (probs * targets).sum(dim=(2, 3))
    union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    return (2 * inter + eps) / (union + eps)


def seg_loss(logits, targets, bce_weight=0.5):
    """logits/targets: (B,1,H,W) float。"""
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    dice = 1.0 - dice_coeff(logits, targets).mean()
    return bce_weight * bce + (1.0 - bce_weight) * dice


def compute_class_weights(counts):
    """counts: list[int]；返回 Tensor 权重 w_c = total / (C * count_c)（缓解不平衡）。
    任一类别为 0 则返回 None（无法加权）。"""
    total = sum(counts)
    if total == 0 or any(c == 0 for c in counts):
        return None
    w = [total / (len(counts) * c) for c in counts]
    return torch.tensor(w, dtype=torch.float32)


def multi_head_loss(logits_dict, targets_dict, valid_dict, weight_dict=None):
    """多任务头分类损失（通用）。
    logits_dict : {name: (N, C)}
    targets_dict: {name: (N,) long}（无效样本为 -1，会被 valid_dict 排除）
    valid_dict  : {name: (N,) bool}（哪些样本该头参与）
    weight_dict : {name: Tensor 或 None}（类别权重）
    返回 (total, per_head_dict)。某头无有效样本时该头损失记 0。
    """
    total = torch.zeros((), device=next(iter(logits_dict.values())).device)
    per = {}
    for name, lg in logits_dict.items():
        t = targets_dict[name].long()
        v = valid_dict[name]
        if v.any():
            w = None
            if weight_dict and weight_dict.get(name) is not None:
                w = weight_dict[name].to(lg.device)
            loss = F.cross_entropy(lg[v], t[v], weight=w)
        else:
            loss = torch.zeros((), device=lg.device)
        per[name] = loss
        total = total + loss
    return total, per


def compute_multi_weights(ds, heads):
    """根据训练集统计每个头的类别频率，返回 {name: weight Tensor 或 None}。
    heads: [(name, num_classes, names), ...]"""
    counts = {name: [0] * n for name, n, _ in heads}
    for s in ds.samples:
        for name, n, _ in heads:
            v = int(s[name])
            if 0 <= v < n:
                counts[name][v] += 1
    out = {}
    for name, n, _ in heads:
        w = compute_class_weights(counts[name])
        out[name] = w
        if w is not None:
            print(f"[cls] {name} 计数={counts[name]} -> weight={w.tolist()}")
        else:
            print(f"[cls] {name} 计数={counts[name]} -> 不加权（存在空类或全缺）")
    return out

