"""损失函数：
分割用 Dice+BCE（应对小目标/强不平衡）；
三分类（任务 C）用带 class_weight 的 CrossEntropy（应对类别不平衡）。
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


class ClsCE(nn.Module):
    """三分类交叉熵，支持按类别频率加权的 class_weight（缓解不平衡）。"""
    def __init__(self, class_weight=None):
        super().__init__()
        self.class_weight = class_weight  # Tensor length=C 或 None

    def forward(self, logits, targets):
        # logits: (N, C) ; targets: (N,) 整数类别
        w = self.class_weight.to(logits.device) if self.class_weight is not None else None
        return F.cross_entropy(logits, targets.long(), weight=w)
