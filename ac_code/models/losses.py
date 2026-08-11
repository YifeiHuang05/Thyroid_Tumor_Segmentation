"""损失函数：分割用 Dice+BCE（应对小目标/强不平衡），PTC 用带 pos_weight 的 BCE。"""
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


class PTCBCE(nn.Module):
    def __init__(self, pos_weight=None):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        # logits/targets: (N,) 或 (N,1)
        if targets.dim() == 1:
            targets = targets.unsqueeze(1)
        if logits.dim() == 1:
            logits = logits.unsqueeze(1)
        pw = self.pos_weight.to(logits.device) if self.pos_weight is not None else None
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw)
