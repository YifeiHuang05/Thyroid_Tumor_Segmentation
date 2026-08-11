"""评估指标：
任务 A（分割）：Dice / IoU / HD95（在原始分辨率上计算）
任务 C（PTC）：Accuracy / Balanced Accuracy / Macro-F1（患者级聚合）
"""
import numpy as np
from scipy.ndimage import distance_transform_edt
from collections import defaultdict

try:
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score,
        f1_score, precision_recall_fscore_support,
    )
except Exception:  # 评估时若无 sklearn 给出明确报错
    accuracy_score = balanced_accuracy_score = f1_score = None


# ---------------- 分割 ----------------
def dice_score(gt, pred):
    gt = gt.astype(bool)
    pred = pred.astype(bool)
    if gt.sum() + pred.sum() == 0:
        return 1.0
    inter = np.logical_and(gt, pred).sum()
    union = gt.sum() + pred.sum()
    return 2.0 * inter / float(union) if union > 0 else 0.0


def iou_score(gt, pred):
    gt = gt.astype(bool)
    pred = pred.astype(bool)
    if gt.sum() + pred.sum() == 0:
        return 1.0
    inter = np.logical_and(gt, pred).sum()
    union = np.logical_or(gt, pred).sum()
    return inter / float(union) if union > 0 else 0.0


def hd95(gt, pred, max_dist=373.0):
    """95 百分位 Hausdorff 距离（像素）。全空/全无时返回 max_dist。"""
    gt = gt.astype(bool)
    pred = pred.astype(bool)
    if gt.sum() == 0 and pred.sum() == 0:
        return 0.0
    if gt.sum() == 0 or pred.sum() == 0:
        return max_dist
    dt_gt = distance_transform_edt(~gt)
    dt_pred = distance_transform_edt(~pred)
    sds = np.concatenate([dt_pred[gt], dt_gt[pred]])
    if len(sds) == 0:
        return 0.0
    return float(np.percentile(sds, 95))


# ---------------- 分类（PTC，患者级） ----------------
def aggregate_patient(preds, labels, patient_ids):
    """preds/labels: list；按患者聚合图像级预测（取均值）后投票。"""
    by_p = defaultdict(list)
    for p, lab, pid in zip(preds, labels, patient_ids):
        by_p[pid].append((float(p), float(lab)))
    agg = {}
    for pid, items in by_p.items():
        prob = np.mean([x[0] for x in items])
        lab = items[0][1]
        agg[pid] = (prob, lab)
    return agg


def classification_report_patient(preds, labels, patient_ids):
    agg = aggregate_patient(preds, labels, patient_ids)
    y_true = np.array([v[1] for v in agg.values()])
    y_prob = np.array([v[0] for v in agg.values()])
    y_pred = (y_prob >= 0.5).astype(int)
    if accuracy_score is None:
        raise RuntimeError("请安装 scikit-learn 以计算分类指标")
    acc = accuracy_score(y_true, y_pred)
    bal = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    return dict(
        n_patients=len(agg),
        accuracy=acc,
        balanced_accuracy=bal,
        macro_f1=macro_f1,
        per_class=dict(
            p0=dict(precision=p[0], recall=r[0], f1=f1[0]),
            p1=dict(precision=p[1], recall=r[1], f1=f1[1]),
        ),
        y_true=y_true,
        y_pred=y_pred,
    )
