"""评估指标：
任务 A（分割）：Dice / IoU / HD95（在原始分辨率上计算）
任务 C（多分类）：Accuracy / Balanced Accuracy / Macro-F1 / Micro-F1（患者级聚合）+ 混淆矩阵
"""
import numpy as np
from scipy.ndimage import distance_transform_edt
from collections import defaultdict

try:
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score,
        f1_score, precision_recall_fscore_support, confusion_matrix,
    )
except Exception:  # 评估时若无 sklearn 给出明确报错
    accuracy_score = balanced_accuracy_score = f1_score = None
    confusion_matrix = None


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


# ---------------- 分类（三分类，患者级） ----------------
def aggregate_patient(probs, labels, patient_ids, num_classes=3):
    """probs: 每个样本的长度为 C 的概率数组（或单值旧接口兼容）；labels: 整数；按患者聚合。"""
    by_p = defaultdict(list)
    for p, lab, pid in zip(probs, labels, patient_ids):
        by_p[pid].append((np.asarray(p, dtype=float), int(lab)))
    agg = {}
    for pid, items in by_p.items():
        prob = np.mean([x[0] for x in items], axis=0)   # 长度 C
        lab = items[0][1]
        agg[pid] = (prob, lab)
    return agg


def classification_report_patient(probs, labels, patient_ids,
                                 num_classes=3, class_names=None):
    """probs: list of length-C 概率数组；labels: list of int（患者级一致）。"""
    if class_names is None:
        class_names = [f"c{i}" for i in range(num_classes)]
    agg = aggregate_patient(probs, labels, patient_ids, num_classes)
    y_true = np.array([v[1] for v in agg.values()])
    y_prob = np.stack([v[0] for v in agg.values()], axis=0)
    y_pred = np.argmax(y_prob, axis=1)
    if accuracy_score is None:
        raise RuntimeError("请安装 scikit-learn 以计算分类指标")
    acc = accuracy_score(y_true, y_pred)
    bal = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)

    per_class = {}
    for i in range(num_classes):
        per_class[class_names[i]] = dict(precision=p[i], recall=r[i], f1=f1[i])

    cm = None
    if confusion_matrix is not None:
        cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes))).tolist()

    return dict(
        n_patients=len(agg),
        accuracy=acc,
        balanced_accuracy=bal,
        macro_f1=macro_f1,
        micro_f1=micro_f1,
        per_class=per_class,
        confusion_matrix=cm,
        class_names=class_names,
        y_true=y_true,
        y_pred=y_pred,
    )
