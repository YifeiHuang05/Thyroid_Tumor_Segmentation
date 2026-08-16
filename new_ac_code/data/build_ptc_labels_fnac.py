"""
[备份] 使用 FNAC 的默认 PTC 标签生成器（原始版本）。
仅当用户需要“组织学 + FNAC 共同判定 PTC”时使用。
当前主用版本为 build_ptc_labels.py（仅用 Histopathology，不用 FNAC）。
"""
import os
import csv
import json
import glob


def _norm(v):
    return (v or "").strip().lower()


def build_ptc_labels(data_root, out_csv):
    ann_path = os.path.join(data_root, "train_annotations.json")
    with open(ann_path, "r", encoding="utf-8") as f:
        ann = json.load(f)
    pats = ann["info"] if isinstance(ann, dict) and "info" in ann else ann

    # 患者 -> 是否良性（读 labels_orig）
    patient_benign = {}
    for fp in glob.glob(os.path.join(data_root, "labels_orig", "*.json")):
        name = os.path.basename(fp).replace(".json", "")
        pid = name.split("_")[0]
        try:
            d = json.load(open(fp, "r", encoding="utf-8"))
        except Exception:
            continue
        lab = _norm(d.get("label"))
        if pid not in patient_benign:
            patient_benign[pid] = (lab == "lanhtinh")

    rows = []
    for pid, rec in pats.items():
        n1 = rec.get("nodule_1") or {}
        hist = _norm(n1.get("Histopathology"))
        fna = _norm(n1.get("FNAC"))
        is_ptc = ("papillary" in hist and "thyroid" in hist) or ("papillary" in fna)
        if is_ptc:
            ptc = 1
        elif patient_benign.get(pid, False):
            ptc = 0
        else:
            ptc = -1
        rows.append((pid, ptc))

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["patient_id", "ptc"])
        w.writerows(rows)

    n_pos = sum(1 for _, p in rows if p == 1)
    n_neg = sum(1 for _, p in rows if p == 0)
    n_unk = sum(1 for _, p in rows if p == -1)
    print(f"[build_ptc_labels] 写出 {len(rows)} 名患者 -> {out_csv}")
    print(f"  PTC=1: {n_pos}   PTC=0: {n_neg}   排除(-1): {n_unk}")
    return rows


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    build_ptc_labels(root, os.path.join(root, "ptc_labels.csv"))
