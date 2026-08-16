"""
从 train_annotations.json + labels_orig 生成默认的三分类标签 CSV（cls_labels.csv）。

分类目标（任务 C 改为三分类）：
  0 = 良性（benign）
  1 = 恶性-乳头状（malignant papillary）
  2 = 恶性-非乳头状（malignant non-papillary）
 -1 = 排除（训练/评估时不使用）

【标签口径 B（最可行的默认口径）】
  - 良性：labels_orig 标为 lanhtinh（图像级良性标注，可靠）。
  - 恶性：labels_orig 标为 actinh（图像级恶性标注）。
      * 其中 Histopathology 含 "papillary thyroid" -> 类 1（恶性乳头状，组织学确诊）
      * 其余（含“无组织学记录”的恶性）-> 类 2（恶性非乳头状）
  - 矛盾处理：标 benign 但组织学为乳头状者 -> 归为类 1（乳头状恶性优先，医学上乳头状即恶性）。
  - 既非良性也非恶性的患者 -> 排除(-1)。

⚠️ 重要说明：数据集中“恶性-非乳头状”的可靠组织学金标准极少（仅约 10 例有非乳头状恶性记录），
其余 374 例类 2 实际来自“恶性(actinh)但未确认乳头状”，可能混入少量真实乳头状但无活检者。
这是对现有标注最合理的默认；若后续有人提供更精确的 PTC/亚型区分标签，
直接覆盖 cls_labels.csv（保持 patient_id,cls 两列）即可被自动读取，无需改代码。
"""
import os
import csv
import json
import glob


def _norm(v):
    return (v or "").strip().lower()


def build_cls_labels(data_root, out_csv):
    ann_path = os.path.join(data_root, "train_annotations.json")
    with open(ann_path, "r", encoding="utf-8") as f:
        ann = json.load(f)
    pats = ann["info"] if isinstance(ann, dict) and "info" in ann else ann

    # 患者级 labels_orig 良性/恶性（聚合图像级标签）
    patient_lo = {}
    for fp in glob.glob(os.path.join(data_root, "labels_orig", "*.json")):
        name = os.path.basename(fp).replace(".json", "")
        pid = name.split("_")[0]
        try:
            d = json.load(open(fp, "r", encoding="utf-8"))
        except Exception:
            continue
        lab = _norm(d.get("label"))
        if pid not in patient_lo:
            patient_lo[pid] = lab  # 'lanhtinh' / 'actinh' / ''

    rows = []
    for pid, rec in pats.items():
        n1 = rec.get("nodule_1") or {}
        hist = _norm(n1.get("Histopathology"))
        is_pap = ("papillary" in hist and "thyroid" in hist)
        lo = patient_lo.get(pid, "")
        if lo == "lanhtinh":
            cls = 1 if is_pap else 0          # 矛盾：标良性但组织学乳头状 -> 类1
        elif lo == "actinh":
            cls = 1 if is_pap else 2          # 恶性中，乳头状=1，其余(含无组织学)=2
        else:
            cls = -1                          # 既非良性也非恶性 -> 排除
        rows.append((pid, cls))

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["patient_id", "cls"])
        w.writerows(rows)

    cnt = {k: sum(1 for _, c in rows if c == k) for k in (0, 1, 2)}
    n_ex = sum(1 for _, c in rows if c == -1)
    print(f"[build_cls_labels] 写出 {len(rows)} 名患者 -> {out_csv}")
    print(f"  良性(0)={cnt[0]}  恶性乳头状(1)={cnt[1]}  恶性非乳头状(2)={cnt[2]}  排除(-1)={n_ex}")
    return rows


if __name__ == "__main__":
    # ac_code/data/build_cls_labels.py -> 上三级为数据根目录(train)
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    build_cls_labels(root, os.path.join(root, "cls_labels.csv"))
