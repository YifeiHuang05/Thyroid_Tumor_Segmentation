"""
从 train_annotations.json + labels_orig 生成默认的分类标签 CSV（cls_labels.csv）。

分类目标（任务 C，全部基于任务 A 裁剪出的结节 patch，多任务头）：
  ── bm  ：良性 / 恶性（所有结节级标签样本均参与）
      bm = 0  良性（labels_orig 标 lanhtinh）
      bm = 1  恶性（labels_orig 标 actinh）
      bm = -1 排除（既非良性也非恶性）
  ── ptc ：在“恶性”内部区分 PTC/非 PTC —— 仅使用明确病理金标准，不使用 FNAC：
      ptc = 1  恶性 且 Histopathology 含 "papillary thyroid"（组织学确诊乳头状）
      ptc = 0  恶性 且 有明确非乳头状病理（如 medullary / follicular / hurthle 等）
      ptc = -1 良性（不适用），或 恶性但无明确病理记录（不参与 PTC 训练/评估）
  ── fnac：FNAC（细针穿刺细胞学）三分类（Bethesda），按参赛须知合并：
      fnac = 0  Bethesda II            （原始 FNAC 值 == 2）
      fnac = 1  Bethesda III-IV        （原始 FNAC 值 ∈ {3, 4}）
      fnac = 2  Bethesda V-VI          （原始 FNAC 值 ∈ {5, 6}）
      fnac = -1 Bethesda I 或 缺失（不参与 FNAC 训练/评估）
  ── tirands：TI-RADS 五分类（原始 1-5 -> 0-4）：
      tirands = 0..4  对应 TR1..TR5
      tirands = -1    缺失 / 非法（不参与 TI-RADS 训练/评估）

【重要说明】
  - PTC 标签严格只取“良恶性已确证(actinh) + 有病理记录”的结节；FNAC 完全不参与 PTC。
  - FNAC / TI-RADS 是患者级标签（每患者单结节），直接用于预测对应头。
  - 若后续有人提供更精确的子类型/病理区分标签，直接覆盖 cls_labels.csv
    （保持 patient_id,bm,ptc,fnac,tirands 五列）即可被自动读取，无需改代码。
"""
import os
import csv
import json
import glob


def _norm(v):
    if v is None:
        return ""
    if isinstance(v, (int, float)):
        return str(v)
    return str(v).strip()


def _fnac_to_class(raw):
    """Bethesda 三分类映射；返回 0/1/2/-1。"""
    s = _norm(raw)
    if s == "":
        return -1
    # 取前导整数（'6 (papillary thyroid carcinoma)' -> 6）
    lead = ""
    for ch in s:
        if ch.isdigit():
            lead += ch
        else:
            break
    if lead == "":
        return -1
    v = int(lead)
    if v == 2:
        return 0          # Bethesda II
    elif v in (3, 4):
        return 1          # Bethesda III-IV
    elif v in (5, 6):
        return 2          # Bethesda V-VI
    return -1             # Bethesda I(=1) 或未知 -> 排除


def _tirads_to_class(raw):
    """TI-RADS 1-5 -> 0-4；缺失/非法 -> -1。"""
    s = _norm(raw)
    if s == "":
        return -1
    try:
        v = int(s)
    except ValueError:
        return -1
    if 1 <= v <= 5:
        return v - 1
    return -1


def build_cls_labels(data_root, out_csv):
    ann_path = os.path.join(data_root, "train_annotations.json")
    with open(ann_path, "r", encoding="utf-8") as f:
        ann = json.load(f)
    pats = ann["info"] if isinstance(ann, dict) and "info" in ann else ann

    # 患者级 labels_orig 良性/恶性（聚合图像级标签，取首个出现值）
    patient_lo = {}
    for fp in glob.glob(os.path.join(data_root, "labels_orig", "*.json")):
        name = os.path.basename(fp).replace(".json", "")
        pid = name.split("_")[0]
        try:
            d = json.load(open(fp, "r", encoding="utf-8"))
        except Exception:
            continue
        lab = _norm(d.get("label")).lower()
        if pid not in patient_lo:
            patient_lo[pid] = lab  # 'lanhtinh' / 'actinh' / ''

    rows = []
    for pid, rec in pats.items():
        n1 = rec.get("nodule_1") or {}
        hist = _norm(n1.get("Histopathology")).lower()
        lo = patient_lo.get(pid, "")
        is_pap = ("papillary" in hist and "thyroid" in hist)

        if lo == "lanhtinh":
            bm = 0
            ptc = -1                                  # 良性：PTC 不适用
        elif lo == "actinh":
            bm = 1
            if is_pap:
                ptc = 1                               # 恶性 + 乳头状病理
            elif hist:
                ptc = 0                               # 恶性 + 明确非乳头状病理
            else:
                ptc = -1                              # 恶性 + 无明确病理记录
        else:
            bm = -1
            ptc = -1                                  # 既非良性也非恶性 -> 排除

        fnac = _fnac_to_class(n1.get("FNAC"))
        tirands = _tirads_to_class(n1.get("TIRADS"))

        rows.append((pid, bm, ptc, fnac, tirands))

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["patient_id", "bm", "ptc", "fnac", "tirands"])
        w.writerows(rows)

    # ---- 统计 ----
    def _count(idx):
        c = {}
        for r in rows:
            v = r[idx]
            c[v] = c.get(v, 0) + 1
        return c

    bm_c = _count(1)
    ptc_c = _count(2)
    fnac_c = _count(3)
    tir_c = _count(4)
    print(f"[build_cls_labels] 写出 {len(rows)} 名患者 -> {out_csv}")
    print(f"  [bm]     良性(0)={bm_c.get(0,0)}  恶性(1)={bm_c.get(1,0)}  排除(-1)={bm_c.get(-1,0)}")
    print(f"  [ptc]    PTC(1)={ptc_c.get(1,0)}  非PTC(0)={ptc_c.get(0,0)}  不参与(-1)={ptc_c.get(-1,0)}")
    print(f"  [fnac]   II(0)={fnac_c.get(0,0)}  III-IV(1)={fnac_c.get(1,0)}  V-VI(2)={fnac_c.get(2,0)}  排除(-1)={fnac_c.get(-1,0)}")
    print(f"  [tirands]TR1..TR5={[tir_c.get(k,0) for k in range(5)]}  排除(-1)={tir_c.get(-1,0)}")
    print(f"  ⚠ PTC 负类（非乳头状恶性）仅 {ptc_c.get(0,0)} 名患者，极度稀缺；不使用 FNAC，仅依赖明确病理。")
    return rows


if __name__ == "__main__":
    # new_ac_code/data/build_cls_labels.py -> 上三级为数据根目录(train)
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    build_cls_labels(root, os.path.join(root, "cls_labels.csv"))
