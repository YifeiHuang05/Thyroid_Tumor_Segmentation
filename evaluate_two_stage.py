from __future__ import annotations

import csv
from datetime import datetime
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from ultralytics import YOLO

try:
    from scipy.ndimage import binary_erosion
    from scipy.spatial.distance import cdist

    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


ROOT = Path(__file__).resolve().parent
STAGE1_WEIGHTS = ROOT / "runs_stage1" / "stage1_final" / "weights" / "best.pt"
STAGE2_WEIGHTS = ROOT / "runs_stage2" / "stage2_final" / "weights" / "best.pt"
STAGE1_DATA = ROOT / "data" / "ThyroidXL_two_stage" / "stage1" / "stage1.yaml"
STAGE1_TEST_IMAGES = ROOT / "data" / "ThyroidXL_two_stage" / "stage1" / "images" / "test"
STAGE1_TEST_LABELS = ROOT / "data" / "ThyroidXL_two_stage" / "stage1" / "labels" / "test"


def _to_builtin(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    return value


def _write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_builtin(payload), indent=2), encoding="utf-8")


def _collect_metric_payload(metrics) -> dict:
    payload = {
        "results_dict": _to_builtin(getattr(metrics, "results_dict", {})),
        "fitness": _to_builtin(getattr(metrics, "fitness", None)),
        "speed": _to_builtin(getattr(metrics, "speed", {})),
        "save_dir": _to_builtin(getattr(metrics, "save_dir", None)),
    }
    if hasattr(metrics, "top1"):
        payload["top1"] = _to_builtin(metrics.top1)
    if hasattr(metrics, "top5"):
        payload["top5"] = _to_builtin(metrics.top5)
    if hasattr(metrics, "maps"):
        payload["maps"] = _to_builtin(metrics.maps)
    if hasattr(metrics, "names"):
        payload["names"] = _to_builtin(metrics.names)
    return payload


def _build_stage2_classification_dataset() -> Path:
    src_root = ROOT / "data" / "ThyroidXL_two_stage" / "stage2"
    out_root = ROOT / "data" / "ThyroidXL_two_stage" / "stage2_cls"
    class_map = {"0": "PTC", "1": "NonPTC"}

    for split in ("train", "val", "test"):
        for class_name in class_map.values():
            (out_root / split / class_name).mkdir(parents=True, exist_ok=True)

        labels_dir = src_root / "labels" / split
        images_dir = src_root / "images" / split

        for label_path in sorted(labels_dir.glob("*.txt")):
            label_value = label_path.read_text(encoding="utf-8").strip()
            class_name = class_map.get(label_value)
            if class_name is None:
                continue

            stem = label_path.stem
            image_path = None
            for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"):
                candidate = images_dir / f"{stem}{ext}"
                if candidate.exists():
                    image_path = candidate
                    break

            if image_path is None:
                for candidate in images_dir.glob(f"{stem}.*"):
                    if candidate.is_file():
                        image_path = candidate
                        break

            if image_path is None:
                continue

            target_path = out_root / split / class_name / image_path.name
            shutil.copy2(image_path, target_path)

    return out_root


def _build_binary_mask_from_label_file(label_path: Path, width: int, height: int) -> np.ndarray:
    canvas = Image.new("1", (width, height), 0)
    draw = ImageDraw.Draw(canvas)

    if not label_path.exists():
        return np.zeros((height, width), dtype=bool)

    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        coords = parts[1:]
        if len(coords) % 2 != 0:
            continue

        polygon = []
        for idx in range(0, len(coords), 2):
            x = min(max(float(coords[idx]) * width, 0.0), max(width - 1, 0))
            y = min(max(float(coords[idx + 1]) * height, 0.0), max(height - 1, 0))
            polygon.append((x, y))
        if len(polygon) >= 3:
            draw.polygon(polygon, fill=1)

    return np.array(canvas, dtype=bool)


def _build_binary_mask_from_prediction(result, width: int, height: int, conf_thres: float = 0.25) -> np.ndarray:
    canvas = Image.new("1", (width, height), 0)
    draw = ImageDraw.Draw(canvas)

    if result.masks is None or result.boxes is None:
        return np.zeros((height, width), dtype=bool)

    polygons = result.masks.xyn
    confidences = result.boxes.conf.tolist() if result.boxes.conf is not None else []

    for idx, poly in enumerate(polygons):
        if idx < len(confidences) and float(confidences[idx]) < conf_thres:
            continue
        if poly is None or len(poly) < 3:
            continue

        points = []
        for xy in poly:
            x = min(max(float(xy[0]) * width, 0.0), max(width - 1, 0))
            y = min(max(float(xy[1]) * height, 0.0), max(height - 1, 0))
            points.append((x, y))
        if len(points) >= 3:
            draw.polygon(points, fill=1)

    return np.array(canvas, dtype=bool)


def _iou(mask_true: np.ndarray, mask_pred: np.ndarray) -> float:
    inter = np.logical_and(mask_true, mask_pred).sum()
    union = np.logical_or(mask_true, mask_pred).sum()
    if union == 0:
        return 1.0
    return float(inter / union)


def _hd95(mask_true: np.ndarray, mask_pred: np.ndarray) -> float | None:
    if not SCIPY_AVAILABLE:
        return None

    true_count = int(mask_true.sum())
    pred_count = int(mask_pred.sum())
    if true_count == 0 and pred_count == 0:
        return 0.0
    if true_count == 0 or pred_count == 0:
        return None

    true_boundary = np.logical_and(mask_true, np.logical_not(binary_erosion(mask_true)))
    pred_boundary = np.logical_and(mask_pred, np.logical_not(binary_erosion(mask_pred)))

    true_pts = np.argwhere(true_boundary)
    pred_pts = np.argwhere(pred_boundary)
    if true_pts.size == 0 or pred_pts.size == 0:
        return None

    if len(true_pts) > 4000:
        true_pts = true_pts[:: max(1, len(true_pts) // 4000)]
    if len(pred_pts) > 4000:
        pred_pts = pred_pts[:: max(1, len(pred_pts) // 4000)]

    d_true = cdist(true_pts, pred_pts).min(axis=1)
    d_pred = cdist(pred_pts, true_pts).min(axis=1)
    all_dist = np.concatenate([d_true, d_pred])
    return float(np.percentile(all_dist, 95))


def _summarize_rows(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get(key, "unknown")), []).append(row)

    summary: dict[str, dict[str, Any]] = {}
    for group_name, group_rows in groups.items():
        iou_values = [float(r["iou"]) for r in group_rows if r.get("iou", "") != ""]
        hd95_values = [float(r["hd95"]) for r in group_rows if r.get("hd95", "") != ""]
        summary[group_name] = {
            "count": len(group_rows),
            "mean_iou": float(np.mean(iou_values)) if iou_values else None,
            "median_iou": float(np.median(iou_values)) if iou_values else None,
            "mean_hd95": float(np.mean(hd95_values)) if hd95_values else None,
            "median_hd95": float(np.median(hd95_values)) if hd95_values else None,
            "hd95_valid_count": len(hd95_values),
        }
    return summary


def _summarize_patient_rows(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get(key, "unknown")), []).append(row)

    summary: dict[str, dict[str, Any]] = {}
    for group_name, group_rows in groups.items():
        iou_values = [float(r["mean_iou"]) for r in group_rows if r.get("mean_iou", "") != ""]
        hd95_values = [float(r["mean_hd95"]) for r in group_rows if r.get("mean_hd95", "") != ""]
        summary[group_name] = {
            "count": len(group_rows),
            "mean_iou": float(np.mean(iou_values)) if iou_values else None,
            "median_iou": float(np.median(iou_values)) if iou_values else None,
            "mean_hd95": float(np.mean(hd95_values)) if hd95_values else None,
            "median_hd95": float(np.median(hd95_values)) if hd95_values else None,
            "hd95_valid_count": len(hd95_values),
        }
    return summary


def _normalize_patient_id(value: object) -> str:
    try:
        return str(int(value)).zfill(8)
    except (TypeError, ValueError):
        return str(value)


def _load_patient_labels() -> dict[str, int]:
    annot_path = ROOT / "data" / "ThyroidXL" / "train" / "train_annotations.json"
    with annot_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    labels: dict[str, int] = {}
    for patient_id, patient_record in data.get("info", {}).items():
        nodule_keys = ("nodule_1", "nodule_2")
        patient_label = 2
        for key in nodule_keys:
            nodule = patient_record.get(key) or {}
            histopathology = str(nodule.get("Histopathology") or "").strip().lower()
            conclusion = str(nodule.get("Conclusion") or "").strip().lower()
            if "papillary" in histopathology or "papillary thyroid carcinoma" in histopathology:
                patient_label = 0
                break
            if "carcinoma" in histopathology or "malignant" in histopathology or "cancer" in histopathology:
                patient_label = 1
                break
            if "benign" in conclusion:
                patient_label = 2
                break
            if "malignant" in conclusion or "carcinoma" in conclusion:
                patient_label = 1
                break
        labels[_normalize_patient_id(patient_id)] = patient_label
    return labels


def _load_test_patients() -> set[str]:
    split_path = ROOT / "data" / "ThyroidXL_two_stage" / "metadata" / "patient_split.csv"
    with split_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return {_normalize_patient_id(row["patient_id"]) for row in reader if row["split"] == "test"}


def _image_to_patient_map() -> dict[str, str]:
    annot_path = ROOT / "data" / "ThyroidXL" / "train" / "train_annotations.json"
    with annot_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    image_patient: dict[str, str] = {}
    for patient_id, patient_record in data.get("info", {}).items():
        patient_key = _normalize_patient_id(patient_id)
        for image_name in patient_record.get("images", []):
            image_patient[image_name] = patient_key
    return image_patient


def evaluate_stage1() -> dict:
    if not STAGE1_WEIGHTS.exists():
        raise FileNotFoundError(f"Stage 1 weights not found: {STAGE1_WEIGHTS}")

    print("\n=== Stage 1 evaluation (Benign vs Malignant) ===")
    model = YOLO(str(STAGE1_WEIGHTS))
    metrics = model.val(
        data=str(STAGE1_DATA),
        split="test",
        device=0,
        verbose=False,
        project=str(ROOT / "runs_stage1"),
        name="stage1_eval",
        exist_ok=True,
    )
    payload = _collect_metric_payload(metrics)
    print("Stage 1 metrics:", payload.get("results_dict", {}))
    return payload


def evaluate_stage2() -> dict:
    if not STAGE2_WEIGHTS.exists():
        raise FileNotFoundError(f"Stage 2 weights not found: {STAGE2_WEIGHTS}")

    print("\n=== Stage 2 evaluation (PTC vs NonPTC) ===")
    stage2_cls_root = _build_stage2_classification_dataset()
    model = YOLO(str(STAGE2_WEIGHTS))
    metrics = model.val(
        data=str(stage2_cls_root),
        split="test",
        device=0,
        verbose=False,
        project=str(ROOT / "runs_stage2"),
        name="stage2_eval",
        exist_ok=True,
    )
    payload = _collect_metric_payload(metrics)
    print("Stage 2 metrics:", payload.get("results_dict", {}))
    return payload


def evaluate_final_three_class(export_dir: Path) -> dict:
    print("\n=== Final 3-class evaluation (Benign / PTC / NonPTC) ===")

    stage1_model = YOLO(str(STAGE1_WEIGHTS))
    stage2_model = YOLO(str(STAGE2_WEIGHTS))

    patient_labels = _load_patient_labels()
    test_patients = _load_test_patients()
    image_to_patient = _image_to_patient_map()

    all_true = []
    all_pred = []

    test_images = sorted(STAGE1_TEST_IMAGES.glob("*.*"))
    if not test_images:
        raise FileNotFoundError("No images found in stage1/test split")

    per_image_rows: list[dict[str, Any]] = []
    seg_rows: list[dict[str, Any]] = []

    for image_path in test_images:
        image_name = image_path.name
        patient_id = image_to_patient.get(image_name)
        if patient_id is None or patient_id not in test_patients:
            continue

        true_label = patient_labels.get(patient_id, 2)
        true_name = "Benign" if true_label == 2 else ("PTC" if true_label == 0 else "NonPTC")

        stage1_res = stage1_model(str(image_path), verbose=False)[0]
        stage1_name = "Benign"
        if hasattr(stage1_res, "probs") and stage1_res.probs is not None:
            stage1_name = stage1_res.names[stage1_res.probs.top1]
        else:
            if len(stage1_res.boxes) > 0:
                cls_idx = int(stage1_res.boxes.cls[0].item())
                stage1_name = stage1_res.names[cls_idx]

        if stage1_name == "Benign":
            pred_name = "Benign"
            stage2_name = "NA"
        else:
            stage2_res = stage2_model(str(image_path), verbose=False)[0]
            if hasattr(stage2_res, "probs") and stage2_res.probs is not None:
                stage2_name = stage2_res.names[stage2_res.probs.top1]
            else:
                if len(stage2_res.boxes) > 0:
                    cls_idx = int(stage2_res.boxes.cls[0].item())
                    stage2_name = stage2_res.names[cls_idx]
                else:
                    stage2_name = "NonPTC"
            pred_name = stage2_name

        all_true.append(true_name)
        all_pred.append(pred_name)

        width, height = stage1_res.orig_shape[1], stage1_res.orig_shape[0]
        gt_label_path = STAGE1_TEST_LABELS / f"{image_path.stem}.txt"
        gt_mask = _build_binary_mask_from_label_file(gt_label_path, width, height)
        pred_mask = _build_binary_mask_from_prediction(stage1_res, width, height)

        iou_value = _iou(gt_mask, pred_mask)
        hd95_value = _hd95(gt_mask, pred_mask)

        seg_rows.append(
            {
                "image": image_name,
                "patient_id": patient_id,
                "iou": iou_value,
                "hd95": "" if hd95_value is None else hd95_value,
            }
        )

        per_image_rows.append(
            {
                "image": image_name,
                "patient_id": patient_id,
                "true_label": true_name,
                "stage1_pred": stage1_name,
                "stage2_pred": stage2_name,
                "final_pred": pred_name,
                "iou": iou_value,
                "hd95": "" if hd95_value is None else hd95_value,
            }
        )

    classes = ["Benign", "PTC", "NonPTC"]
    matrix = {true: {pred: 0 for pred in classes} for true in classes}
    for true_name, pred_name in zip(all_true, all_pred):
        matrix[true_name][pred_name] += 1

    print("Confusion matrix:")
    for true_name in classes:
        print(true_name, matrix[true_name])

    correct = sum(matrix[c][c] for c in classes)
    total = len(all_true)
    acc = correct / total if total else 0.0
    print(f"Overall accuracy: {acc:.4f}")

    per_class = {}
    for c in classes:
        tp = matrix[c][c]
        fp = sum(matrix[other][c] for other in classes if other != c)
        fn = sum(matrix[c][other] for other in classes if other != c)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[c] = {"precision": precision, "recall": recall, "f1": f1}
        print(f"{c}: precision={precision:.4f}, recall={recall:.4f}, f1={f1:.4f}")

    macro_f1 = float(sum(per_class[c]["f1"] for c in classes) / len(classes))

    hd95_values = [float(r["hd95"]) for r in seg_rows if r["hd95"] != ""]
    iou_values = [float(r["iou"]) for r in seg_rows]
    seg_summary = {
        "scipy_available": SCIPY_AVAILABLE,
        "n_images": len(seg_rows),
        "mean_iou": float(np.mean(iou_values)) if iou_values else None,
        "median_iou": float(np.median(iou_values)) if iou_values else None,
        "mean_hd95": float(np.mean(hd95_values)) if hd95_values else None,
        "median_hd95": float(np.median(hd95_values)) if hd95_values else None,
        "hd95_valid_count": len(hd95_values),
    }

    predictions_csv = export_dir / "final_predictions.csv"
    with predictions_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image", "patient_id", "true_label", "stage1_pred", "stage2_pred", "final_pred", "iou", "hd95"],
        )
        writer.writeheader()
        writer.writerows(per_image_rows)

    seg_csv = export_dir / "stage1_segmentation_iou_hd95.csv"
    with seg_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "patient_id", "iou", "hd95"])
        writer.writeheader()
        writer.writerows(seg_rows)

    patient_rows = []
    patient_groups: dict[str, list[dict[str, Any]]] = {}
    for row in per_image_rows:
        patient_groups.setdefault(str(row["patient_id"]), []).append(row)

    for patient_id, rows in sorted(patient_groups.items()):
        true_label = rows[0]["true_label"]
        n_images = len(rows)
        n_correct = sum(1 for r in rows if r["true_label"] == r["final_pred"])

        vote_counts = {"Benign": 0, "PTC": 0, "NonPTC": 0}
        for r in rows:
            vote_counts[r["final_pred"]] = vote_counts.get(r["final_pred"], 0) + 1
        patient_pred = max(vote_counts.items(), key=lambda x: x[1])[0]

        iou_values = [float(r["iou"]) for r in rows if r.get("iou", "") != ""]
        hd95_values = [float(r["hd95"]) for r in rows if r.get("hd95", "") != ""]

        patient_rows.append(
            {
                "patient_id": patient_id,
                "true_label": true_label,
                "patient_pred": patient_pred,
                "n_images": n_images,
                "n_correct_images": n_correct,
                "image_accuracy": (n_correct / n_images) if n_images else 0.0,
                "mean_iou": float(np.mean(iou_values)) if iou_values else "",
                "mean_hd95": float(np.mean(hd95_values)) if hd95_values else "",
            }
        )

    patient_csv = export_dir / "patient_level_predictions.csv"
    with patient_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "patient_id",
                "true_label",
                "patient_pred",
                "n_images",
                "n_correct_images",
                "image_accuracy",
                "mean_iou",
                "mean_hd95",
            ],
        )
        writer.writeheader()
        writer.writerows(patient_rows)

    patient_classes = ["Benign", "PTC", "NonPTC"]
    patient_conf = {t: {p: 0 for p in patient_classes} for t in patient_classes}
    for row in patient_rows:
        patient_conf[row["true_label"]][row["patient_pred"]] += 1

    patient_total = len(patient_rows)
    patient_correct = sum(patient_conf[c][c] for c in patient_classes)
    patient_acc = (patient_correct / patient_total) if patient_total else 0.0

    patient_per_class = {}
    for c in patient_classes:
        tp = patient_conf[c][c]
        fp = sum(patient_conf[other][c] for other in patient_classes if other != c)
        fn = sum(patient_conf[c][other] for other in patient_classes if other != c)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        patient_per_class[c] = {"precision": precision, "recall": recall, "f1": f1}

    classwise_seg_summary = _summarize_rows(per_image_rows, key="true_label")
    patient_seg_summary = _summarize_patient_rows(patient_rows, key="true_label")

    classwise_csv = export_dir / "classwise_segmentation_metrics.csv"
    with classwise_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["class", "count", "mean_iou", "median_iou", "mean_hd95", "median_hd95", "hd95_valid_count"],
        )
        writer.writeheader()
        for class_name, info in classwise_seg_summary.items():
            writer.writerow({"class": class_name, **info})

    patient_classwise_csv = export_dir / "patient_level_classwise_segmentation_metrics.csv"
    with patient_classwise_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["class", "count", "mean_iou", "median_iou", "mean_hd95", "median_hd95", "hd95_valid_count"],
        )
        writer.writeheader()
        for class_name, info in patient_seg_summary.items():
            writer.writerow({"class": class_name, **info})

    patient_metrics = {
        "confusion_matrix": patient_conf,
        "accuracy": patient_acc,
        "per_class": patient_per_class,
    }
    _write_json(export_dir / "patient_level_metrics.json", patient_metrics)

    _write_json(export_dir / "classwise_segmentation_summary.json", classwise_seg_summary)
    _write_json(export_dir / "patient_level_classwise_segmentation_summary.json", patient_seg_summary)

    result = {
        "matrix": matrix,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "segmentation": seg_summary,
        "classwise_segmentation": classwise_seg_summary,
        "patient_level": patient_metrics,
        "patient_level_segmentation": patient_seg_summary,
        "exports": {
            "predictions_csv": str(predictions_csv),
            "segmentation_csv": str(seg_csv),
            "patient_predictions_csv": str(patient_csv),
            "classwise_segmentation_csv": str(classwise_csv),
            "patient_level_classwise_segmentation_csv": str(patient_classwise_csv),
        },
    }
    _write_json(export_dir / "final_three_class_metrics.json", result)
    _write_json(export_dir / "stage1_segmentation_summary.json", seg_summary)
    return result


def run_full_evaluation() -> dict:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_dir = ROOT / "runs_metrics" / f"eval_{timestamp}"
    export_dir.mkdir(parents=True, exist_ok=True)

    stage1_payload = evaluate_stage1()
    stage2_payload = evaluate_stage2()
    final_payload = evaluate_final_three_class(export_dir=export_dir)

    _write_json(export_dir / "stage1_metrics.json", stage1_payload)
    _write_json(export_dir / "stage2_metrics.json", stage2_payload)

    summary = {
        "export_dir": str(export_dir),
        "stage1": stage1_payload,
        "stage2": stage2_payload,
        "final_three_class": final_payload,
    }
    _write_json(export_dir / "summary_all_metrics.json", summary)

    print(f"\nAll exports saved to: {export_dir}")
    return summary


if __name__ == "__main__":
    run_full_evaluation()
