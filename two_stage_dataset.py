from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from sklearn.model_selection import train_test_split


PTC_TERMS = {
    "papillary thyroid carcinoma",
    "micro-papillary thyroid carcinoma",
    "papillary thyroid carcinoma, chronic thyroiditis",
}

NON_PTC_TERMS = {
    "follicular thyroid carcinoma",
    "medullary thyroid carcinoma",
    "hurthle cell carcinoma",
    "follicular thyroid tumor",
    "anaplastic",
    "carcinoma",
    "malignant",
    "thyroiditis",
    "cancer",
}

BENIGN_TERMS = {
    "thyroid cyst disease",
    "colloid nodule",
    "follicular thyroid adenoma",
    "hurthle cell adenoma",
    "adenoma",
    "benign",
}


def normalize_patient_id(value: object) -> str:
    try:
        return str(int(value)).zfill(8)
    except (TypeError, ValueError):
        return str(value)


def _clean_text(value: object) -> str:
    return str(value or "").strip().lower()


def infer_patient_label(patient_record: Dict) -> int:
    """Return the original three-class label: 0=PTC, 1=NonPTC, 2=Benign."""
    for nodule_key in ("nodule_1", "nodule_2"):
        nodule = patient_record.get(nodule_key) or {}
        histopathology = _clean_text(nodule.get("Histopathology"))
        conclusion = _clean_text(nodule.get("Conclusion"))

        if histopathology and any(term in histopathology for term in PTC_TERMS):
            return 0
        if histopathology and any(term in histopathology for term in NON_PTC_TERMS):
            return 1
        if histopathology and any(term in histopathology for term in BENIGN_TERMS):
            return 2
        if "benign" in conclusion:
            return 2
        if "malignant" in conclusion or "carcinoma" in conclusion:
            return 1

    final_conclusion = _clean_text(patient_record.get("conclusion"))
    if final_conclusion in {"3", "benign"}:
        return 2
    if final_conclusion in {"4", "malignant"}:
        return 1
    return 2


def load_patient_records(json_path: Path) -> List[Dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = []
    for patient_id, patient_record in data.get("info", {}).items():
        patient_records = {
            "patient_id": normalize_patient_id(patient_id),
            "label": infer_patient_label(patient_record),
            "images": list(patient_record.get("images", [])),
        }
        records.append(patient_records)
    return records


def make_stratified_split(
    patient_records: Iterable[Dict],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[set, set, set]:
    """Return (train_ids, val_ids, test_ids) for patient-level splits.

    A strict 70/15/15 split is not always feasible when a dataset has only a small
    number of samples in a class. In that case we preserve class coverage and use the
    largest feasible split while keeping the train/val/test proportions close to the
    requested targets.
    """
    patient_list = list(patient_records)
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-9:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    by_label = defaultdict(list)
    for row in patient_list:
        by_label[int(row["label"])].append(row["patient_id"])

    train_ids, val_ids, test_ids = set(), set(), set()
    rng = __import__("random").Random(seed)

    for label, patient_ids in by_label.items():
        rng.shuffle(patient_ids)
        n_total = len(patient_ids)
        target_train = max(1, round(n_total * train_ratio))
        target_val = max(1, round(n_total * val_ratio))
        target_test = max(1, round(n_total * test_ratio))

        remaining = n_total - target_train - target_val - target_test
        if remaining > 0:
            target_train += remaining
        elif remaining < 0:
            deficit = abs(remaining)
            if target_val >= 1 and target_test >= 1:
                target_val = max(1, target_val - (deficit // 2))
                target_test = max(1, target_test - (deficit - max(1, target_val - 1)))
            else:
                target_test = max(1, target_test - deficit)

        target_train = min(target_train, n_total)
        target_val = min(target_val, n_total - target_train)
        target_test = max(1, min(target_test, n_total - target_train - target_val))

        if target_train + target_val + target_test > n_total:
            overflow = target_train + target_val + target_test - n_total
            target_test = max(1, target_test - overflow)

        if target_train + target_val + target_test < n_total:
            remaining = n_total - (target_train + target_val + target_test)
            target_train += remaining

        split = {
            "train": patient_ids[:target_train],
            "val": patient_ids[target_train:target_train + target_val],
            "test": patient_ids[target_train + target_val:target_train + target_val + target_test],
        }

        if not split["test"]:
            split["test"] = [patient_ids[-1]]
            if split["val"]:
                split["val"] = split["val"][:-1]
            else:
                split["train"] = split["train"][:-1]

        train_ids.update(split["train"])
        val_ids.update(split["val"])
        test_ids.update(split["test"])

    if train_ids & val_ids or val_ids & test_ids or train_ids & test_ids:
        raise RuntimeError("Patient leakage detected in split generation.")

    return set(train_ids), set(val_ids), set(test_ids)


def write_split_manifest(output_dir: Path, train_ids: set, val_ids: set, test_ids: set):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "patient_split.csv"

    rows = []
    for split_name, split_ids in {"train": train_ids, "val": val_ids, "test": test_ids}.items():
        for patient_id in sorted(split_ids):
            rows.append({"patient_id": patient_id, "split": split_name})

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["patient_id", "split"])
        writer.writeheader()
        writer.writerows(rows)

    return csv_path


def polygon_to_yolo_line(segmentation, width: int, height: int, class_id: int) -> List[str]:
    lines = []
    if not isinstance(segmentation, list):
        return lines

    for polygon in segmentation:
        if not isinstance(polygon, list) or len(polygon) < 6 or len(polygon) % 2 != 0:
            continue
        normalized = []
        for index in range(0, len(polygon), 2):
            x = max(0.0, min(1.0, float(polygon[index]) / max(width, 1)))
            y = max(0.0, min(1.0, float(polygon[index + 1]) / max(height, 1)))
            normalized.extend([x, y])
        lines.append(f"{class_id} " + " ".join(f"{value:.6f}" for value in normalized))
    return lines


def prepare_stage_dataset(
    source_root: Path,
    output_root: Path,
    split_name: str,
    patient_ids: set,
    original_image_root: Path,
    original_annotations: Dict,
    stage: str,
    label_policy: str,
    include_stage2_benign: bool = False,
):
    image_root = output_root / "images" / split_name
    label_root = output_root / "labels" / split_name
    image_root.mkdir(parents=True, exist_ok=True)
    label_root.mkdir(parents=True, exist_ok=True)

    images_by_id = {image_info["id"]: image_info for image_info in original_annotations.get("images", [])}
    annotations_by_image = defaultdict(list)
    for annotation in original_annotations.get("annotations", []):
        annotations_by_image[annotation["image_id"]].append(annotation)

    patient_to_label = {
        row["patient_id"]: row["label"]
        for row in load_patient_records(source_root / "train" / "train_annotations.json")
    }

    for image_info in original_annotations.get("images", []):
        patient_id = normalize_patient_id(image_info.get("patient_id"))
        if patient_id not in patient_ids:
            continue

        source_image = original_image_root / image_info["file_name"]
        target_image = image_root / image_info["file_name"]
        if source_image.exists():
            target_image.parent.mkdir(parents=True, exist_ok=True)
            target_image.write_bytes(source_image.read_bytes())

        label_lines = []
        for annotation in annotations_by_image.get(image_info["id"], []):
            segmentation = annotation.get("segmentation", [])
            if stage == "stage1":
                class_id = 0 if patient_to_label[patient_id] == 2 else 1
            elif stage == "stage2":
                original_label = patient_to_label[patient_id]
                if original_label == 2:
                    continue
                class_id = 0 if original_label == 0 else 1
            else:
                class_id = 0
            label_lines.extend(polygon_to_yolo_line(segmentation, image_info["width"], image_info["height"], class_id))

        if label_policy == "classification":
            label_file = label_root / f"{Path(image_info['file_name']).stem}.txt"
            label_file.write_text(f"{class_id}\n", encoding="utf-8")
        else:
            label_file = label_root / f"{Path(image_info['file_name']).stem}.txt"
            label_file.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")


def create_yaml(path: Path, train_dir: str, val_dir: str, test_dir: str, names: Dict[int, str]):
    yaml_lines = [
        f"path: {path.parent.as_posix()}",
        f"train: {train_dir}",
        f"val: {val_dir}",
        f"test: {test_dir}",
        "names:",
    ]
    for class_id, class_name in sorted(names.items()):
        yaml_lines.append(f"  {class_id}: {class_name}")
    path.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")


def build_two_stage_dataset(source_root: Path, output_root: Path, seed: int = 42):
    train_json = source_root / "train" / "train_annotations.json"
    with open(train_json, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    train_records = load_patient_records(train_json)
    train_ids, val_ids, test_ids = make_stratified_split(train_records, seed=seed)

    manifest_path = write_split_manifest(output_root / "metadata", train_ids, val_ids, test_ids)

    stage1_root = output_root / "stage1"
    stage2_root = output_root / "stage2"
    for root in [stage1_root, stage2_root]:
        root.mkdir(parents=True, exist_ok=True)

    patient_labels = {row["patient_id"]: row["label"] for row in train_records}
    patient_to_images = {row["patient_id"]: row["images"] for row in train_records}

    image_lookup = {image_info["id"]: image_info for image_info in train_data.get("images", [])}
    annotations_by_image = defaultdict(list)
    for annotation in train_data.get("annotations", []):
        annotations_by_image[annotation["image_id"]].append(annotation)

    for split_name, split_ids in {"train": train_ids, "val": val_ids, "test": test_ids}.items():
        stage1_images = stage1_root / "images" / split_name
        stage1_labels = stage1_root / "labels" / split_name
        stage1_images.mkdir(parents=True, exist_ok=True)
        stage1_labels.mkdir(parents=True, exist_ok=True)

        stage2_images = stage2_root / "images" / split_name
        stage2_labels = stage2_root / "labels" / split_name
        stage2_images.mkdir(parents=True, exist_ok=True)
        stage2_labels.mkdir(parents=True, exist_ok=True)

        for patient_id in sorted(split_ids):
            patient_label = patient_labels[patient_id]
            for image_name in patient_to_images.get(patient_id, []):
                source_image = source_root / "train" / "images" / image_name
                if not source_image.exists():
                    continue

                image_info = next((img for img in train_data["images"] if img["file_name"] == image_name), None)
                if image_info is None:
                    continue

                stage1_image_path = stage1_images / image_name
                stage1_image_path.write_bytes(source_image.read_bytes())

                class_id = 0 if patient_label == 2 else 1
                lines = []
                for annotation in annotations_by_image.get(image_info["id"], []):
                    seg = annotation.get("segmentation", [])
                    lines.extend(polygon_to_yolo_line(seg, image_info["width"], image_info["height"], class_id))

                stage1_label_path = stage1_labels / f"{Path(image_name).stem}.txt"
                stage1_label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

                if patient_label != 2:
                    stage2_image_path = stage2_images / image_name
                    stage2_image_path.write_bytes(source_image.read_bytes())
                    stage2_class = 0 if patient_label == 0 else 1
                    stage2_label_path = stage2_labels / f"{Path(image_name).stem}.txt"
                    stage2_label_path.write_text(f"{stage2_class}\n", encoding="utf-8")

    stage1_yaml = stage1_root / "stage1.yaml"
    stage2_yaml = stage2_root / "stage2.yaml"
    create_yaml(stage1_yaml, "images/train", "images/val", "images/test", {0: "Benign", 1: "Malignant"})
    create_yaml(stage2_yaml, "images/train", "images/val", "images/test", {0: "PTC", 1: "NonPTC"})

    return {
        "manifest": manifest_path,
        "stage1_yaml": stage1_yaml,
        "stage2_yaml": stage2_yaml,
        "train_ids": train_ids,
        "val_ids": val_ids,
        "test_ids": test_ids,
    }


if __name__ == "__main__":
    source_root = Path("./data/ThyroidXL").resolve()
    output_root = Path("./data/ThyroidXL_two_stage").resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    result = build_two_stage_dataset(source_root, output_root)
    print("Split manifest:", result["manifest"])
    print("Stage 1 yaml:", result["stage1_yaml"])
    print("Stage 2 yaml:", result["stage2_yaml"])
    print(f"Train: {len(result['train_ids'])}, Val: {len(result['val_ids'])}, Test: {len(result['test_ids'])}")
