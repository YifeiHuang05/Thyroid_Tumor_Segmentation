from pathlib import Path
import argparse
import json
import shutil
from collections import Counter

import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# Configuration
# ============================================================

RANDOM_SEED = 42
VAL_RATIO = 0.15


# ============================================================
# JSON / annotation helpers
# ============================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_patient_id(patient_id):
    """Keep patient IDs in a consistent string format."""
    if patient_id is None:
        return None

    try:
        return str(int(patient_id)).zfill(8)
    except (ValueError, TypeError):
        return str(patient_id)


def build_image_lookup(data):
    return {
        int(image["id"]): image
        for image in data.get("images", [])
    }


def build_annotation_lookup(data):
    """
    Build image_id -> annotation mapping.

    ThyroidXL currently has one nodule annotation per image.
    We explicitly check that assumption.
    """
    lookup = {}

    for annotation in data.get("annotations", []):
        image_id = int(annotation["image_id"])

        if image_id in lookup:
            raise RuntimeError(
                f"Multiple annotations found for image_id={image_id}. "
                "This script assumes one nodule annotation per image."
            )

        lookup[image_id] = annotation

    return lookup


def build_patient_info_lookup(data):
    """
    JSON 'info' may be either:
      {patient_id: metadata}
    or a list of metadata dictionaries.

    Return:
      patient_id -> metadata
    """
    info = data.get("info", {})

    if isinstance(info, dict):
        result = {}

        for patient_id, metadata in info.items():
            result[normalize_patient_id(patient_id)] = metadata

        return result

    if isinstance(info, list):
        result = {}

        for metadata in info:
            patient_id = (
                metadata.get("patient_id")
                or metadata.get("id")
            )

            if patient_id is not None:
                result[normalize_patient_id(patient_id)] = metadata

        return result

    return {}


def get_patient_id(image):
    """
    Extract patient ID from image metadata.

    ThyroidXL image records are expected to contain patient_id.
    """
    patient_id = (
        image.get("patient_id")
        or image.get("patientId")
        or image.get("patient")
    )

    if patient_id is None:
        raise RuntimeError(
            f"Cannot find patient_id for image: {image}"
        )

    return normalize_patient_id(patient_id)


# ============================================================
# YOLO segmentation conversion
# ============================================================

def polygon_to_yolo(polygon, width, height):
    """
    Convert absolute polygon coordinates:

        [x1, y1, x2, y2, ...]

    to YOLO normalized coordinates.
    """

    if not polygon or len(polygon) < 6:
        return None

    if len(polygon) % 2 != 0:
        raise ValueError(
            "Polygon has an odd number of coordinates."
        )

    result = []

    for i, value in enumerate(polygon):
        value = float(value)

        if i % 2 == 0:
            # x
            normalized = value / width
        else:
            # y
            normalized = value / height

        normalized = max(0.0, min(1.0, normalized))
        result.append(normalized)

    return " ".join(f"{x:.6f}" for x in result)


def write_yolo_label(label_path, annotation, width, height):
    """
    Generate a pure segmentation label.

    New task:
        class 0 = nodule

    Original category_id (benign/malignant) is deliberately
    ignored here.
    """

    segmentation = annotation.get("segmentation", [])

    if not segmentation:
        raise RuntimeError(
            f"Annotation {annotation['id']} has no segmentation."
        )

    lines = []

    for polygon in segmentation:

        yolo_polygon = polygon_to_yolo(
            polygon,
            width,
            height
        )

        if yolo_polygon is not None:
            lines.append(
                f"0 {yolo_polygon}"
            )

    if not lines:
        raise RuntimeError(
            f"No valid polygon generated for "
            f"annotation {annotation['id']}."
        )

    label_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8"
    )


# ============================================================
# Build image-level metadata
# ============================================================

def build_records(data, source_split):
    """
    Convert one annotation JSON into image-level records.

    source_split:
        'original_train'
        or
        'original_test'
    """

    image_lookup = build_image_lookup(data)
    annotation_lookup = build_annotation_lookup(data)
    patient_info_lookup = build_patient_info_lookup(data)

    records = []

    for image_id, image in image_lookup.items():

        if image_id not in annotation_lookup:
            raise RuntimeError(
                f"Image {image_id} has no annotation."
            )

        annotation = annotation_lookup[image_id]

        patient_id = get_patient_id(image)

        patient_info = patient_info_lookup.get(
            patient_id,
            {}
        )

        category_id = int(
            annotation["category_id"]
        )

        if category_id not in [0, 1]:
            raise RuntimeError(
                f"Unexpected category_id={category_id} "
                f"for image {image_id}."
            )

        records.append({
            "image_id": image_id,
            "image_name": image["file_name"],
            "patient_id": patient_id,
            "width": int(image["width"]),
            "height": int(image["height"]),

            # New segmentation target.
            "segmentation_class": 0,

            # Original category retained for later
            # benign/malignant classification.
            "original_category_id": category_id,

            "source_split": source_split,

            # Preserve patient metadata when available.
            "age": patient_info.get("age"),
            "gender": patient_info.get("gender"),
            "conclusion": patient_info.get("conclusion"),

            "fnac": (
                patient_info.get("nodule_1", {}).get("FNAC")
                if isinstance(
                    patient_info.get("nodule_1"),
                    dict
                )
                else None
            ),

            "tirads": (
                patient_info.get("nodule_1", {}).get("TIRADS")
                if isinstance(
                    patient_info.get("nodule_1"),
                    dict
                )
                else None
            ),

            "histopathology": (
                patient_info.get("nodule_1", {}).get(
                    "Histopathology"
                )
                if isinstance(
                    patient_info.get("nodule_1"),
                    dict
                )
                else None
            ),
        })

    return records


# ============================================================
# Patient-level split
# ============================================================

def create_train_val_split(train_records, seed):
    """
    Split ONLY the original training patients.

    The original test set is never touched.
    """

    train_df = pd.DataFrame(train_records)

    patient_df = (
        train_df[
            [
                "patient_id",
                "original_category_id"
            ]
        ]
        .drop_duplicates()
        .copy()
    )

    # Verify that each patient has a consistent
    # benign/malignant annotation.
    consistency = (
        patient_df
        .groupby("patient_id")["original_category_id"]
        .nunique()
    )

    inconsistent = consistency[
        consistency > 1
    ]

    if len(inconsistent) > 0:
        raise RuntimeError(
            "Some patients have both benign and malignant "
            "annotations in the training data:\n"
            f"{inconsistent}"
        )

    patient_ids = patient_df["patient_id"].values
    patient_labels = patient_df[
        "original_category_id"
    ].values

    train_patients, val_patients = train_test_split(
        patient_ids,
        test_size=VAL_RATIO,
        random_state=seed,
        stratify=patient_labels
    )

    train_patients = set(train_patients)
    val_patients = set(val_patients)

    train_df["split"] = train_df["patient_id"].apply(
        lambda x: "train" if x in train_patients else "val"
    )

    return train_df


# ============================================================
# Copy images + generate labels
# ============================================================

def process_split(
    records_df,
    source_images_dir,
    source_data,
    destination_root,
    split_name
):
    """
    Copy images and regenerate YOLO segmentation labels.
    """

    image_lookup = build_image_lookup(source_data)
    annotation_lookup = build_annotation_lookup(source_data)

    images_out = (
        destination_root
        / "segmentation"
        / "images"
        / split_name
    )

    labels_out = (
        destination_root
        / "segmentation"
        / "labels"
        / split_name
    )

    images_out.mkdir(
        parents=True,
        exist_ok=True
    )

    labels_out.mkdir(
        parents=True,
        exist_ok=True
    )

    missing_images = []

    copied = 0
    labels_written = 0

    for _, row in records_df.iterrows():

        image_name = row["image_name"]

        source_image = (
            source_images_dir / image_name
        )

        # Fallback: search recursively.
        if not source_image.exists():

            matches = list(
                source_images_dir.rglob(image_name)
            )

            if matches:
                source_image = matches[0]
            else:
                missing_images.append(
                    image_name
                )
                continue

        destination_image = (
            images_out / image_name
        )

        destination_label = (
            labels_out
            / f"{Path(image_name).stem}.txt"
        )

        shutil.copy2(
            source_image,
            destination_image
        )

        image_id = int(row["image_id"])

        image = image_lookup[image_id]
        annotation = annotation_lookup[image_id]

        write_yolo_label(
            destination_label,
            annotation,
            int(image["width"]),
            int(image["height"])
        )

        copied += 1
        labels_written += 1

    return copied, labels_written, missing_images


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Create a clean ThyroidXL segmentation dataset "
            "while preserving the official test set."
        )
    )

    parser.add_argument(
        "--train-json",
        required=True,
        help="Original training annotation JSON"
    )

    parser.add_argument(
        "--train-images",
        required=True,
        help="Original training images directory"
    )

    parser.add_argument(
        "--test-json",
        required=True,
        help="Original test annotation JSON"
    )

    parser.add_argument(
        "--test-images",
        required=True,
        help="Original test images directory"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output directory"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED
    )

    args = parser.parse_args()

    train_json = Path(args.train_json)
    train_images = Path(args.train_images)

    test_json = Path(args.test_json)
    test_images = Path(args.test_images)

    output = Path(args.output)

    # --------------------------------------------------------
    # Check inputs
    # --------------------------------------------------------

    for path, description in [
        (train_json, "training JSON"),
        (train_images, "training images"),
        (test_json, "test JSON"),
        (test_images, "test images"),
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"{description} not found: {path}"
            )

    # --------------------------------------------------------
    # Load original JSON files
    # --------------------------------------------------------

    print("=" * 70)
    print("LOADING ORIGINAL THYROIDXL DATA")
    print("=" * 70)

    train_data = load_json(train_json)
    test_data = load_json(test_json)

    print(
        f"Original training images: "
        f"{len(train_data.get('images', []))}"
    )

    print(
        f"Original training annotations: "
        f"{len(train_data.get('annotations', []))}"
    )

    print(
        f"Original test images: "
        f"{len(test_data.get('images', []))}"
    )

    print(
        f"Original test annotations: "
        f"{len(test_data.get('annotations', []))}"
    )

    # --------------------------------------------------------
    # Build records
    # --------------------------------------------------------

    print("\nBuilding metadata...")

    train_records = build_records(
        train_data,
        "original_train"
    )

    test_records = build_records(
        test_data,
        "original_test"
    )

    train_df = pd.DataFrame(train_records)
    test_df = pd.DataFrame(test_records)

    # --------------------------------------------------------
    # Check duplicate image names
    # --------------------------------------------------------

    if train_df["image_name"].duplicated().any():
        raise RuntimeError(
            "Duplicate image names found in training data."
        )

    if test_df["image_name"].duplicated().any():
        raise RuntimeError(
            "Duplicate image names found in test data."
        )

    # --------------------------------------------------------
    # Check patient overlap between official train/test
    # --------------------------------------------------------

    train_patients = set(
        train_df["patient_id"]
    )

    test_patients = set(
        test_df["patient_id"]
    )

    overlap = train_patients & test_patients

    print(
        f"\nTraining patients: {len(train_patients)}"
    )

    print(
        f"Official test patients: {len(test_patients)}"
    )

    if overlap:
        print(
            f"WARNING: {len(overlap)} patients occur "
            "in both original train and test."
        )
        print(
            "This should be investigated before training."
        )
    else:
        print(
            "Original train/test patient separation: PASSED"
        )

    # --------------------------------------------------------
    # Create TRAIN / VAL split
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("SPLITTING ORIGINAL TRAINING SET")
    print("=" * 70)

    train_df = create_train_val_split(
        train_df.to_dict("records"),
        args.seed
    )

    print(
        "\nPatients after split:"
    )

    patient_counts = (
        train_df[
            ["patient_id", "split"]
        ]
        .drop_duplicates()
        ["split"]
        .value_counts()
    )

    print(patient_counts)

    print("\nImages after split:")
    print(train_df["split"].value_counts())

    print("\nBenign/malignant distribution:")
    print(
        pd.crosstab(
            train_df["split"],
            train_df["original_category_id"]
        )
    )

    # --------------------------------------------------------
    # Add test split
    # --------------------------------------------------------

    test_df = test_df.copy()

    test_df["split"] = "test"

    # --------------------------------------------------------
    # Combine master metadata
    # --------------------------------------------------------

    master_df = pd.concat(
        [
            train_df,
            test_df
        ],
        ignore_index=True
    )

    # --------------------------------------------------------
    # Create output directories
    # --------------------------------------------------------

    metadata_dir = (
        output / "metadata"
    )

    annotations_dir = (
        output / "annotations"
    )

    metadata_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    annotations_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save master tables
    # --------------------------------------------------------

    master_df.to_csv(
        metadata_dir / "master_annotations.csv",
        index=False
    )

    patient_split_df = (
        master_df[
            [
                "patient_id",
                "split"
            ]
        ]
        .drop_duplicates()
        .sort_values(
            ["split", "patient_id"]
        )
    )

    patient_split_df.to_csv(
        metadata_dir / "patient_split.csv",
        index=False
    )

    train_df.to_csv(
        metadata_dir / "train_val_annotations.csv",
        index=False
    )

    test_df.to_csv(
        metadata_dir / "test_annotations.csv",
        index=False
    )

    # --------------------------------------------------------
    # Copy original JSON files for traceability
    # --------------------------------------------------------

    shutil.copy2(
        train_json,
        annotations_dir / "original_train_annotations.json"
    )

    shutil.copy2(
        test_json,
        annotations_dir / "original_test_annotations.json"
    )

    # --------------------------------------------------------
    # Generate clean segmentation dataset
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("GENERATING CLEAN YOLO SEGMENTATION DATASET")
    print("=" * 70)

    # Original training images -> new train
    train_train_df = train_df[
        train_df["split"] == "train"
    ].copy()

    copied_train, labels_train, missing_train = (
        process_split(
            train_train_df,
            train_images,
            train_data,
            output,
            "train"
        )
    )

    # Original training images -> new val
    train_val_df = train_df[
        train_df["split"] == "val"
    ].copy()

    copied_val, labels_val, missing_val = (
        process_split(
            train_val_df,
            train_images,
            train_data,
            output,
            "val"
        )
    )

    # ORIGINAL TEST -> new test
    #
    # Absolutely no resplitting happens here.
    copied_test, labels_test, missing_test = (
        process_split(
            test_df,
            test_images,
            test_data,
            output,
            "test"
        )
    )

    print(
        f"\nNew train: "
        f"{copied_train} images / "
        f"{labels_train} labels"
    )

    print(
        f"New val:   "
        f"{copied_val} images / "
        f"{labels_val} labels"
    )

    print(
        f"New test:  "
        f"{copied_test} images / "
        f"{labels_test} labels"
    )

    # --------------------------------------------------------
    # Missing image report
    # --------------------------------------------------------

    all_missing = (
        [("train", x) for x in missing_train]
        + [("val", x) for x in missing_val]
        + [("test", x) for x in missing_test]
    )

    if all_missing:

        missing_file = (
            metadata_dir / "missing_images.csv"
        )

        pd.DataFrame(
            all_missing,
            columns=["split", "image_name"]
        ).to_csv(
            missing_file,
            index=False
        )

        print(
            f"\nWARNING: {len(all_missing)} images missing."
        )

        print(
            f"Missing-image report: {missing_file}"
        )

    else:
        print(
            "\nMissing image check: PASSED"
        )

    # --------------------------------------------------------
    # Create dataset.yaml
    # --------------------------------------------------------

    segmentation_root = (
        output / "segmentation"
    )

    dataset_yaml = (
        segmentation_root / "dataset.yaml"
    )

    yaml_text = f"""path: {segmentation_root.resolve().as_posix()}
train: images/train
val: images/val
test: images/test

nc: 1
names:
  0: nodule
"""

    dataset_yaml.write_text(
        yaml_text,
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # Final consistency checks
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL CHECKS")
    print("=" * 70)

    for split in ["train", "val", "test"]:

        image_dir = (
            segmentation_root
            / "images"
            / split
        )

        label_dir = (
            segmentation_root
            / "labels"
            / split
        )

        image_count = len(
            list(image_dir.iterdir())
        )

        label_count = len(
            list(label_dir.glob("*.txt"))
        )

        print(
            f"{split:5s}: "
            f"{image_count:5d} images | "
            f"{label_count:5d} labels"
        )

        if image_count != label_count:
            raise RuntimeError(
                f"{split}: image/label mismatch."
            )

    # --------------------------------------------------------
    # Patient leakage check
    # --------------------------------------------------------

    split_patient_sets = {
        split: set(
            patient_split_df.loc[
                patient_split_df["split"] == split,
                "patient_id"
            ]
        )
        for split in ["train", "val", "test"]
    }

    for split_a, split_b in [
        ("train", "val"),
        ("train", "test"),
        ("val", "test"),
    ]:

        intersection = (
            split_patient_sets[split_a]
            & split_patient_sets[split_b]
        )

        if intersection:
            raise RuntimeError(
                f"Patient leakage between "
                f"{split_a} and {split_b}: "
                f"{len(intersection)} patients."
            )

    print(
        "Clean dataset patient-level leakage check: PASSED"
    )

    # --------------------------------------------------------
    # Verify every YOLO label is class 0
    # --------------------------------------------------------

    invalid_class_files = []

    for label_path in (
        segmentation_root / "labels"
    ).rglob("*.txt"):

        for line in label_path.read_text(
            encoding="utf-8"
        ).splitlines():

            if not line.strip():
                continue

            class_id = line.split()[0]

            if class_id != "0":
                invalid_class_files.append(
                    str(label_path)
                )
                break

    if invalid_class_files:
        raise RuntimeError(
            f"{len(invalid_class_files)} label files "
            "contain a class other than 0."
        )

    print(
        "YOLO class check: PASSED "
        "(0 = nodule)"
    )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    summary = {
        "random_seed": args.seed,
        "validation_ratio_from_original_train": VAL_RATIO,

        "original_train_images":
            len(train_data.get("images", [])),

        "original_test_images":
            len(test_data.get("images", [])),

        "new_train_images":
            copied_train,

        "new_val_images":
            copied_val,

        "new_test_images":
            copied_test,

        "new_train_patients":
            len(split_patient_sets["train"]),

        "new_val_patients":
            len(split_patient_sets["val"]),

        "new_test_patients":
            len(split_patient_sets["test"]),

        "segmentation_classes": {
            "0": "nodule"
        },

        "official_test_preserved": True
    }

    with open(
        metadata_dir / "dataset_summary.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            summary,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # Final message
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CLEAN DATASET READY")
    print("=" * 70)

    print(f"""
{output.resolve()}

metadata/
  master_annotations.csv
  patient_split.csv
  train_val_annotations.csv
  test_annotations.csv
  dataset_summary.json

annotations/
  original_train_annotations.json
  original_test_annotations.json

segmentation/
  images/
    train/
    val/
    test/

  labels/
    train/
    val/
    test/

  dataset.yaml

IMPORTANT:
  Original train → new train + val
  Original test  → new test
  No test resplitting was performed.

Segmentation:
  class 0 = nodule

YOLO dataset:
  {dataset_yaml}
""")


if __name__ == "__main__":
    main()