"""
ThyroidXL -> YOLO26-seg complete pipeline

TASK
----
One-class PTC instance segmentation.

Class:
    0 = PTC

PIPELINE
--------
1. Read original training COCO-style JSON.
2. Determine PTC status from patient-level Histopathology.
3. Create a reproducible 80/20 patient-level stratified split.
4. Save the split permanently.
5. Convert polygon annotations into YOLO segmentation labels.
6. Copy images into:
       images/train
       images/val
       images/test
7. Keep PTC-negative images with empty labels.
8. Keep the original test cohort completely separate.
9. Create dataset.yaml.
10. Optionally run a short YOLO26n-seg test training run.

IMPORTANT
---------
The train/validation split is created ONLY ONCE.

If metadata/patient_split.csv already exists, the script will
reuse it rather than creating a new random split.

This prevents accidental changes to the validation cohort.
"""

from pathlib import Path
import argparse
import csv
import json
import shutil
import sys
from collections import Counter

from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURATION
# ============================================================
RANDOM_SEED = 2026
VAL_SIZE = 0.20

# PTC definition.
# This is based on patient-level Histopathology.
# nodule_2 is assumed to be null in the training annotation JSON.
PTC_POSITIVE_TERMS = {
    "papillary thyroid carcinoma",
    "micro-papillary thyroid carcinoma",
    "papillary thyroid carcinoma, chronic thyroiditis",
}

# YOLO class definition
PTC_CLASS_ID = 0
PTC_CLASS_NAME = "PTC"

# YOLO26n-seg pretrained model
MODEL_NAME = "model\\yolo26n-seg.pt"


# ============================================================
# PATH ARGUMENTS
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare ThyroidXL for YOLO26-seg and optionally run a test training."
    )

    parser.add_argument(
        "--source",
        required=True,
        help="Original ThyroidXL root directory."
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output ThyroidXL YOLO dataset directory."
    )

    parser.add_argument(
        "--train-json",
        default=None,
        help="Path to train_annotations.json. "
             "If omitted: SOURCE/train/train_annotations.json"
    )

    parser.add_argument(
        "--train-images",
        default=None,
        help="Path to original training images. "
             "If omitted: SOURCE/train/images"
    )

    parser.add_argument(
        "--test-json",
        default=None,
        help="Path to test_annotations.json. "
             "If omitted: SOURCE/test/test_annotations.json"
    )

    parser.add_argument(
        "--test-images",
        default=None,
        help="Path to original test images. "
             "If omitted: SOURCE/test/images"
    )

    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only prepare the dataset. Do not train YOLO."
    )

    parser.add_argument(
        "--test-run",
        action="store_true",
        help="Run a short YOLO26n-seg test training after preparation."
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of epochs for --test-run. Default: 3."
    )

    parser.add_argument(
        "--device",
        default=None,
        help="Training device. Examples: 0, 0,1, cpu. "
             "If omitted, Ultralytics selects automatically."
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Data-loader workers for YOLO training. Default: 4."
    )

    return parser.parse_args()


# ============================================================
# PATH SETUP
# ============================================================
def resolve_paths(args):

    source = Path(args.source).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    # create output directory if it doesn't exist
    output.mkdir(parents=True, exist_ok=True)

    train_json = (
        Path(args.train_json).expanduser().resolve()
        if args.train_json
        else source / "train" / "train_annotations.json"
    )

    train_images = (
        Path(args.train_images).expanduser().resolve()
        if args.train_images
        else source / "train" / "images"
    )

    test_json = (
        Path(args.test_json).expanduser().resolve()
        if args.test_json
        else source / "test" / "test_annotations.json"
    )

    test_images = (
        Path(args.test_images).expanduser().resolve()
        if args.test_images
        else source / "test" / "images"
    )

    return (
        source,
        output,
        train_json,
        train_images,
        test_json,
        test_images,
    )


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================
def create_directories(output):
    directories = [
        output / "images" / "train",
        output / "images" / "val",
        output / "images" / "test",

        output / "labels" / "train",
        output / "labels" / "val",
        output / "labels" / "test",

        output / "metadata",
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# JSON
# ============================================================
def load_json(path):

    print(f"\nLoading JSON:")
    print(f"  {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ============================================================
# PATIENT ID NORMALISATION
# ============================================================
def normalise_patient_id(patient_id):
    """
    Keep patient IDs consistent.
    Example:
        603       -> 00000603
        "603"     -> 00000603
        "00000603" -> 00000603
    """
    try:
        return str(int(patient_id)).zfill(8)
    except (ValueError, TypeError):
        return str(patient_id)

# ============================================================
# PTC STATUS
# ============================================================
def get_histopathology(patient_record):
    values = []

    for nodule_key in ["nodule_1", "nodule_2"]:

        nodule = patient_record.get(nodule_key)

        if not nodule:
            continue

        value = nodule.get("Histopathology")

        if value is not None:
            value = str(value).strip()

            if value:
                values.append(value)

    return values

def is_ptc_patient(patient_record):
    histopathology = get_histopathology(patient_record)
    for diagnosis in histopathology:
        diagnosis_lower = diagnosis.lower().strip()

        for positive_term in PTC_POSITIVE_TERMS:
            if diagnosis_lower == positive_term.lower():
                return True

    return False

# ============================================================
# PATIENT TABLE
# ============================================================
def build_patient_table(data):
    rows = []
    for patient_id_raw, patient_record in data["info"].items():
        patient_id = normalise_patient_id(patient_id_raw)
        images = patient_record.get("images", [])
        histopathology = get_histopathology(patient_record)
        ptc_status = int(
            is_ptc_patient(patient_record)
        )

        rows.append({
            "patient_id": patient_id,
            "ptc_status": ptc_status,
            "histopathology": " | ".join(histopathology),
            "n_images": len(images),
        })

    return rows

# ============================================================
# CREATE / LOAD PERMANENT PATIENT SPLIT
# ============================================================
def create_new_patient_split(patient_table):

    patient_ids = [
        row["patient_id"]
        for row in patient_table
    ]

    labels = [
        row["ptc_status"]
        for row in patient_table
    ]

    train_patients, val_patients = train_test_split(
        patient_ids,
        test_size=VAL_SIZE,
        random_state=RANDOM_SEED,
        stratify=labels,
    )

    train_patients = set(train_patients)
    val_patients = set(val_patients)

    overlap = train_patients & val_patients

    if overlap:
        raise RuntimeError(
            f"Patient leakage detected: {len(overlap)} patients "
            f"occur in both train and validation."
        )

    return train_patients, val_patients

def load_existing_patient_split(csv_path):

    train_patients = set()
    val_patients = set()

    with open(
        csv_path,
        "r",
        encoding="utf-8",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            patient_id = normalise_patient_id(
                row["patient_id"]
            )

            split = row["split"].strip().lower()

            if split == "train":
                train_patients.add(patient_id)

            elif split == "val":
                val_patients.add(patient_id)

            else:
                raise ValueError(
                    f"Unknown split '{split}' "
                    f"for patient {patient_id}"
                )

    overlap = train_patients & val_patients

    if overlap:
        raise RuntimeError(
            "Existing patient_split.csv contains patient leakage."
        )

    return train_patients, val_patients


def save_patient_split(
    patient_table,
    train_patients,
    val_patients,
    metadata_dir,
):
    csv_path = metadata_dir / "patient_split.csv"

    with open(
        csv_path,
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "patient_id",
            "ptc_status",
            "histopathology",
            "n_images",
            "split",
        ])

        for row in sorted(
            patient_table,
            key=lambda x: x["patient_id"]
        ):

            patient_id = row["patient_id"]

            if patient_id in train_patients:
                split = "train"

            elif patient_id in val_patients:
                split = "val"

            else:
                raise RuntimeError(
                    f"Patient {patient_id} has no split."
                )

            writer.writerow([
                patient_id,
                row["ptc_status"],
                row["histopathology"],
                row["n_images"],
                split,
            ])

    print(f"\nSaved permanent split:")
    print(f"  {csv_path}")

def save_patient_lists(
    train_patients,
    val_patients,
    metadata_dir,
):

    train_path = metadata_dir / "train_patients.txt"
    val_path = metadata_dir / "val_patients.txt"

    with open(train_path, "w", encoding="utf-8") as f:

        for patient_id in sorted(train_patients):
            f.write(patient_id + "\n")

    with open(val_path, "w", encoding="utf-8") as f:

        for patient_id in sorted(val_patients):
            f.write(patient_id + "\n")

    print(f"  {train_path}")
    print(f"  {val_path}")

# ============================================================
# SPLIT STATISTICS
# ============================================================
def print_split_statistics(
    patient_table,
    train_patients,
    val_patients,
):

    table = {
        row["patient_id"]: row
        for row in patient_table
    }

    def calculate(patient_ids):

        patients = [
            table[pid]
            for pid in patient_ids
        ]

        n_patients = len(patients)

        n_images = sum(
            row["n_images"]
            for row in patients
        )

        n_ptc = sum(
            row["ptc_status"]
            for row in patients
        )

        prevalence = (
            n_ptc / n_patients
            if n_patients
            else 0
        )

        return (
            n_patients,
            n_images,
            n_ptc,
            prevalence,
        )

    train = calculate(train_patients)
    val = calculate(val_patients)

    print("\n" + "=" * 70)
    print("PATIENT-LEVEL SPLIT")
    print("=" * 70)

    print(
        f"TRAIN: "
        f"{train[0]:4d} patients | "
        f"{train[1]:5d} images | "
        f"{train[2]:4d} PTC+ | "
        f"{100 * train[3]:6.2f}% PTC"
    )

    print(
        f"VAL:   "
        f"{val[0]:4d} patients | "
        f"{val[1]:5d} images | "
        f"{val[2]:4d} PTC+ | "
        f"{100 * val[3]:6.2f}% PTC"
    )

    print("=" * 70)

# ============================================================
# IMAGE LOOKUP
# ============================================================
def build_image_lookup(data):
    return {
        image["id"]: image
        for image in data["images"]
    }

# ============================================================
# ANNOTATION LOOKUP
# ============================================================
def build_annotation_lookup(data):
    lookup = {}

    for annotation in data["annotations"]:
        image_id = annotation["image_id"]
        lookup.setdefault(
            image_id,
            []
        ).append(annotation)

    return lookup

# ============================================================
# POLYGON -> YOLO SEGMENTATION
# ============================================================
def annotation_to_yolo_lines(
    annotation,
    image_width,
    image_height,
):

    segmentation = annotation.get(
        "segmentation"
    )

    if not segmentation:
        return []

    lines = []

    # COCO polygon format:
    # [
    #   [x1, y1, x2, y2, ...]
    # ]
    # or potentially multiple polygons.

    if not isinstance(segmentation, list):
        return []

    for polygon in segmentation:

        if not isinstance(polygon, list):
            continue

        # Minimum 3 points = 6 coordinates
        if len(polygon) < 6:
            continue

        if len(polygon) % 2 != 0:
            print(
                "WARNING: odd number of polygon coordinates; "
                "skipping polygon."
            )
            continue

        points = []

        for i in range(
            0,
            len(polygon),
            2
        ):

            x = float(polygon[i])
            y = float(polygon[i + 1])

            x_norm = x / image_width
            y_norm = y / image_height

            # Clip numerical errors
            x_norm = max(
                0.0,
                min(1.0, x_norm)
            )

            y_norm = max(
                0.0,
                min(1.0, y_norm)
            )

            points.extend([
                x_norm,
                y_norm,
            ])

        line = " ".join(
            [str(PTC_CLASS_ID)]
            + [
                f"{value:.6f}"
                for value in points
            ]
        )

        lines.append(line)

    return lines

# ============================================================
# PROCESS TRAIN / VAL
# ============================================================
def process_split(
    data,
    patient_ids,
    source_images_dir,
    output_images_dir,
    output_labels_dir,
    split_name,
):

    patient_ids = {
        normalise_patient_id(pid)
        for pid in patient_ids
    }

    annotation_lookup = build_annotation_lookup(data)

    selected_images = 0
    selected_ptc_annotations = 0
    missing_images = 0
    missing_segmentation = 0

    original_category_counts = Counter()

    for image in data["images"]:

        patient_id = normalise_patient_id(
            image["patient_id"]
        )

        if patient_id not in patient_ids:
            continue

        selected_images += 1

        file_name = image["file_name"]

        source_image = (
            source_images_dir /
            file_name
        )

        destination_image = (
            output_images_dir /
            file_name
        )

        if not source_image.exists():

            print(
                f"WARNING: image missing:\n"
                f"  {source_image}"
            )

            missing_images += 1
            continue
        # Copy image
        shutil.copy2(
            source_image,
            destination_image,
        )
        # Generate label
        label_path = (
            output_labels_dir /
            f"{Path(file_name).stem}.txt"
        )

        annotations = annotation_lookup.get(
            image["id"],
            []
        )

        yolo_lines = []

        for annotation in annotations:
            # Keep an audit trail of the original annotation
            # category, but do NOT use category_id as the PTC
            # definition.
            original_category_counts[
                annotation.get(
                    "category_id",
                    "missing"
                )
            ] += 1

            lines = annotation_to_yolo_lines(
                annotation,
                image["width"],
                image["height"],
            )

            if not lines:

                missing_segmentation += 1
                continue

            yolo_lines.extend(lines)

        # Every image receives a TXT file.
        # For a PTC-negative image, this file is empty.
        # This explicitly tells the dataset-building pipeline that the image contains no PTC object.

        with open(
            label_path,
            "w",
            encoding="utf-8"
        ) as f:

            if yolo_lines:
                f.write(
                    "\n".join(yolo_lines)
                    + "\n"
                )

                selected_ptc_annotations += len(
                    yolo_lines
                )

    print()
    print(f"{split_name.upper()} DATASET")
    print("-" * 50)
    print(f"Images processed:          {selected_images}")
    print(f"PTC polygons written:      {selected_ptc_annotations}")
    print(f"Missing images:             {missing_images}")
    print(f"Missing/invalid polygons:  {missing_segmentation}")

    print()
    print(
        "Original annotation category counts "
        "(AUDIT ONLY):"
    )

    for category, count in sorted(
        original_category_counts.items(),
        key=lambda x: str(x[0])
    ):

        print(
            f"  category_id={category}: {count}"
        )

    return {
        "images": selected_images,
        "ptc_polygons": selected_ptc_annotations,
        "missing_images": missing_images,
        "missing_segmentation": missing_segmentation,
        "original_category_counts":
            dict(original_category_counts),
    }

# ============================================================
# SUBSET JSON
# ============================================================
def create_subset_json(
    data,
    patient_ids,
    output_path,
):

    patient_ids = {
        normalise_patient_id(pid)
        for pid in patient_ids
    }

    selected_images = []

    for image in data["images"]:

        patient_id = normalise_patient_id(
            image["patient_id"]
        )

        if patient_id in patient_ids:
            selected_images.append(image)

    selected_image_ids = {
        image["id"]
        for image in selected_images
    }

    selected_annotations = [
        annotation
        for annotation in data["annotations"]
        if annotation["image_id"]
        in selected_image_ids
    ]

    selected_info = {}

    for patient_id_raw, record in data["info"].items():

        patient_id = normalise_patient_id(
            patient_id_raw
        )

        if patient_id in patient_ids:
            selected_info[patient_id_raw] = record

    subset = {
        "info": selected_info,
        "licenses": data.get(
            "licenses",
            {}
        ),
        "categories": data.get(
            "categories",
            []
        ),
        "images": selected_images,
        "annotations": selected_annotations,
    }

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            subset,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"Saved subset JSON: {output_path}"
    )

# ============================================================
# PROCESS TEST SET
# ============================================================
def process_test_set(
    test_data,
    test_images_dir,
    output_images_dir,
    output_labels_dir,
):

    test_patient_ids = set(
        normalise_patient_id(pid)
        for pid in test_data["info"].keys()
    )

    return process_split(
        data=test_data,
        patient_ids=test_patient_ids,
        source_images_dir=test_images_dir,
        output_images_dir=output_images_dir,
        output_labels_dir=output_labels_dir,
        split_name="test",
    )

# ============================================================
# DATASET YAML
# ============================================================
def create_dataset_yaml(output):

    yaml_path = output / "dataset.yaml"

    # Absolute path is deliberately used.
    # This avoids problems when training from another working
    # directory over SSH.

    yaml_content = f"""path: {output.as_posix()}

train: images/train
val: images/val
test: images/test

names:
  0: PTC
"""

    with open(
        yaml_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(yaml_content)

    print(
        f"\nSaved dataset.yaml:\n"
        f"  {yaml_path}"
    )


# ============================================================
# DATASET AUDIT
# ============================================================
def audit_dataset(output):

    print("\n" + "=" * 70)
    print("DATASET AUDIT")
    print("=" * 70)

    for split in [
        "train",
        "val",
        "test",
    ]:

        image_dir = (
            output /
            "images" /
            split
        )

        label_dir = (
            output /
            "labels" /
            split
        )

        images = list(
            image_dir.glob("*")
        )

        labels = list(
            label_dir.glob("*.txt")
        )

        image_stems = {
            image.stem
            for image in images
        }

        label_stems = {
            label.stem
            for label in labels
        }

        missing_labels = (
            image_stems - label_stems
        )

        extra_labels = (
            label_stems - image_stems
        )

        print()
        print(split.upper())
        print(f"  images: {len(images)}")
        print(f"  labels: {len(labels)}")
        print(
            f"  images without label file: "
            f"{len(missing_labels)}"
        )
        print(
            f"  labels without image: "
            f"{len(extra_labels)}"
        )

        if missing_labels:
            print(
                "  WARNING: some images have no "
                "corresponding label file."
            )

        if extra_labels:
            print(
                "  WARNING: some labels have no "
                "corresponding image."
            )

# ============================================================
# YOLO TEST TRAINING
# ============================================================
def run_yolo_test(
    output,
    epochs,
    device,
    workers,
):
    try:
        from ultralytics import YOLO
    except ImportError:
        raise RuntimeError(
            "\nUltralytics is not installed.\n"
            "Install it with:\n\n"
            "    pip install -U ultralytics\n"
        )

    dataset_yaml = output / "dataset.yaml"

    print("\n" + "=" * 70)
    print("YOLO26n-seg TEST RUN")
    print("=" * 70)

    print(
        f"Model:     {MODEL_NAME}"
    )
    print(
        f"Epochs:    {epochs}"
    )
    print(
        "imgsz:     DEFAULT (640)"
    )
    print(
        f"Dataset:   {dataset_yaml}"
    )
    if device is not None:
        print(
            f"Device:    {device}"
        )

    model = YOLO(
        MODEL_NAME
    )

    train_kwargs = {
        "data": str(dataset_yaml),
        "epochs": epochs,
        "workers": workers,
        "project": str(
            output / "runs"
        ),
        "name": "yolo26n_seg_test",
        "exist_ok": True,
        "pretrained": True,
    }

    if device is not None:
        train_kwargs["device"] = 0

    results = model.train(
        **train_kwargs
    )
    print("\nTraining finished.")
    return results

# ============================================================
# MAIN
# ============================================================
def main():

    args = parse_args()
    (
        source,
        output,
        train_json_path,
        train_images_dir,
        test_json_path,
        test_images_dir,
    ) = resolve_paths(args)

    print("=" * 70)
    print("ThyroidXL -> YOLO26-seg")
    print("=" * 70)

    print()
    print(f"Source: {source}")
    print(f"Output: {output}")

    # --------------------------------------------------------
    # Check inputs
    # --------------------------------------------------------
    if not train_json_path.exists():
        raise FileNotFoundError(
            f"Training JSON not found:\n"
            f"{train_json_path}"
        )

    if not train_images_dir.exists():
        raise FileNotFoundError(
            f"Training image directory not found:\n"
            f"{train_images_dir}"
        )
    # --------------------------------------------------------
    # Create directories
    # --------------------------------------------------------
    create_directories(output)

    metadata_dir = (
        output / "metadata"
    )
    # --------------------------------------------------------
    # Load training JSON
    # --------------------------------------------------------

    train_data = load_json(
        train_json_path
    )
    print()
    print(
        f"Training patients: "
        f"{len(train_data['info'])}"
    )

    print(
        f"Training images: "
        f"{len(train_data['images'])}"
    )

    print(
        f"Training annotations: "
        f"{len(train_data['annotations'])}"
    )
    # --------------------------------------------------------
    # Build patient table
    # --------------------------------------------------------
    patient_table = build_patient_table(
        train_data
    )

    n_ptc = sum(
        row["ptc_status"]
        for row in patient_table
    )

    n_non_ptc = (
        len(patient_table)
        - n_ptc
    )

    print()
    print(
        f"PTC-positive patients: "
        f"{n_ptc}"
    )

    print(
        f"PTC-negative patients: "
        f"{n_non_ptc}"
    )
    # --------------------------------------------------------
    # Load existing split OR create new one
    # --------------------------------------------------------
    split_csv = (
        metadata_dir /
        "patient_split.csv"
    )

    if split_csv.exists():

        print()
        print(
            "Existing patient_split.csv found."
        )

        print(
            "Reusing the existing split."
        )

        train_patients, val_patients = (
            load_existing_patient_split(
                split_csv
            )
        )

    else:

        print()
        print(
            "No existing patient split found."
        )

        print(
            "Creating a new stratified 80/20 split..."
        )

        train_patients, val_patients = (
            create_new_patient_split(
                patient_table
            )
        )

        save_patient_split(
            patient_table,
            train_patients,
            val_patients,
            metadata_dir,
        )

    # --------------------------------------------------------
    # Verify every patient is assigned exactly once
    # --------------------------------------------------------

    all_patients = {
        row["patient_id"]
        for row in patient_table
    }

    assigned_patients = (
        train_patients |
        val_patients
    )

    if all_patients != assigned_patients:

        missing = (
            all_patients -
            assigned_patients
        )

        extra = (
            assigned_patients -
            all_patients
        )

        raise RuntimeError(
            "Patient split mismatch.\n"
            f"Missing: {missing}\n"
            f"Extra: {extra}"
        )

    if train_patients & val_patients:

        raise RuntimeError(
            "Patient leakage detected."
        )

    # --------------------------------------------------------
    # Save patient lists
    # --------------------------------------------------------

    save_patient_lists(
        train_patients,
        val_patients,
        metadata_dir,
    )

    # --------------------------------------------------------
    # Print statistics
    # --------------------------------------------------------

    print_split_statistics(
        patient_table,
        train_patients,
        val_patients,
    )

    # --------------------------------------------------------
    # Process TRAIN
    # --------------------------------------------------------

    process_split(
        data=train_data,
        patient_ids=train_patients,
        source_images_dir=train_images_dir,
        output_images_dir=(
            output / "images" / "train"
        ),
        output_labels_dir=(
            output / "labels" / "train"
        ),
        split_name="train",
    )

    # --------------------------------------------------------
    # Process VAL
    # --------------------------------------------------------

    process_split(
        data=train_data,
        patient_ids=val_patients,
        source_images_dir=train_images_dir,
        output_images_dir=(
            output / "images" / "val"
        ),
        output_labels_dir=(
            output / "labels" / "val"
        ),
        split_name="val",
    )

    # --------------------------------------------------------
    # Save train/val JSON
    # --------------------------------------------------------

    create_subset_json(
        train_data,
        train_patients,
        metadata_dir /
        "train_annotations.json",
    )

    create_subset_json(
        train_data,
        val_patients,
        metadata_dir /
        "val_annotations.json",
    )

    # --------------------------------------------------------
    # Process TEST
    # --------------------------------------------------------

    if test_json_path.exists() and test_images_dir.exists():

        test_data = load_json(
            test_json_path
        )

        print()
        print(
            f"Test patients: "
            f"{len(test_data['info'])}"
        )

        print(
            f"Test images: "
            f"{len(test_data['images'])}"
        )

        process_test_set(
            test_data,
            test_images_dir,
            output / "images" / "test",
            output / "labels" / "test",
        )

        shutil.copy2(
            test_json_path,
            metadata_dir /
            "test_annotations.json",
        )

    else:

        print()
        print(
            "WARNING: test JSON or test images not found."
        )

        print(
            f"Expected JSON:   {test_json_path}"
        )

        print(
            f"Expected images: {test_images_dir}"
        )

        print(
            "Skipping test processing."
        )

    # --------------------------------------------------------
    # Create dataset.yaml
    # --------------------------------------------------------
    create_dataset_yaml(
        output
    )
    # --------------------------------------------------------
    # Audit
    # --------------------------------------------------------
    audit_dataset(
        output
    )
    # --------------------------------------------------------
    # Stop here if requested
    # --------------------------------------------------------
    if args.prepare_only:

        print()
        print("=" * 70)
        print("DATASET PREPARATION COMPLETE")
        print("=" * 70)

        return

    # --------------------------------------------------------
    # Optional YOLO test run
    # --------------------------------------------------------

    if args.test_run:

        run_yolo_test(
            output=output,
            epochs=args.epochs,
            device=args.device,
            workers=args.workers,
        )

    else:

        print()
        print("=" * 70)
        print("DATASET PREPARATION COMPLETE")
        print("=" * 70)

        print()
        print(
            "No YOLO training was requested."
        )

        print()
        print(
            "To run a short test training:"
        )

        print(
            "    python run_thyroidxl.py "
            "--source ... "
            "--output ... "
            "--test-run"
        )


if __name__ == "__main__":
    main()
