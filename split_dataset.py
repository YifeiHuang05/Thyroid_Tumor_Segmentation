"""
ThyroidXL patient-level train/validation split.

What this script does
---------------------
1. Reads the original training COCO-style JSON.
2. Determines PTC status at the PATIENT level from Histopathology.
3. Performs an 80:20 stratified patient-level split.
4. Saves the split permanently in metadata/patient_split.csv.
5. Copies images into:
       images/train/
       images/val/
6. Converts polygon segmentation annotations into YOLO segmentation format:
       labels/train/
       labels/val/
7. Writes train/val JSON subsets for traceability.
8. Optionally copies the existing test set and converts its annotations.
9. Creates dataset.yaml.

IMPORTANT
---------
The test set is NEVER used for train/validation splitting.
The same patient can never occur in both train and validation.
"""

from pathlib import Path
import json
import shutil
import csv
from collections import Counter

from sklearn.model_selection import train_test_split


# ============================================================
# 1. CONFIGURATION
# ============================================================

# -------- Original dataset --------

SOURCE_ROOT = Path("./data/ThyroidXL")

TRAIN_IMAGES_DIR = SOURCE_ROOT / "train" / "images"
TRAIN_JSON = SOURCE_ROOT / "train" / "train_annotations.json"

# Existing test set
TEST_IMAGES_DIR = SOURCE_ROOT / "test" / "images"
TEST_JSON = SOURCE_ROOT / "test" / "test_annotations.json"


# -------- New output dataset --------

OUTPUT_ROOT = Path("./data/ThyroidXL_new")


# -------- Split parameters --------

VAL_SIZE = 0.20
RANDOM_SEED = 2026


# ============================================================
# PTC DEFINITION
# ============================================================

# A patient is considered PTC-positive if ANY of their
# histopathology fields contains one of these diagnoses.

PTC_POSITIVE_TERMS = {
    "papillary thyroid carcinoma",
    "micro-papillary thyroid carcinoma",
    "papillary thyroid carcinoma, chronic thyroiditis",
}


# ============================================================
# OPTIONAL TEST SET
# ============================================================

# Set this to True if you want this script to populate:
#
#   images/test/
#   labels/test/
#
# from your existing test set.
#
# The test set itself is NOT involved in the train/val split.

PROCESS_TEST_SET = True


# ============================================================
# 2. OUTPUT DIRECTORIES
# ============================================================

IMAGES_TRAIN = OUTPUT_ROOT / "images" / "train"
IMAGES_VAL = OUTPUT_ROOT / "images" / "val"
IMAGES_TEST = OUTPUT_ROOT / "images" / "test"

LABELS_TRAIN = OUTPUT_ROOT / "labels" / "train"
LABELS_VAL = OUTPUT_ROOT / "labels" / "val"
LABELS_TEST = OUTPUT_ROOT / "labels" / "test"

METADATA_DIR = OUTPUT_ROOT / "metadata"


def make_directories():
    """Create the complete output directory structure."""

    directories = [
        IMAGES_TRAIN,
        IMAGES_VAL,
        IMAGES_TEST,
        LABELS_TRAIN,
        LABELS_VAL,
        LABELS_TEST,
        METADATA_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# 3. LOAD JSON
# ============================================================

def load_json(path):
    print(f"Loading: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# 4. DETERMINE PTC STATUS
# ============================================================

def get_histopathology_values(patient_record):
    """
    Extract all available Histopathology values from nodule_1
    and nodule_2.
    """

    values = []

    for nodule_key in ["nodule_1", "nodule_2"]:

        nodule = patient_record.get(nodule_key)

        if not nodule:
            continue

        histopathology = nodule.get("Histopathology")

        if histopathology:
            values.append(str(histopathology).strip())

    return values


def is_ptc_positive(patient_record):
    """
    Return True if the patient's histopathology matches one
    of the predefined PTC-positive diagnoses.
    """

    histopathology_values = get_histopathology_values(patient_record)

    for diagnosis in histopathology_values:

        diagnosis_lower = diagnosis.lower().strip()

        for positive_term in PTC_POSITIVE_TERMS:

            if diagnosis_lower == positive_term.lower():
                return True

    return False


# ============================================================
# 5. BUILD PATIENT-LEVEL TABLE
# ============================================================

def build_patient_table(data):

    patient_table = []

    for patient_id, patient_record in data["info"].items():

        ptc_status = int(is_ptc_positive(patient_record))

        image_names = patient_record.get("images", [])

        histopathology = get_histopathology_values(patient_record)

        patient_table.append({
            "patient_id": patient_id,
            "ptc_status": ptc_status,
            "histopathology": " | ".join(histopathology),
            "n_images": len(image_names),
        })

    return patient_table


# ============================================================
# 6. PATIENT-LEVEL STRATIFIED SPLIT
# ============================================================

def create_patient_split(patient_table):

    patient_ids = [
        row["patient_id"]
        for row in patient_table
    ]

    ptc_labels = [
        row["ptc_status"]
        for row in patient_table
    ]

    train_patients, val_patients = train_test_split(
        patient_ids,
        test_size=VAL_SIZE,
        random_state=RANDOM_SEED,
        stratify=ptc_labels,
    )

    train_patients = set(train_patients)
    val_patients = set(val_patients)

    # Safety check
    overlap = train_patients.intersection(val_patients)

    if overlap:
        raise RuntimeError(
            f"Patient leakage detected! "
            f"{len(overlap)} patients occur in both train and val."
        )

    return train_patients, val_patients


# ============================================================
# 7. SAVE PATIENT SPLIT
# ============================================================

def save_patient_split(patient_table, train_patients, val_patients):

    csv_path = METADATA_DIR / "patient_split.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:

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
                    f"Patient {patient_id} not assigned to a split."
                )

            writer.writerow([
                patient_id,
                row["ptc_status"],
                row["histopathology"],
                row["n_images"],
                split,
            ])

    print(f"Saved: {csv_path}")


def save_patient_lists(train_patients, val_patients):

    train_txt = METADATA_DIR / "train_patients.txt"
    val_txt = METADATA_DIR / "val_patients.txt"

    with open(train_txt, "w", encoding="utf-8") as f:

        for patient_id in sorted(train_patients):
            f.write(f"{patient_id}\n")

    with open(val_txt, "w", encoding="utf-8") as f:

        for patient_id in sorted(val_patients):
            f.write(f"{patient_id}\n")

    print(f"Saved: {train_txt}")
    print(f"Saved: {val_txt}")


# ============================================================
# 8. CREATE IMAGE LOOKUP
# ============================================================

def build_image_lookup(data):

    """
    Maps image_id -> image metadata.
    """

    lookup = {}

    for image in data["images"]:

        lookup[image["id"]] = image

    return lookup


# ============================================================
# 9. CREATE ANNOTATION LOOKUP
# ============================================================

def build_annotation_lookup(data):

    """
    Maps image_id -> list of annotations.
    """

    lookup = {}

    for annotation in data["annotations"]:

        image_id = annotation["image_id"]

        if image_id not in lookup:
            lookup[image_id] = []

        lookup[image_id].append(annotation)

    return lookup


# ============================================================
# 10. CONVERT COCO POLYGON TO YOLO SEGMENTATION
# ============================================================

def polygon_to_yolo(annotation, image_width, image_height):

    """
    Convert one COCO polygon annotation into YOLO segmentation format.

    YOLO segmentation format:

        class_id x1 y1 x2 y2 x3 y3 ...

    Coordinates are normalized to [0, 1].
    """

    category_id = annotation["category_id"]

    segmentation = annotation.get("segmentation")

    if not segmentation:
        return None

    # We expect polygon segmentation.
    # COCO polygon segmentation is usually:
    #
    # [
    #     [x1, y1, x2, y2, ...]
    # ]

    if not isinstance(segmentation, list):
        return None

    lines = []

    for polygon in segmentation:

        if not isinstance(polygon, list):
            continue

        if len(polygon) < 6:
            continue

        # Need pairs of x/y coordinates.
        if len(polygon) % 2 != 0:
            print(
                "WARNING: odd number of polygon coordinates; "
                "skipping malformed polygon."
            )
            continue

        normalized_points = []

        for i in range(0, len(polygon), 2):

            x = polygon[i]
            y = polygon[i + 1]

            x_norm = x / image_width
            y_norm = y / image_height

            # Safety clipping
            x_norm = max(0.0, min(1.0, x_norm))
            y_norm = max(0.0, min(1.0, y_norm))

            normalized_points.extend([
                x_norm,
                y_norm,
            ])

        line = " ".join(
            [str(category_id)] +
            [f"{x:.6f}" for x in normalized_points]
        )

        lines.append(line)

    if not lines:
        return None

    return lines


# ============================================================
# 11. COPY IMAGES + GENERATE YOLO LABELS
# ============================================================

def process_split(
    data,
    patient_ids,
    source_images_dir,
    destination_images_dir,
    destination_labels_dir,
):

    image_lookup = build_image_lookup(data)
    annotation_lookup = build_annotation_lookup(data)

    patient_ids = set(patient_ids)

    selected_images = 0
    selected_annotations = 0
    missing_images = 0
    missing_segmentation = 0

    for image in data["images"]:

        patient_id = str(image["patient_id"]).zfill(8)

        if patient_id not in patient_ids:
            continue

        selected_images += 1

        file_name = image["file_name"]

        source_image = source_images_dir / file_name
        destination_image = destination_images_dir / file_name

        if not source_image.exists():

            print(
                f"WARNING: image not found: {source_image}"
            )

            missing_images += 1
            continue

        # ----------------------------------------------------
        # Copy image
        # ----------------------------------------------------

        shutil.copy2(
            source_image,
            destination_image,
        )

        # ----------------------------------------------------
        # Generate YOLO segmentation label
        # ----------------------------------------------------

        label_file = (
            destination_labels_dir /
            f"{Path(file_name).stem}.txt"
        )

        annotations = image_lookup.get(
            image["id"],
            []
        )

        yolo_lines = []

        for annotation in annotation_lookup.get(
            image["id"],
            []
        ):

            lines = polygon_to_yolo(
                annotation,
                image["width"],
                image["height"],
            )

            if lines is None:

                missing_segmentation += 1
                continue

            yolo_lines.extend(lines)

            selected_annotations += 1

        # Write label file.
        #
        # If an image somehow has no valid segmentation,
        # an empty label file is still created.

        with open(
            label_file,
            "w",
            encoding="utf-8"
        ) as f:

            if yolo_lines:
                f.write(
                    "\n".join(yolo_lines) + "\n"
                )

    print()
    print("Split processing complete")
    print(f"Images: {selected_images}")
    print(f"Annotations: {selected_annotations}")
    print(f"Missing images: {missing_images}")
    print(f"Invalid/missing segmentation: {missing_segmentation}")

    return {
        "images": selected_images,
        "annotations": selected_annotations,
        "missing_images": missing_images,
        "missing_segmentation": missing_segmentation,
    }


# ============================================================
# 12. SAVE SUBSET JSON
# ============================================================

def create_subset_json(
    data,
    patient_ids,
    output_path,
):

    patient_ids = set(patient_ids)

    selected_images = []

    for image in data["images"]:

        patient_id = str(
            image["patient_id"]
        ).zfill(8)

        if patient_id in patient_ids:
            selected_images.append(image)

    selected_image_ids = {
        image["id"]
        for image in selected_images
    }

    selected_annotations = [
        annotation
        for annotation in data["annotations"]
        if annotation["image_id"] in selected_image_ids
    ]

    subset = {
        "info": {
            patient_id: data["info"][patient_id]
            for patient_id in patient_ids
            if patient_id in data["info"]
        },
        "licenses": data.get("licenses", {}),
        "categories": data["categories"],
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

    print(f"Saved subset JSON: {output_path}")


# ============================================================
# 13. DATASET STATISTICS
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

    def stats(patient_ids):

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
            if n_patients > 0
            else 0
        )

        return (
            n_patients,
            n_images,
            n_ptc,
            prevalence,
        )

    train_stats = stats(train_patients)
    val_stats = stats(val_patients)

    print()
    print("=" * 70)
    print("PATIENT-LEVEL SPLIT")
    print("=" * 70)

    print(
        f"TRAIN: "
        f"{train_stats[0]:4d} patients | "
        f"{train_stats[1]:5d} images | "
        f"{train_stats[2]:4d} PTC+ | "
        f"{train_stats[3] * 100:.2f}% PTC"
    )

    print(
        f"VAL:   "
        f"{val_stats[0]:4d} patients | "
        f"{val_stats[1]:5d} images | "
        f"{val_stats[2]:4d} PTC+ | "
        f"{val_stats[3] * 100:.2f}% PTC"
    )

    print("=" * 70)


# ============================================================
# 14. CREATE DATASET.YAML
# ============================================================

def create_dataset_yaml():

    yaml_path = OUTPUT_ROOT / "dataset.yaml"

    yaml_content = """path: .
train: images/train
val: images/val
test: images/test

names:
  0: benign
  1: malignant
"""

    with open(
        yaml_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(yaml_content)

    print(f"Saved: {yaml_path}")


# ============================================================
# 15. MAIN
# ============================================================

def main():

    print("=" * 70)
    print("ThyroidXL Patient-Level Dataset Split")
    print("=" * 70)

    # --------------------------------------------------------
    # Check input paths
    # --------------------------------------------------------

    if not TRAIN_JSON.exists():
        raise FileNotFoundError(
            f"Training JSON not found:\n{TRAIN_JSON}"
        )

    if not TRAIN_IMAGES_DIR.exists():
        raise FileNotFoundError(
            f"Training image directory not found:\n"
            f"{TRAIN_IMAGES_DIR}"
        )

    # --------------------------------------------------------
    # Create output directories
    # --------------------------------------------------------

    make_directories()

    # --------------------------------------------------------
    # Load training JSON
    # --------------------------------------------------------

    train_data = load_json(TRAIN_JSON)

    print(
        f"Patients: {len(train_data['info'])}"
    )

    print(
        f"Images: {len(train_data['images'])}"
    )

    print(
        f"Annotations: {len(train_data['annotations'])}"
    )

    # --------------------------------------------------------
    # Build patient-level table
    # --------------------------------------------------------

    patient_table = build_patient_table(
        train_data
    )

    ptc_count = sum(
        row["ptc_status"]
        for row in patient_table
    )

    print(
        f"PTC-positive patients: "
        f"{ptc_count}"
    )

    print(
        f"PTC-negative patients: "
        f"{len(patient_table) - ptc_count}"
    )

    # --------------------------------------------------------
    # Create stratified patient split
    # --------------------------------------------------------

    train_patients, val_patients = create_patient_split(
        patient_table
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
    # Save permanent patient split
    # --------------------------------------------------------

    save_patient_split(
        patient_table,
        train_patients,
        val_patients,
    )

    save_patient_lists(
        train_patients,
        val_patients,
    )

    # --------------------------------------------------------
    # Copy TRAIN images + generate labels
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("PROCESSING TRAIN")
    print("=" * 70)

    process_split(
        data=train_data,
        patient_ids=train_patients,
        source_images_dir=TRAIN_IMAGES_DIR,
        destination_images_dir=IMAGES_TRAIN,
        destination_labels_dir=LABELS_TRAIN,
    )

    # --------------------------------------------------------
    # Copy VAL images + generate labels
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("PROCESSING VALIDATION")
    print("=" * 70)

    process_split(
        data=train_data,
        patient_ids=val_patients,
        source_images_dir=TRAIN_IMAGES_DIR,
        destination_images_dir=IMAGES_VAL,
        destination_labels_dir=LABELS_VAL,
    )

    # --------------------------------------------------------
    # Save train/val JSON
    # --------------------------------------------------------

    create_subset_json(
        train_data,
        train_patients,
        METADATA_DIR / "train_annotations.json",
    )

    create_subset_json(
        train_data,
        val_patients,
        METADATA_DIR / "val_annotations.json",
    )

    # --------------------------------------------------------
    # PROCESS EXISTING TEST SET
    # --------------------------------------------------------

    if PROCESS_TEST_SET:

        print()
        print("=" * 70)
        print("PROCESSING TEST SET")
        print("=" * 70)

        if not TEST_JSON.exists():

            print(
                "WARNING: test JSON not found:"
            )

            print(TEST_JSON)

            print(
                "Skipping test processing."
            )

        elif not TEST_IMAGES_DIR.exists():

            print(
                "WARNING: test image directory not found:"
            )

            print(TEST_IMAGES_DIR)

            print(
                "Skipping test processing."
            )

        else:

            test_data = load_json(
                TEST_JSON
            )

            # All test patients are processed.
            test_patient_ids = set(
                test_data["info"].keys()
            )

            process_split(
                data=test_data,
                patient_ids=test_patient_ids,
                source_images_dir=TEST_IMAGES_DIR,
                destination_images_dir=IMAGES_TEST,
                destination_labels_dir=LABELS_TEST,
            )

            # Save a copy of the test JSON for traceability.
            shutil.copy2(
                TEST_JSON,
                METADATA_DIR /
                "test_annotations.json",
            )

    # --------------------------------------------------------
    # dataset.yaml
    # --------------------------------------------------------

    create_dataset_yaml()

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("DATASET CREATION COMPLETE")
    print("=" * 70)

    print()
    print(f"Output: {OUTPUT_ROOT}")

    print()
    print("Directory structure:")
    print()
    print("ThyroidXL/")
    print("├── images/")
    print("│   ├── train/")
    print("│   ├── val/")
    print("│   └── test/")
    print("│")
    print("├── labels/")
    print("│   ├── train/")
    print("│   ├── val/")
    print("│   └── test/")
    print("│")
    print("├── metadata/")
    print("│   ├── patient_split.csv")
    print("│   ├── train_patients.txt")
    print("│   ├── val_patients.txt")
    print("│   ├── train_annotations.json")
    print("│   ├── val_annotations.json")
    print("│   └── test_annotations.json")
    print("│")
    print("└── dataset.yaml")
    print()
    print("IMPORTANT:")
    print("This patient split is now fixed.")
    print("Reuse patient_split.csv for all future experiments.")


if __name__ == "__main__":
    main()