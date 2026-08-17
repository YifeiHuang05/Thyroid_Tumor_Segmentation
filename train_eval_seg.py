from pathlib import Path
import csv
import time

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO


# ============================================================
# Configuration
# ============================================================

DATASET_YAML = Path(
    "data\ThyroidXL_clean\segmentation\dataset.yaml"
)

MODEL_NAME = "yolo26n-seg"

RUNS_DIR = Path(
    "runs_clean_seg"
)

RUN_NAME = "thyroidxl_seg_clean"

# Training
EPOCHS = 40
PATIENCE = 8
#IMGSZ = 640
BATCH = -1
DEVICE = 0

# Inference threshold used for the reported test IoU.
CONF = 0.25


# ============================================================
# Helper: convert YOLO polygon label to binary mask
# ============================================================

def yolo_label_to_mask(label_path, width, height):
    """
    Read a YOLO segmentation label and convert all polygons
    into one binary nodule mask.

    Class 0 = nodule.
    """

    mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    if not label_path.exists():
        return mask

    with open(
        label_path,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            values = line.split()

            class_id = int(values[0])

            # Only class 0 exists in our clean dataset.
            if class_id != 0:
                continue

            coords = np.asarray(
                [float(x) for x in values[1:]],
                dtype=np.float32
            )

            if len(coords) < 6:
                continue

            points = coords.reshape(-1, 2)

            # YOLO coordinates are normalized.
            points[:, 0] *= width
            points[:, 1] *= height

            points = np.round(points).astype(np.int32)

            cv2.fillPoly(
                mask,
                [points],
                1
            )

    return mask


# ============================================================
# Helper: convert YOLO prediction to binary mask
# ============================================================

def prediction_to_mask(result):
    """
    Combine all predicted segmentation masks into one binary
    nodule mask.

    Since the clean dataset has only one class (nodule), all
    predicted masks are treated as nodule masks.
    """

    height, width = result.orig_shape

    mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    if result.masks is None:
        return mask

    # result.masks.xy contains polygons in original-image
    # coordinates.
    polygons = result.masks.xy

    for polygon in polygons:

        if polygon is None or len(polygon) < 3:
            continue

        polygon = np.asarray(
            polygon,
            dtype=np.float32
        )

        # Round the polygon coordinates to the nearest integer.
        polygon = np.round(
            polygon
        ).astype(np.int32)

        cv2.fillPoly(
            mask,
            [polygon],
            1
        )

    return mask


# ============================================================
# IoU
# ============================================================

def calculate_iou(pred_mask, gt_mask):
    """
    Binary segmentation IoU:

        intersection / union
    """

    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)

    intersection = np.logical_and(
        pred,
        gt
    ).sum()

    union = np.logical_or(
        pred,
        gt
    ).sum()

    # Both masks empty.
    if union == 0:
        return 1.0

    return intersection / union


# ============================================================
# Train
# ============================================================

def train_model():

    print("=" * 70)
    print("STARTING YOLO26n-SEG TRAINING")
    print("=" * 70)

    print(f"Dataset: {DATASET_YAML}")
    print(f"Model:   {MODEL_NAME}")
    print(f"Epochs:  {EPOCHS}")
    print(f"Patience: {PATIENCE}")
    #print(f"Image size: {IMGSZ}")
    print(f"Batch:   {BATCH}")
    print(f"Device:  {DEVICE}")

    model = YOLO(
        "model/yolo26n-seg.pt"
    )

    start_time = time.time()

    results = model.train(
        data=str(DATASET_YAML),

        epochs=EPOCHS,
        patience=PATIENCE,

        #imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,

        project=str(RUNS_DIR),
        name=RUN_NAME,

        # Save best.pt and last.pt
        save=True,
        save_period=-1,

        # Validation during training
        val=True,

        # Reproducibility
        seed=42,

        # Use the standard YOLO augmentation defaults.
        # Do not manually alter them under deadline.
        pretrained=True,

        # Keep output reasonably clean.
        verbose=False,
    )

    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"Training time: "
        f"{elapsed / 3600:.2f} hours"
    )

    best_model = (
        "runs/segment/runs_clean_seg/thyroidxl_seg_clean/weights/best.pt"
    )

    last_model = (
        "runs/segment/runs_clean_seg/thyroidxl_seg_clean/weights/last.pt"
    )

    print(f"Best model: {best_model}")
    print(f"Last model: {last_model}")

    if not best_model.exists():
        raise FileNotFoundError(
            f"best.pt was not found at:\n{best_model}"
        )

    return best_model


# ============================================================
# Evaluate official test set
# ============================================================

def evaluate_test(model_path):

    print("\n" + "=" * 70)
    print("EVALUATING OFFICIAL TEST SET")
    print("=" * 70)

    test_image_dir = (
        DATASET_YAML.parent
        / "images"
        / "test"
    )

    test_label_dir = (
        DATASET_YAML.parent
        / "labels"
        / "test"
    )

    if not test_image_dir.exists():
        raise FileNotFoundError(
            f"Test image directory not found:\n"
            f"{test_image_dir}"
        )

    if not test_label_dir.exists():
        raise FileNotFoundError(
            f"Test label directory not found:\n"
            f"{test_label_dir}"
        )

    model = YOLO(
        str(model_path)
    )

    image_paths = sorted(
        [
            p for p in test_image_dir.iterdir()
            if p.is_file()
        ]
    )

    print(
        f"Test images: {len(image_paths)}"
    )

    rows = []

    for index, image_path in enumerate(
        image_paths,
        start=1
    ):

        # Read image to get original dimensions.
        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            print(
                f"WARNING: could not read "
                f"{image_path.name}"
            )
            continue

        height, width = image.shape[:2]

        label_path = (
            test_label_dir
            / f"{image_path.stem}.txt"
        )

        gt_mask = yolo_label_to_mask(
            label_path,
            width,
            height
        )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prediction_start = time.perf_counter()

        results = model.predict(
            source=str(image_path),
            #imgsz=IMGSZ,
            conf=CONF,
            device=DEVICE,
            verbose=False
        )

        prediction_time = (
            time.perf_counter()
            - prediction_start
        )

        result = results[0]

        pred_mask = prediction_to_mask(
            result
        )

        # ----------------------------------------------------
        # IoU
        # ----------------------------------------------------

        iou = calculate_iou(
            pred_mask,
            gt_mask
        )

        gt_area = int(
            gt_mask.sum()
        )

        pred_area = int(
            pred_mask.sum()
        )

        intersection = int(
            np.logical_and(
                pred_mask.astype(bool),
                gt_mask.astype(bool)
            ).sum()
        )

        union = int(
            np.logical_or(
                pred_mask.astype(bool),
                gt_mask.astype(bool)
            ).sum()
        )

        has_prediction = int(
            pred_area > 0
        )

        rows.append({
            "image": image_path.name,
            "iou": iou,
            "gt_area": gt_area,
            "pred_area": pred_area,
            "intersection": intersection,
            "union": union,
            "has_prediction": has_prediction,
            "inference_time_ms":
                prediction_time * 1000
        })

        if index % 100 == 0:
            print(
                f"Processed "
                f"{index}/{len(image_paths)}"
            )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        rows
    )

    evaluation_dir = (
        RUNS_DIR
        / RUN_NAME
        / "test_evaluation"
    )

    evaluation_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    csv_path = (
        evaluation_dir
        / "segmentation_iou_per_image.csv"
    )

    results_df.to_csv(
        csv_path,
        index=False
    )

    # --------------------------------------------------------
    # Overall metrics
    # --------------------------------------------------------

    mean_iou = results_df[
        "iou"
    ].mean()

    median_iou = results_df[
        "iou"
    ].median()

    std_iou = results_df[
        "iou"
    ].std()

    detection_rate = results_df[
        "has_prediction"
    ].mean()

    mean_inference_ms = results_df[
        "inference_time_ms"
    ].mean()

    summary = {
        "model": str(model_path),
        "test_images": len(results_df),
        "confidence_threshold": CONF,
        "mean_IoU": float(mean_iou),
        "median_IoU": float(median_iou),
        "std_IoU": float(std_iou),
        "prediction_presence_rate": float(
            detection_rate
        ),
        "mean_inference_time_ms": float(
            mean_inference_ms
        )
    }

    summary_path = (
        evaluation_dir
        / "segmentation_iou_summary.csv"
    )

    pd.DataFrame(
        [summary]
    ).to_csv(
        summary_path,
        index=False
    )

    print("\n" + "=" * 70)
    print("TEST SEGMENTATION RESULTS")
    print("=" * 70)

    print(
        f"Images evaluated:       {len(results_df)}"
    )

    print(
        f"Mean IoU:               {mean_iou:.4f}"
    )

    print(
        f"Median IoU:             {median_iou:.4f}"
    )

    print(
        f"Std IoU:                {std_iou:.4f}"
    )

    print(
        f"Prediction presence:    "
        f"{detection_rate:.4f}"
    )

    print(
        f"Mean inference time:    "
        f"{mean_inference_ms:.2f} ms"
    )

    print(
        f"\nPer-image results:"
        f"\n{csv_path}"
    )

    print(
        f"\nSummary:"
        f"\n{summary_path}"
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    #best_model_path = train_model()
    best_model_path = "runs/segment/runs_clean_seg/thyroidxl_seg_clean/weights/best.pt"
    evaluate_test(
        best_model_path
    )