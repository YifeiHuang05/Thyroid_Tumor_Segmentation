from pathlib import Path
import argparse
import csv
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
def load_ground_truth_labels(dataset_root):
    """Ground truth PTC status from YOLO segmentation labels: 0=non-PTC, 1=PTC."""
    image_dir = Path(dataset_root) / "images" / "test"
    label_dir = Path(dataset_root) / "labels" / "test"

    ground_truth = {}

    for image_path in sorted(image_dir.iterdir()):
        if not image_path.is_file():
            continue

        label_path = label_dir / f"{image_path.stem}.txt"

        if not label_path.exists():
            raise FileNotFoundError(f"Missing label file:\n{label_path}")

        classes = []

        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    classes.append(int(line.split()[0]))

        if not classes:
            raise RuntimeError(
                f"No segmentation annotation found for {image_path}."
            )

        if any(c not in (0, 1) for c in classes):
            raise RuntimeError(
                f"Unexpected class ID in {label_path}: {classes}"
            )

        # Image-level ground truth:
        # if at least one PTC object exists, image = PTC.
        ground_truth[image_path.stem] = int(1 in classes)

    return ground_truth


def image_level_evaluation(model_path, dataset_root, device, conf, output_dir):
    print("\n" + "=" * 70)
    print("IMAGE-LEVEL PTC TEST SET EVALUATION")
    print("=" * 70)

    model_path = Path(model_path)
    dataset_root = Path(dataset_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model:          {model_path}")
    print(f"Dataset:        {dataset_root}")
    print(f"Test images:    {dataset_root / 'images' / 'test'}")
    print(f"Confidence:     {conf}")
    print(f"Device:         {device}")

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found:\n{model_path}")

    image_dir = dataset_root / "images" / "test"

    if not image_dir.exists():
        raise FileNotFoundError(f"Test image directory not found:\n{image_dir}")

    # Load model.
    model = YOLO(str(model_path))

    # Load image-level ground truth.
    ground_truth = load_ground_truth_labels(dataset_root)

    image_paths = [
        p for p in sorted(image_dir.iterdir())
        if p.is_file()
    ]

    if len(image_paths) == 0:
        raise RuntimeError(f"No test images found in:\n{image_dir}")

    print(f"\nNumber of test images: {len(image_paths)}")

    y_true = []
    y_pred = []
    inference_times = []
    output_rows = []

    # Optional warm-up to avoid including initial CUDA/model setup overhead
    # in the reported per-image inference time.
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        print("\nRunning GPU warm-up...")
        warmup_image = str(image_paths[0])
        _ = model.predict(
            source=warmup_image,
            conf=conf,
            device=device,
            verbose=False,
            save=False,
        )
        torch.cuda.synchronize()

    print("\nRunning test inference...")

    for i, image_path in enumerate(image_paths, start=1):
        stem = image_path.stem

        if stem not in ground_truth:
            raise KeyError(
                f"No ground-truth label found for test image: {image_path.name}"
            )

        if torch.cuda.is_available() and str(device).startswith("cuda"):
            torch.cuda.synchronize()

        start = time.perf_counter()

        results = model.predict(
            source=str(image_path),
            conf=conf,
            device=device,
            verbose=False,
            save=False,
        )

        if torch.cuda.is_available() and str(device).startswith("cuda"):
            torch.cuda.synchronize()

        inference_ms = (time.perf_counter() - start) * 1000
        inference_times.append(inference_ms)

        result = results[0]

        predicted_classes = []

        if result.boxes is not None and len(result.boxes) > 0:
            predicted_classes = [
                int(c)
                for c in result.boxes.cls.cpu().numpy()
            ]

        # Image-level classification rule:
        # at least one predicted PTC object -> PTC image.
        pred = int(1 in predicted_classes)

        gt = ground_truth[stem]

        y_true.append(gt)
        y_pred.append(pred)

        output_rows.append({
            "image": image_path.name,
            "ground_truth": gt,
            "prediction": pred,
            "n_predictions": len(predicted_classes),
            "predicted_classes": ",".join(
                map(str, predicted_classes)
            ),
            "inference_ms": inference_ms,
        })

        if i % 100 == 0 or i == len(image_paths):
            print(f"  Processed {i}/{len(image_paths)} images")

    # ---------------------------------------------------------
    # Calculate metrics
    # ---------------------------------------------------------

    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )
    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    tn, fp, fn, tp = cm.ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    sensitivity = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    mean_inference_ms = float(np.mean(inference_times))
    median_inference_ms = float(np.median(inference_times))
    p95_inference_ms = float(
        np.percentile(inference_times, 95)
    )

    # ---------------------------------------------------------
    # Print results
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL TEST SET RESULTS")
    print("=" * 70)

    print(f"\nNumber of test images: {len(y_true)}")

    print("\nImage-level PTC metrics:")
    print(f"Accuracy:             {accuracy:.4f}")
    print(f"Balanced accuracy:    {balanced_accuracy:.4f}")
    print(f"Macro F1:              {macro_f1:.4f}")
    print(f"F1:                    {f1:.4f}")
    print(f"Precision:             {precision:.4f}")
    print(f"Sensitivity/Recall:    {sensitivity:.4f}")
    print(f"Specificity:           {specificity:.4f}")

    print("\nConfusion matrix:")
    print("                 Predicted")
    print("                 Negative Positive")
    print(f"Actual Negative  {tn:8d} {fp:8d}")
    print(f"Actual Positive  {fn:8d} {tp:8d}")

    print("\nInference time:")
    print(f"Mean:                {mean_inference_ms:.2f} ms/image")
    print(f"Median:              {median_inference_ms:.2f} ms/image")
    print(f"P95:                 {p95_inference_ms:.2f} ms/image")

    # ---------------------------------------------------------
    # Save image-level predictions
    # ---------------------------------------------------------

    csv_path = output_dir / "image_level_predictions.csv"

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image",
                "ground_truth",
                "prediction",
                "n_predictions",
                "predicted_classes",
                "inference_ms",
            ],
        )
        writer.writeheader()
        writer.writerows(output_rows)

    # ---------------------------------------------------------
    # Save metrics
    # ---------------------------------------------------------

    metrics = {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "f1": f1,
        "precision": precision,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "mean_inference_ms": mean_inference_ms,
        "median_inference_ms": median_inference_ms,
        "p95_inference_ms": p95_inference_ms,
        "confidence_threshold": conf,
        "n_test_images": len(y_true),
    }

    metrics_path = output_dir / "image_level_metrics.csv"

    with open(
        metrics_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])

        for key, value in metrics.items():
            writer.writerow([key, value])

    # ---------------------------------------------------------
    # Save confusion matrix
    # ---------------------------------------------------------

    cm_path = output_dir / "confusion_matrix.csv"

    np.savetxt(
        cm_path,
        cm,
        delimiter=",",
        fmt="%d",
    )

    # ---------------------------------------------------------
    # Save confusion matrix figure
    # ---------------------------------------------------------

    figure_path = output_dir / "confusion_matrix.png"

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=["non-PTC", "PTC"],
        yticklabels=["non-PTC", "PTC"],
    )

    plt.title("PTC Classification Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(
        figure_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # ---------------------------------------------------------
    # Final output
    # ---------------------------------------------------------

    print("\nSaved:")
    print(f"  {csv_path}")
    print(f"  {metrics_path}")
    print(f"  {cm_path}")
    print(f"  {figure_path}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate YOLO PTC model on the held-out test set."
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to best.pt",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Root directory of YOLO dataset.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="CUDA device such as 0, or cpu.",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for prediction.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="runs/final/test_metrics",
        help="Directory for test results.",
    )

    args = parser.parse_args()

    image_level_evaluation(
        model_path=args.model,
        dataset_root=args.dataset,
        device=args.device,
        conf=args.conf,
        output_dir=args.output,
    )